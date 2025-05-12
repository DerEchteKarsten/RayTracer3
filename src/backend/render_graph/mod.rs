use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
};

use anyhow::Result;
use ash::vk::{self, Format, ImageUsageFlags};
use build::ImageSize;
use derivative::Derivative;
use enum_dispatch::enum_dispatch;
use glam::UVec2;
use gpu_allocator::MemoryLocation;

use crate::raytracing::AccelerationStructure;
use build::{ComputePass, RasterPass, RayTracingPass};

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    vulkan::{
        buffer::{Buffer, BufferHandle},
        image::{Image, ImageHandle, ImageType},
        swapchain::{Swapchain, FRAMES_IN_FLIGHT},
        Context,
    },
};

pub mod bake;
pub mod build;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ResourceHandle(u32);

pub const IMPORTED: NodeHandle = !0;

#[derive(PartialEq, Eq)]
enum ResourceType {
    Image(Image),
    Buffer(Buffer),
    ExternalDescriptor,
}

impl ResourceType {
    fn buffer(&self) -> &Buffer {
        match self {
            Self::Buffer(buffer) => buffer,
            _ => unreachable!(),
        }
    }
    fn image(&self) -> &Image {
        match self {
            Self::Image(image) => image,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct Barrier {
    resource: ResourceHandle,
    layout: vk::ImageLayout,
    access: vk::AccessFlags2,
    stages: vk::PipelineStageFlags2,
}

impl Barrier {
    fn need_invalidate(&self, event: &Event) -> bool {
        (0..64)
            .map(|i| {
                self.access.contains(
                    event.invalidated_in_stage[((self.stages.as_raw() >> i) & 1) as usize / 2],
                )
            })
            .fold(false, |acc, a| acc || a)
    }
}

impl Barrier {
    fn new(resource: ResourceHandle) -> Self {
        Self {
            resource,
            layout: vk::ImageLayout::UNDEFINED,
            access: vk::AccessFlags2::empty(),
            stages: vk::PipelineStageFlags2::empty(),
        }
    }
}

#[derive(Debug)]
struct Barriers {
    invalidates: Vec<Barrier>,
    flushes: Vec<Barrier>,
}

#[derive(PartialEq, Eq)]
struct Resource {
    event: Event,
    descriptor: DescriptorResourceHandle,
    ty: ResourceType,
}

impl Resource {
    fn new(descriptor: DescriptorResourceHandle, ty: ResourceType) -> Self {
        // let event = unsafe {
        //     Context::get()
        //         .device
        //         .create_event(&vk::EventCreateInfo::default(), None)
        //         .unwrap()
        // };
        // Context::get().set_debug_name(&format!("EventFor{}", descriptor.index()), event);
        Self {
            descriptor,
            ty,
            event: Event {
                // event,
                invalidated_in_stage: [vk::AccessFlags2::empty(); 25],
                pipeline_barrier_src_stages: vk::PipelineStageFlags2::empty(),
                to_flush: vk::AccessFlags2::default(),
                layout: vk::ImageLayout::UNDEFINED,
            },
        }
    }
}

type NodeHandle = usize;

#[enum_dispatch]
pub trait ExecutionTrait {
    fn execute(&self, cmd: &vk::CommandBuffer, rg: &RenderGraph, edges: &[NodeEdge]) -> Result<()>;
    fn get_stages(&self) -> vk::PipelineStageFlags2;
}

#[enum_dispatch(ExecutionTrait)]
#[derive(PartialEq)]
enum Execution {
    RayTracingPass,
    ComputePass,
    RasterPass,
}

#[derive(PartialEq)]
struct Node {
    name: &'static str,
    execution: Execution,
    constant_offset: Option<u32>,
    edges: Vec<NodeEdge>,
}

impl Node {
    fn parents<'b>(&self) -> Vec<NodeHandle> {
        self.edges
            .iter()
            .filter_map(|r| r.origin)
            .collect::<Vec<_>>()
    }

    fn bindings<'b>(
        &'b self,
    ) -> std::iter::Filter<std::slice::Iter<'b, NodeEdge>, impl FnMut(&&'b NodeEdge) -> bool> {
        self.edges.iter().filter(|e| {
            e.edge_type != EdgeType::ColorAttachmentOutput
                && e.edge_type != EdgeType::DepthAttachment
                && e.edge_type != EdgeType::StencilAttachment
        })
    }

    fn cmd_push_constants(&self, rg: &RenderGraph, frame: &FrameData, descriptor_offset: u32) {
        let ctx = Context::get();
        unsafe {
            let mut constants = [0u8; 16];
            constants[0..4].copy_from_slice(&self.constant_offset.unwrap_or(0).to_ne_bytes());

            constants[4..8].copy_from_slice(
                &if self.bindings().count() == 0 {
                    0
                } else {
                    descriptor_offset
                }
                .to_ne_bytes(),
            );
            constants[8..12]
                .copy_from_slice(&rg.descriptor_buffer.descriptor.0.to_ne_bytes());
            constants[12..16]
                .copy_from_slice(&rg.constants_buffer.descriptor.0.to_ne_bytes());

            ctx.device.cmd_push_constants(
                frame.cmd,
                BindlessDescriptorHeap::get().layout,
                vk::ShaderStageFlags::ALL,
                0,
                &constants,
            )
        };
    }

    fn get_barriers(&self, rg: &RenderGraph) -> Barriers {
        let mut invalidates: HashMap<ResourceHandle, Barrier> = HashMap::new();
        let mut flushes: HashMap<ResourceHandle, Barrier> = HashMap::new();

        for edge in &self.edges {
            match edge.edge_type {
                EdgeType::ShaderRead => {
                    let barrier = invalidates
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    barrier.stages |= self.execution.get_stages();
                    if let Some(image) = rg.image_handle(edge.resource)
                        && image.usage.contains(vk::ImageUsageFlags::STORAGE)
                    {
                        barrier.access |= vk::AccessFlags2::SHADER_STORAGE_READ;
                        barrier.layout = vk::ImageLayout::GENERAL;
                    } else {
                        barrier.access |= vk::AccessFlags2::SHADER_SAMPLED_READ;
                        barrier.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    }
                }
                EdgeType::ColorAttachmentOutput => {
                    let barrier = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    barrier.access |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
                    barrier.stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
                    barrier.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                }
                EdgeType::DepthAttachment | EdgeType::StencilAttachment => {
                    let src = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    let dst = invalidates
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    dst.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    dst.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    dst.stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;

                    src.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    src.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    dst.stages |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
                }
                EdgeType::ShaderReadWrite => {
                    let flush = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    flush.stages |= self.execution.get_stages();
                    flush.access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
                    flush.layout = vk::ImageLayout::GENERAL;

                    let invalidate = invalidates
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    invalidate.stages |= self.execution.get_stages();
                    if let Some(image) = rg.image_handle(edge.resource)
                        && image.usage.contains(vk::ImageUsageFlags::STORAGE)
                    {
                        invalidate.access |= vk::AccessFlags2::SHADER_STORAGE_READ;
                        invalidate.layout = vk::ImageLayout::GENERAL;
                    } else {
                        invalidate.access |= vk::AccessFlags2::SHADER_SAMPLED_READ;
                        invalidate.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    }
                }
                EdgeType::ShaderWrite => {
                    let flush = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    flush.stages |= self.execution.get_stages();
                    flush.access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
                    flush.layout = vk::ImageLayout::GENERAL;

                    if !invalidates.contains_key(&edge.resource) && rg.swapchain_images[rg.swapchain_image_index] == flush.resource {
                        invalidates.insert(
                            edge.resource,
                            Barrier {
                                resource: edge.resource,
                                layout: vk::ImageLayout::GENERAL,
                                access: vk::AccessFlags2::NONE,
                                stages: self.execution.get_stages(),
                            },
                        );
                    }
                }
                EdgeType::TransferDst => {
                    todo!();
                }
                EdgeType::TransferSrc => {
                    todo!();
                }
            }
        }

        Barriers {
            invalidates: invalidates.into_values().collect::<Vec<_>>(),
            flushes: flushes.into_values().collect::<Vec<_>>(),
        }
    }
}

pub fn depends_on(rg: &RenderGraph, other: NodeHandle, s: NodeHandle) -> bool {
    other == s
        || rg.nodes[other]
            .edges
            .iter()
            .find(|e| {
                if let Some(origin) = e.origin
                    && origin == s
                {
                    true
                } else {
                    false
                }
            })
            .is_some()
}

#[derive(Clone)]
struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Event {
    // event: vk::Event,
    pipeline_barrier_src_stages: vk::PipelineStageFlags2,
    to_flush: vk::AccessFlags2,
    invalidated_in_stage: [vk::AccessFlags2; 25],
    layout: vk::ImageLayout,
}

pub struct RenderGraph {
    resources: Vec<Resource>,

    constants_buffer: Resource,
    descriptor_buffer: Resource,

    nodes: Vec<Node>,
    constants: Vec<(*const c_void, usize)>,

    frame_data: [FrameData; FRAMES_IN_FLIGHT],
    frame_timeline_semaphore: vk::Semaphore,
    swapchain: Swapchain,
    swapchain_images: Vec<ResourceHandle>,
    swapchain_image_index: usize,
    pub frame_number: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum EdgeType {
    ShaderRead,
    ShaderReadWrite,
    ShaderWrite,
    ColorAttachmentOutput,
    DepthAttachment,
    StencilAttachment,
    TransferSrc,
    TransferDst,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NodeEdge {
    edge_type: EdgeType,
    origin: Option<NodeHandle>,
    resource: ResourceHandle,
}

trait Importable {
    fn resource(self) -> Resource;
}

impl Importable for Buffer {
    fn resource(self) -> Resource {
        let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&self);
        Resource::new(descriptor, ResourceType::Buffer(self))
    }
}
impl Importable for BufferHandle {
    fn resource(self) -> Resource {
        let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&self);
        Resource::new(
            descriptor,
            ResourceType::Buffer(Buffer {
                address: self.address,
                allocation: None,
                buffer: self.buffer,
                size: self.size,
                usage: self.usage,
            }),
        )
    }
}
impl Importable for Image {
    fn resource(self) -> Resource {
        let descriptor = if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            BindlessDescriptorHeap::get_mut().allocate_image_handle(&self)
        } else {
            BindlessDescriptorHeap::get_mut().allocate_texture_handle(&self)
        };
        Resource::new(descriptor, ResourceType::Image(self))
    }
}
impl Importable for ImageHandle {
    fn resource(self) -> Resource {
        let descriptor = if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            BindlessDescriptorHeap::get_mut().allocate_image_handle(&self)
        } else {
            BindlessDescriptorHeap::get_mut().allocate_texture_handle(&self)
        };
        Resource::new(
            descriptor,
            ResourceType::Image(Image {
                allocation: None,
                extent: self.extent,
                format: self.format,
                image: self.image,
                usage: self.usage,
                view: self.view,
            }),
        )
    }
}
impl Importable for DescriptorResourceHandle {
    fn resource(self) -> Resource {
        Resource::new(self, ResourceType::ExternalDescriptor)
    }
}

impl RenderGraph {
    pub fn new() -> Self {
        let ctx = Context::get();
        let mut frame_data: [FrameData; FRAMES_IN_FLIGHT] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        let mut timeline_create_info = vk::SemaphoreTypeCreateInfo::default()
            .initial_value(FRAMES_IN_FLIGHT as u64 - 1)
            .semaphore_type(vk::SemaphoreType::TIMELINE);

        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_create_info);
        let frame_timeline_semaphore =
            unsafe { ctx.device.create_semaphore(&create_info, None).unwrap() };

        for i in 0..FRAMES_IN_FLIGHT {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.graphics_queue_family.index);

            let command_pool = unsafe {
                ctx.device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap()
            };

            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY);
            let command_buffers =
                unsafe { ctx.device.allocate_command_buffers(&allocate_info).unwrap() };

            frame_data[i] = FrameData {
                cmd: command_buffers[0],
                frame_number: i as u64,
                command_pool,
            }
        }
        let mut resources = Vec::new();
        let swapchain = Swapchain::new().unwrap();
        let swapchain_images = swapchain
            .images
            .iter()
            .map(|image| {
                let descriptor = BindlessDescriptorHeap::get_mut().allocate_image_handle(image);
                let index = resources.len();
                resources.push(Resource::new(
                    descriptor,
                    ResourceType::Image(image.clone()),
                ));
                ResourceHandle(index as u32)
            })
            .collect::<Vec<_>>();
        let descriptor_buffer = {
            let buffer = Buffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                size_of::<u32>() as u64 * 256,
            )
            .unwrap();
            let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&buffer);
            Resource::new(descriptor, ResourceType::Buffer(buffer))
        };

        let constants_buffer = {
            let buffer = Buffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                size_of::<u32>() as u64 * 1024,
            )
            .unwrap();
            let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&buffer);
            Resource::new(descriptor, ResourceType::Buffer(buffer))
        };

        Self {
            constants: Vec::new(),
            nodes: Vec::new(),
            swapchain,
            frame_data,
            frame_number: 0,
            resources,
            frame_timeline_semaphore,
            swapchain_images,
            constants_buffer,
            descriptor_buffer,
            swapchain_image_index: 0,
        }
    }

    pub fn get_swapchain(&self) -> ResourceHandle {
        self.swapchain_images[self.swapchain_image_index]
    }

    fn import<T>(&mut self, value: T) -> ResourceHandle
    where
        T: Importable,
    {
        let index = self.resources.len();
        self.resources.push(value.resource());
        ResourceHandle(index as u32)
    }

    pub fn buffer(&mut self, size: u64, usage: vk::BufferUsageFlags, name: &str) -> ResourceHandle {
        let buffer = Buffer::new(usage, MemoryLocation::GpuOnly, size).unwrap();
        Context::get().set_debug_name(name, buffer.buffer);
        self.import(buffer)
    }

    pub fn image(
        &mut self,
        size: ImageSize,
        usage: vk::ImageUsageFlags,
        format: Format,
        name: &str,
    ) -> ResourceHandle {
        let size = size.size();
        let image = Image::new_2d(usage, MemoryLocation::GpuOnly, format, size.x, size.y).unwrap();
        Context::get().set_debug_name(name, image.image);
        self.import(image)
    }

    fn image_handle<'a>(&'a self, handle: ResourceHandle) -> Option<ImageHandle> {
        if let ResourceType::Image(image) = &self.resources[handle.0 as usize].ty {
            Some(image.handle())
        } else {
            None
        }
    }
    fn buffer_handle<'a>(&'a self, handle: ResourceHandle) -> Option<BufferHandle> {
        if let ResourceType::Buffer(buffer) = &self.resources[handle.0 as usize].ty {
            Some(buffer.handle())
        } else {
            None
        }
    }
}
