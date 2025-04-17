use std::{collections::HashMap, ffi::c_void};

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
        image::{Image, ImageHandle},
        swapchain::{Swapchain, FRAMES_IN_FLIGHT},
        Context,
    },
};

pub mod bake;
pub mod build;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ResourceHandle(u32);

#[derive(PartialEq)]
enum ResourceMemoryType {
    ImportedBuffer = 0,
    ImportedImage = 1,
    Buffer = 2,
    Image = 3,
    ExternalReadonly = 4,
}

pub const IMPORTED: NodeHandle = !0;

impl ResourceHandle {
    const fn new(ty: ResourceMemoryType, index: usize) -> Self {
        Self((ty as u8 as u32) << 24u32 | index as u32)
    }
    fn index(&self) -> usize {
        (self.0 & !(0xff << 24)) as usize
    }
    fn ty(&self) -> ResourceMemoryType {
        match self.0 >> 24 {
            0 => ResourceMemoryType::ImportedBuffer,
            1 => ResourceMemoryType::ImportedImage,
            2 => ResourceMemoryType::Buffer,
            3 => ResourceMemoryType::Image,
            4 => ResourceMemoryType::ExternalReadonly,
            _ => unreachable!(),
        }
    }
}

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

#[derive(PartialEq, Eq)]
struct Resource {
    descriptor: DescriptorResourceHandle,
    ty: ResourceType,
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
    execution: Execution,
    edges: Vec<NodeEdge>,
    constants: Option<Constant>,
}

#[derive(PartialEq, Eq, Clone)]
struct Constant {
    pointer: *const c_void,
    size: usize,
}

impl Node {
    fn parents<'b>(&self) -> Vec<NodeHandle> {
        self.edges
            .iter()
            .filter_map(|r| r.origin)
            .collect::<Vec<_>>()
    }

    fn depends_on(&self, rg: &RenderGraph, other: &Node) -> bool {
        let other_index = rg.graph.element_offset(other).unwrap();
        other == self
            || self
                .edges
                .iter()
                .find(|e| {
                    if let Some(origin) = e.origin
                        && origin == other_index
                        && (e.edge_type == EdgeType::ShaderRead
                            || e.edge_type == EdgeType::ShaderReadWrite)
                    {
                        true
                    } else {
                        false
                    }
                })
                .is_some()
    }

    fn bindings<'b>(
        &'b self,
    ) -> std::iter::Filter<std::slice::Iter<'b, NodeEdge>, impl FnMut(&&'b NodeEdge) -> bool> {
        self.edges.iter().filter(|e| {
            e.edge_type != EdgeType::ColorAttachment
                && e.edge_type != EdgeType::DepthAttachment
                && e.edge_type != EdgeType::StencilAttachment
        })
    }
}

#[derive(Clone)]
struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

pub struct RenderGraph {
    resources: Vec<Resource>,

    constants_buffer: ResourceHandle,
    descriptor_buffer: ResourceHandle,

    graph: Vec<Node>,
    constants_cache: Vec<(Constant, usize)>,

    frame_data: [FrameData; FRAMES_IN_FLIGHT],
    frame_timeline_semaphore: vk::Semaphore,
    swapchain: Swapchain,
    swapchain_images: Vec<ResourceHandle>,
    swapchain_image_index: usize,
    pub frame_number: u64,
}

#[derive(Clone, PartialEq)]
enum EdgeType {
    ShaderRead,
    ShaderReadWrite,
    ShaderWrite,
    ColorAttachment,
    DepthAttachment,
    StencilAttachment,
    TransferSrc,
    TransferDst,
}

impl EdgeType {
    pub fn access_flags(&self) -> vk::AccessFlags2 {
        match self {
            EdgeType::ShaderRead => vk::AccessFlags2::SHADER_READ,
            EdgeType::ShaderReadWrite => {
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE
            }
            EdgeType::ShaderWrite => vk::AccessFlags2::SHADER_WRITE,
            EdgeType::ColorAttachment => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            EdgeType::DepthAttachment => {
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            EdgeType::StencilAttachment => {
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            EdgeType::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            EdgeType::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct NodeEdge {
    edge_type: EdgeType,
    origin: Option<NodeHandle>,
    resource: ResourceHandle,
    layout: Option<vk::ImageLayout>,
}

trait Importable {
    fn resource(self) -> Resource;
}

impl Importable for Buffer {
    fn resource(self) -> Resource {
        let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&self);
        Resource {
            descriptor: descriptor,
            ty: ResourceType::Buffer(self),
        }
    }
}
impl Importable for BufferHandle {
    fn resource(self) -> Resource {
        let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&self);
        Resource {
            descriptor: descriptor,
            ty: ResourceType::Buffer(Buffer {
                address: self.address,
                allocation: None,
                buffer: self.buffer,
                size: self.size,
                usage: self.usage,
            }),
        }
    }
}
impl Importable for Image {
    fn resource(self) -> Resource {
        let descriptor = BindlessDescriptorHeap::get_mut().allocate_image_handle(&self);
        Resource {
            descriptor: descriptor,
            ty: ResourceType::Image(self),
        }
    }
}
impl Importable for ImageHandle {
    fn resource(self) -> Resource {
        let descriptor = BindlessDescriptorHeap::get_mut().allocate_image_handle(&self);
        Resource {
            descriptor: descriptor,
            ty: ResourceType::Image(Image {
                allocation: None,
                extent: self.extent,
                format: self.format,
                image: self.image,
                usage: self.usage,
                view: self.view,
            }),
        }
    }
}
impl Importable for DescriptorResourceHandle {
    fn resource(self) -> Resource {
        Resource {
            descriptor: self,
            ty: ResourceType::ExternalDescriptor,
        }
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
                resources.push(Resource {
                    descriptor,
                    ty: ResourceType::Image(image.clone()),
                });
                ResourceHandle::new(ResourceMemoryType::ImportedImage, index)
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
            let index = resources.len();
            resources.push(Resource {
                descriptor,
                ty: ResourceType::Buffer(buffer),
            });
            ResourceHandle::new(ResourceMemoryType::Buffer, index)
        };

        let constants_buffer = {
            let buffer = Buffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                size_of::<u32>() as u64 * 1024,
            )
            .unwrap();
            let descriptor = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&buffer);
            let index = resources.len();
            resources.push(Resource {
                descriptor,
                ty: ResourceType::Buffer(buffer),
            });
            ResourceHandle::new(ResourceMemoryType::Buffer, index)
        };

        Self {
            constants_cache: Vec::new(),
            graph: Vec::new(),
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

    pub fn get_swapchain_format(&self) -> vk::Format {
        self.swapchain.format
    }

    fn import(&mut self, value: impl Importable) -> ResourceHandle {
        let index = self.resources.len();
        self.resources.push(value.resource());
        ResourceHandle::new(ResourceMemoryType::ExternalReadonly, index)
    }

    pub fn buffer(&mut self, size: u64, usage: vk::BufferUsageFlags) -> ResourceHandle {
        let buffer = Buffer::new(usage, MemoryLocation::GpuOnly, size).unwrap();
        self.import(buffer)
    }

    pub fn image(
        &mut self,
        size: ImageSize,
        usage: vk::ImageUsageFlags,
        format: Format,
    ) -> ResourceHandle {
        let size = size.size();
        let image = Image::new_2d(usage, MemoryLocation::GpuOnly, format, size.x, size.y).unwrap();
        self.import(image)
    }

    fn image_handle<'a>(&'a self, handle: ResourceHandle) -> Option<ImageHandle> {
        if let ResourceType::Image(image) = &self.resources[handle.index()].ty {
            Some(image.handle())
        } else {
            None
        }
    }
    fn buffer_handle<'a>(&'a self, handle: ResourceHandle) -> Option<BufferHandle> {
        if let ResourceType::Buffer(buffer) = &self.resources[handle.index()].ty {
            Some(buffer.handle())
        } else {
            None
        }
    }
}
