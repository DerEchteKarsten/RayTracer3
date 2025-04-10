use std::{collections::HashMap, ffi::c_void};

use anyhow::Result;
use ash::vk::{self, Format};
use build::ImageSize;
use derivative::Derivative;
use glam::UVec2;
use gpu_allocator::MemoryLocation;

use crate::raytracing::AccelerationStructure;

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    vulkan::{
        swapchain::{Swapchain, FRAMES_IN_FLIGHT},
        utils::{Buffer, BufferHandle, Image, ImageHandle, ImageResource},
        Context,
    },
};

pub mod bake;
pub mod build;

pub const IMPORTED: NodeHandle = !0;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ResourceHandle(u32);

#[derive(PartialEq)]
enum ResourceMemoryType {
    ImportedBuffer = 0,
    ImportedImage = 1,
    Buffer = 2,
    Image = 3,
    ExternalReadonly = 4,
    Swapchain = 5,
}

impl ResourceHandle {
    const fn new(ty: ResourceMemoryType, index: usize) -> Self {
        Self((ty as u8 as u32) << 24u32 | index as u32)
    }
    fn index(&self) -> usize {
        (self.0 & !(0xff << 24)) as usize
    }
    fn descriptor(&self) -> usize {
        (self.0 >> 24) as usize
    }
    fn ty(&self) -> ResourceMemoryType {
        match self.0 >> 24 {
            0 => ResourceMemoryType::ImportedBuffer,
            1 => ResourceMemoryType::ImportedImage,
            2 => ResourceMemoryType::Buffer,
            3 => ResourceMemoryType::Image,
            4 => ResourceMemoryType::ExternalReadonly,
            5 => ResourceMemoryType::Swapchain,
            _ => unreachable!(),
        }
    }
}

pub const SWPACHAIN: ResourceHandle = ResourceHandle::new(ResourceMemoryType::Swapchain, 0xffffff);

type NodeHandle = usize;

pub(crate) trait Execution {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()>;
    fn get_stages(&self) -> vk::PipelineStageFlags2;
}

struct Node {
    execution: Box<dyn Execution>,
    descriptor_buffer: ResourceHandle,
    constants_buffer: Option<ResourceHandle>,
    reads: Vec<NodeEdge>,
    writes: Vec<NodeEdge>,
    constants: Option<*const c_void>,
    constants_size: usize,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.reads == other.reads && self.writes == other.writes
    }
}

impl Node {
    fn parents<'a>(&self) -> Vec<NodeHandle> {
        self.reads
            .iter()
            .filter_map(|r| {
                if r.output_of == IMPORTED {
                    None
                } else {
                    Some(r.output_of)
                }
            })
            .collect()
    }

    fn depends_on(&self, rg: &RenderGraph, other: &Node) -> bool {
        other == self
            || other
                .writes
                .iter()
                .find(|w| {
                    if let Some(other_index) = rg.graph.iter().position(|e| e == other)
                        && w.output_of == other_index
                    {
                        true
                    } else {
                        false
                    }
                })
                .is_some()
    }

    fn num_bindings(&self) -> usize {
        let mut bindings = self.reads.iter().map(|e| e.resource).collect::<Vec<_>>();
        self.writes.iter().for_each(|e| {
            if !bindings.contains(&e.resource) {
                bindings.push(e.resource);
            }
        });
        bindings.len() + self.constants.is_some() as usize
    }
}

#[derive(Clone)]
struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

pub struct RenderGraph {
    external_image_handles: Vec<ImageHandle>,
    external_buffer_handles: Vec<BufferHandle>,

    internal_images: Vec<ImageResource>,
    internal_buffers: Vec<Buffer>,

    descriptor_handles: [Vec<DescriptorResourceHandle>; 5],

    graph: Vec<Node>,
    execution_order: Vec<NodeHandle>,

    frame_data: [FrameData; FRAMES_IN_FLIGHT],
    frame_timeline_semaphore: vk::Semaphore,
    swapchain: Swapchain,
    swapchain_image_descriptors: Vec<DescriptorResourceHandle>,
    pub frame_number: u64,
}

#[derive(Clone, PartialEq)]
struct NodeEdge {
    output_of: NodeHandle,
    output_index: usize,
    resource: ResourceHandle,
    layout: Option<vk::ImageLayout>,
}

impl NodeEdge {
    fn get_parrent<'a>(&self, rg: &'a RenderGraph) -> &'a Node {
        &rg.graph[self.output_of]
    }
}

trait Importable {}

impl Importable for BufferHandle {}
impl Importable for AccelerationStructure {}
impl Importable for ImageHandle {}

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
        let swapchain = Swapchain::new().unwrap();
        let swapchain_image_descriptors = swapchain
            .images
            .iter()
            .map(|image| BindlessDescriptorHeap::get_mut().allocate_image_handle(&image.handle()))
            .collect::<Vec<_>>();

        Self {
            graph: Vec::new(),
            swapchain,
            frame_data,
            frame_number: 0,
            external_buffer_handles: Vec::new(),
            external_image_handles: Vec::new(),
            internal_buffers: Vec::new(),
            internal_images: Vec::new(),
            descriptor_handles: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            execution_order: Vec::new(),
            frame_timeline_semaphore,
            swapchain_image_descriptors,
        }
    }

    pub fn get_swapchain_format(&self) -> vk::Format {
        self.swapchain.format
    }

    fn import(&mut self, desc: DescriptorResourceHandle) -> ResourceHandle {
        let index = self.descriptor_handles[4].len();
        self.descriptor_handles[4].push(desc);
        ResourceHandle::new(ResourceMemoryType::ExternalReadonly, index)
    }
    pub fn import_tlas(&mut self, tlas: &AccelerationStructure) -> ResourceHandle {
        let desc = BindlessDescriptorHeap::get_mut().allocate_acceleration_structure_handle(tlas);
        self.import(desc)
    }
    pub fn import_image(&mut self, image: &ImageHandle) -> ResourceHandle {
        let desc = BindlessDescriptorHeap::get_mut().allocate_image_handle(image);
        self.import(desc)
    }
    pub fn import_buffer(&mut self, buffer: &BufferHandle) -> ResourceHandle {
        let desc = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(buffer);
        self.import(desc)
    }

    pub fn import_storage_buffer(&mut self, buffer: BufferHandle) -> ResourceHandle {
        let handle = self.buffer_resource(&buffer, false);
        self.external_buffer_handles.push(buffer);
        handle
    }
    pub fn buffer(&mut self, size: u64, usage: vk::BufferUsageFlags) -> ResourceHandle {
        self.internal_buffer(size, usage, MemoryLocation::GpuOnly)
    }
    
    fn internal_buffer(&mut self, size: u64, usage: vk::BufferUsageFlags, location: MemoryLocation) -> ResourceHandle {
        let buffer = Buffer::new(usage, location, size).unwrap();
        let handle = self.buffer_resource(&buffer.handle(), true);
        self.internal_buffers.push(buffer);
        handle
    }
    fn buffer_resource(&mut self, buffer: &BufferHandle, internal: bool) -> ResourceHandle {
        let index = if internal {self.internal_buffers.len()} else {self.external_buffer_handles.len()};
        let desc = BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&buffer);
        let handle = ResourceHandle::new(ResourceMemoryType::Buffer, index);
        self.descriptor_handles[handle.descriptor()].push(desc);
        handle
    }

    pub fn import_storage_image(&mut self, image: ImageHandle) -> ResourceHandle {
        let handle = self.image_resource(&image, false);
        self.external_image_handles.push(image);
        handle
    }
    pub fn image(&mut self, size: ImageSize, usage: vk::ImageUsageFlags, format: Format) -> ResourceHandle {
        self.internal_image(size, usage, format, MemoryLocation::GpuOnly)
    }
    
    fn internal_image(&mut self, size: ImageSize, usage: vk::ImageUsageFlags, format: Format, location: MemoryLocation) -> ResourceHandle {
        let size = size.size();
        let image = ImageResource::new_2d(usage, location, format, size.x, size.y).unwrap();
        let handle = self.image_resource(&image.handle(), true);
        self.internal_images.push(image);
        handle
    }
    fn image_resource(&mut self, image: &ImageHandle, internal: bool) -> ResourceHandle {
        let index = if internal {self.internal_images.len()} else {self.external_image_handles.len()};
        let desc = BindlessDescriptorHeap::get_mut().allocate_image_handle(&image);
        let handle = ResourceHandle::new(ResourceMemoryType::Image, index);
        self.descriptor_handles[handle.descriptor()].push(desc);
        handle
    }

    fn image_handle<'a>(&'a self, handle: ResourceHandle, swapchain_image: usize) -> Option<ImageHandle> {
        match handle.ty() {
            ResourceMemoryType::ImportedImage => {
                Some(self.external_image_handles[handle.index()].clone())
            }
            ResourceMemoryType::Image => Some(self.internal_images[handle.index()].handle()),
            ResourceMemoryType::Swapchain => Some(self.swapchain.images[swapchain_image].handle()),
            _ => None,
        }
    }
    fn buffer_handle<'a>(&'a self, handle: ResourceHandle) -> Option<BufferHandle> {
        match handle.ty() {
            ResourceMemoryType::ImportedBuffer => {
                Some(self.external_buffer_handles[handle.index()].clone())
            }
            ResourceMemoryType::Buffer => Some(self.internal_buffers[handle.index()].handle()),
            _ => None,
        }
    }
}
