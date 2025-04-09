use std::collections::HashMap;

use anyhow::Result;
use ash::vk;
use build::ImageSize;
use derivative::Derivative;
use glam::UVec2;
use gpu_allocator::MemoryLocation;

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    vulkan::{
        swapchain::{Swapchain, FRAMES_IN_FLIGHT},
        utils::{Buffer, BufferHandle, ImageHandle, ImageResource},
        Context,
    },
};

pub mod bake;
pub mod build;

const IMPORTED: NodeHandle = !0;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ResourceMemoryHandle(u32);

#[derive(PartialEq)]
enum ResourceMemoryType {
    ImportedBuffer = 0,
    ImportedImage = 1,
    PercistentBuffer = 2,
    PercistentImage = 3,
    TransientBuffer = 4,
    TransientImage = 5,
    ExternalReadonly = 6,
}

impl ResourceMemoryHandle {
    fn new(ty: ResourceMemoryType, index: usize) -> Self {
        Self((ty as u8 as u32) << 24u32 | index as u32)
    }
    fn index(&self) -> usize {
        (self.0 & !(0xff << 24)) as usize
    }
    fn descriptor(&self) -> usize {
        match (self.0 >> 24) as usize {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 2,
            5 => 3,
            _ => unreachable!(),
        }
    }
    fn ty(&self) -> ResourceMemoryType {
        match self.0 >> 24 {
            0 => ResourceMemoryType::ImportedBuffer,
            1 => ResourceMemoryType::ImportedImage,
            2 => ResourceMemoryType::PercistentBuffer,
            3 => ResourceMemoryType::PercistentImage,
            4 => ResourceMemoryType::TransientBuffer,
            5 => ResourceMemoryType::TransientImage,
            6 => ResourceMemoryType::ExternalReadonly,
            _ => unreachable!(),
        }
    }
}

type NodeHandle = usize;

pub(crate) trait Execution {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()>;
    fn get_stages(&self) -> vk::PipelineStageFlags2;
}

struct Node {
    execution: Box<dyn Execution>,
    descriptor_buffer: BufferHandle,
    reads: Vec<NodeEdge>,
    writes: Vec<NodeEdge>,
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
}

#[derive(Clone)]
struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

impl FrameData {
    pub fn record_command_buffer<F>(
        &mut self,
        wait_semaphore: vk::Semaphore,
        swapchain: &Swapchain,
        frame_in_flight: usize,
        func: F,
    ) where
        F: FnOnce(vk::CommandBuffer),
    {
        let ctx = Context::get();
        let semaphore = [wait_semaphore];
        let values = [self.frame_number];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphore)
            .values(&values);
        unsafe { ctx.device.wait_semaphores(&wait_info, 1000000000).unwrap() };
        unsafe {
            ctx.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap()
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            ctx.device
                .begin_command_buffer(self.cmd, &begin_info)
                .unwrap()
        };

        func(self.cmd.clone());

        unsafe { ctx.device.end_command_buffer(self.cmd).unwrap() };

        let wait_semaphores = [vk::SemaphoreSubmitInfo::default()
            .semaphore(swapchain.frame_resources[frame_in_flight].image_availible_semaphore)
            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)];

        let signal_frame_value = self.frame_number + FRAMES_IN_FLIGHT as u64;
        self.frame_number = signal_frame_value;

        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    swapchain.frame_resources[frame_in_flight as usize].render_finished_semaphore,
                )
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(wait_semaphore)
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .value(signal_frame_value),
        ];

        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];

        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signal_semaphores)
            .wait_semaphore_infos(&wait_semaphores);

        unsafe {
            ctx.device
                .queue_submit2(ctx.graphics_queue, &[submit], vk::Fence::null())
                .unwrap()
        };
    }
}

pub struct RenderGraph {
    trainsient_resource_cache: HashMap<ResourceDescription, ResourceMemoryHandle>,

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
    pub frame_number: u64,
    output: Option<(NodeHandle, usize)>,
}

#[derive(Derivative, Clone, Debug)]
#[derivative(PartialEq, Eq, Hash)]
pub enum ResourceDescription {
    Image {
        index: usize,
        format: vk::Format,
        size: UVec2,
        usage: vk::ImageUsageFlags,
        #[derivative(PartialEq = "ignore")]
        #[derivative(Hash = "ignore")]
        layout: vk::ImageLayout,
    },
    Buffer {
        index: usize,
        size: u64,
        usage: vk::BufferUsageFlags,
    },
    PercistentBuffer {
        #[derivative(PartialEq = "ignore")]
        #[derivative(Hash = "ignore")]
        memory: ResourceMemoryHandle,
    },
    PercistentImage {
        #[derivative(PartialEq = "ignore")]
        #[derivative(Hash = "ignore")]
        layout: vk::ImageLayout,
        #[derivative(PartialEq = "ignore")]
        #[derivative(Hash = "ignore")]
        memory: ResourceMemoryHandle,
    },
}

impl ResourceDescription {
    pub fn eql(&self, other: &ResourceDescription) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }

        match self {
            Self::Buffer {
                index: _,
                size,
                usage,
            } => match other {
                Self::Buffer {
                    index: _,
                    size: s,
                    usage: u,
                } => *size == *s && *usage == *u,
                _ => unreachable!(),
            },
            Self::Image {
                index: _,
                format,
                size,
                usage,
                layout: _,
            } => match other {
                Self::Image {
                    index: _,
                    size: s,
                    usage: u,
                    format: f,
                    layout: _,
                } => *size == *s && *usage == *u && *format == *f,
                _ => unreachable!(),
            },
            _ => other == self,
        }
    }
    pub fn get_layout(&self) -> Option<vk::ImageLayout> {
        match self {
            Self::Image { layout, .. } => Some(*layout),
            Self::PercistentImage { layout, .. } => Some(*layout),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq)]
struct NodeEdge {
    output_of: NodeHandle,
    output_index: usize,
    description: ResourceDescription,
    name: String,
}

impl NodeEdge {
    fn get_parrent<'a>(&self, rg: &'a RenderGraph) -> &'a Node {
        &rg.graph[self.output_of]
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

        Self {
            trainsient_resource_cache: HashMap::new(),
            graph: Vec::new(),
            swapchain: Swapchain::new().unwrap(),
            frame_data,
            frame_number: 0,
            output: None,
            external_buffer_handles: Vec::new(),
            external_image_handles: Vec::new(),
            internal_buffers: Vec::new(),
            internal_images: Vec::new(),
            descriptor_handles: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            execution_order: Vec::new(),
            frame_timeline_semaphore,
        }
    }

    pub fn get_swapchain_format(&self) -> vk::Format {
        self.swapchain.format
    }

    pub fn set_output(&mut self, node: NodeHandle, index: usize) {
        self.output = Some((node, index));
    }

    pub fn import(&mut self, desc: DescriptorResourceHandle) -> ResourceMemoryHandle {
        let index = self.descriptor_handles[4].len();
        self.descriptor_handles[4].push(desc);
        ResourceMemoryHandle::new(ResourceMemoryType::ExternalReadonly, index)
    }

    pub fn import_storage_buffer(&mut self, buffer: BufferHandle) -> ResourceMemoryHandle {
        let index = self.external_buffer_handles.len();
        let memory = ResourceMemoryHandle::new(ResourceMemoryType::ImportedBuffer, index);
        self.descriptor_handles[memory.descriptor()]
            .push(BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&buffer));
        self.external_buffer_handles.push(buffer);
        memory
    }

    pub fn import_read_write_image(&mut self, image: ImageHandle) -> ResourceMemoryHandle {
        let index = self.external_image_handles.len();
        let memory = ResourceMemoryHandle::new(ResourceMemoryType::ImportedImage, index);
        self.descriptor_handles[memory.descriptor()]
            .push(BindlessDescriptorHeap::get_mut().allocate_storage_image_handle(&image));
        self.external_image_handles.push(image);
        memory
    }

    pub fn import_read_write_texture(&mut self, texture: ImageHandle) -> ResourceMemoryHandle {
        let index = self.external_image_handles.len();
        let memory = ResourceMemoryHandle::new(ResourceMemoryType::ImportedImage, index);
        self.descriptor_handles[memory.descriptor()]
            .push(BindlessDescriptorHeap::get_mut().allocate_texture_handle(&texture));
        self.external_image_handles.push(texture);
        memory
    }

    pub fn percistent_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> ResourceMemoryHandle {
        let buffer = Buffer::new(usage, gpu_allocator::MemoryLocation::GpuOnly, size).unwrap();
        let index = self.internal_buffers.len();
        self.internal_buffers.push(buffer);
        ResourceMemoryHandle::new(ResourceMemoryType::PercistentBuffer, index)
    }

    fn percistent_image(
        &mut self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        size: ImageSize,
    ) -> ResourceMemoryHandle {
        let size = size.size();
        let image =
            ImageResource::new_2d(usage, MemoryLocation::GpuOnly, format, size.x, size.y).unwrap();
        let index = self.internal_images.len();
        self.internal_images.push(image);
        ResourceMemoryHandle::new(ResourceMemoryType::PercistentImage, index)
    }

    fn get_handle(&self, desc: &ResourceDescription) -> ResourceMemoryHandle {
        match desc {
            ResourceDescription::PercistentBuffer { memory } => memory.clone(),
            ResourceDescription::PercistentImage { layout: _, memory } => memory.clone(),
            _ => self.trainsient_resource_cache[desc].clone(),
        }
    }

    fn image_handle<'a>(&'a self, handle: ResourceMemoryHandle) -> Option<ImageHandle> {
        match handle.ty() {
            ResourceMemoryType::ImportedImage => {
                Some(self.external_image_handles[handle.index()].clone())
            }
            ResourceMemoryType::PercistentImage => {
                Some(self.internal_images[handle.index()].handle())
            }
            ResourceMemoryType::TransientImage => {
                Some(self.internal_images[handle.index()].handle())
            }
            _ => None,
        }
    }
    fn buffer_handle<'a>(&'a self, handle: ResourceMemoryHandle) -> Option<BufferHandle> {
        match handle.ty() {
            ResourceMemoryType::ImportedBuffer => {
                Some(self.external_buffer_handles[handle.index()].clone())
            }
            ResourceMemoryType::PercistentBuffer => {
                Some(self.internal_buffers[handle.index()].handle())
            }
            ResourceMemoryType::TransientBuffer => {
                Some(self.internal_buffers[handle.index()].handle())
            }
            _ => None,
        }
    }
}
