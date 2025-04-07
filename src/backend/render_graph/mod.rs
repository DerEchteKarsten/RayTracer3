use std::collections::HashMap;

use anyhow::Result;
use ash::vk;
use derivative::Derivative;
use glam::UVec2;
use gpu_allocator::MemoryLocation;

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    utils::{Buffer, ImageResource},
    vulkan_context::Context,
};

mod bake;
mod build;

pub(crate) const FRAMES_IN_FLIGHT: usize = 3;

#[derive(Clone)]
enum ResourceOrigin {
    Imported {
        name: String,
    },
    RenderGraphTransient {
        output_of: NodeHandle,
        output_index: usize,
    },
    RenderGraphPercistent {
        memory: ResourceMemoryHandle,
        output_of: NodeHandle,
        output_index: usize,
    },
}

#[derive(Clone)]
struct ResourceMemoryHandle(u32);

#[derive(PartialEq)]
#[repr(u8)]
enum ResourceMemoryType {
    Buffer = 0,
    Image = 1,
}

impl ResourceMemoryHandle {
    fn new(ty: ResourceMemoryType, index: usize) -> Self {
        Self((ty as u8 as u32) << 31u32 | index as u32)
    }
    fn index(&self) -> usize {
        (self.0 & !(1 << 31)) as usize
    }
    fn ty(&self) -> ResourceMemoryType {
        match self.0 >> 31 {
            0 => ResourceMemoryType::Buffer,
            1 => ResourceMemoryType::Image,
            _ => unreachable!(),
        }
    }
}

type NodeHandle = usize;

struct Resource {
    descriptor: DescriptorResourceHandle,
    memory: ResourceMemoryHandle,
}

trait Execution {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()>;
}

struct Node {
    name: String,
    execution: Box<dyn Execution>,
    reads: Vec<NodeEdge>,
    writes: Vec<NodeEdge>,
}

struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

pub struct RenderGraph {
    trainsient_resource_cache: HashMap<ResourceDescription, Vec<Resource>>,
    imported_resources: HashMap<String, DescriptorResourceHandle>,
    buffer_memory: Vec<Buffer>,
    image_memory: Vec<ImageResource>,

    graph: Vec<Node>,

    frame_data: [FrameData; FRAMES_IN_FLIGHT],
    pub frame_number: u64,
    output: Option<NodeEdge>,
}

#[derive(Derivative, Clone, Copy, Debug)]
#[derivative(PartialEq, Hash)]
enum ResourceDescription {
    Image {
        format: vk::Format,
        size: UVec2,
        usage: vk::ImageUsageFlags,
        #[derivative(PartialEq = "ignore")]
        #[derivative(Hash = "ignore")]
        layout: vk::ImageLayout,
    },
    Buffer {
        size: u64,
        usage: vk::BufferUsageFlags,
    },
    PercistentBuffer,
    PercistentImage {
        #[derivative(PartialEq = "ignore")]
        #[derivative(Hash = "ignore")]
        layout: vk::ImageLayout,
    },
}

impl ResourceDescription {
    fn ty(&self) -> ResourceMemoryType {
        match self {
            ResourceDescription::Buffer { size: _, usage: _ } => ResourceMemoryType::Buffer,
            ResourceDescription::Image {
                format: _,
                layout: _,
                size: _,
                usage: _,
            } => ResourceMemoryType::Image,
            ResourceDescription::PercistentBuffer => ResourceMemoryType::Buffer,
            ResourceDescription::PercistentImage { layout: _ } => ResourceMemoryType::Image,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct NodeEdge {
    origin: ResourceOrigin,
    description: ResourceDescription,
}

impl RenderGraph {
    pub fn new() -> Self {
        let ctx = Context::get();
        let mut frame_data: [FrameData; FRAMES_IN_FLIGHT] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

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
            imported_resources: HashMap::new(),
            buffer_memory: Vec::new(),
            image_memory: Vec::new(),
            graph: Vec::new(),
            frame_data,
            frame_number: 0,
            output: None,
        }
    }

    pub fn set_output(&mut self, handle: &NodeEdge) {
        self.output = Some(handle.clone());
    }

    pub fn import_buffer(&mut self, name: &str, buffer: Buffer) -> NodeEdge {
        self.imported_resources.insert(
            name.to_string(),
            BindlessDescriptorHeap::get_mut().allocate_buffer_handle(&buffer),
        );

        NodeEdge {
            origin: ResourceOrigin::Imported {
                name: name.to_string(),
            },
            description: ResourceDescription::PercistentBuffer,
        }
    }

    pub fn import_storage_image(&mut self, name: &str, image: &ImageResource) -> NodeEdge {
        self.imported_resources.insert(
            name.to_string(),
            BindlessDescriptorHeap::get_mut().allocate_storage_image_handle(image),
        );

        NodeEdge {
            origin: ResourceOrigin::Imported {
                name: name.to_string(),
            },
            description: ResourceDescription::PercistentImage {
                layout: vk::ImageLayout::GENERAL,
            },
        }
    }

    pub fn import_texture(&mut self, name: &str, image: &ImageResource) -> NodeEdge {
        self.imported_resources.insert(
            name.to_string(),
            BindlessDescriptorHeap::get_mut().allocate_texture_handle(image),
        );

        NodeEdge {
            origin: ResourceOrigin::Imported {
                name: name.to_string(),
            },
            description: ResourceDescription::PercistentImage {
                layout: vk::ImageLayout::GENERAL,
            },
        }
    }

    pub fn percistent_buffer(&mut self, size: u64, usage: vk::BufferUsageFlags) -> NodeEdge {
        let memory = self.allocate_buffer(size, usage);
        NodeEdge {
            origin: ResourceOrigin::RenderGraphPercistent {
                memory,
                output_of: 0,
                output_index: 0,
            },
            description: ResourceDescription::PercistentImage {
                layout: vk::ImageLayout::GENERAL,
            },
        }
    }
    fn allocate_buffer(&mut self, size: u64, usage: vk::BufferUsageFlags) -> ResourceMemoryHandle {
        let buffer = Buffer::new(usage, gpu_allocator::MemoryLocation::GpuOnly, size).unwrap();
        let index = self.buffer_memory.len();
        self.buffer_memory.push(buffer);
        ResourceMemoryHandle::new(ResourceMemoryType::Buffer, index)
    }

    pub fn percistent_image(&mut self, size: u64, usage: vk::BufferUsageFlags) -> NodeEdge {
        let memory = self.allocate_buffer(size, usage);
        NodeEdge {
            origin: ResourceOrigin::RenderGraphPercistent {
                memory,
                output_of: 0,
                output_index: 0,
            },
            description: ResourceDescription::PercistentImage {
                layout: vk::ImageLayout::GENERAL,
            },
        }
    }
    fn allocate_image(
        &mut self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        size: UVec2,
    ) -> ResourceMemoryHandle {
        let image =
            ImageResource::new_2d(usage, MemoryLocation::GpuOnly, format, size.x, size.y).unwrap();
        let index = self.image_memory.len();
        self.image_memory.push(image);
        ResourceMemoryHandle::new(ResourceMemoryType::Image, index)
    }

    fn image<'a>(&'a self, handle: ResourceMemoryHandle) -> Option<&'a ImageResource> {
        if handle.ty() == ResourceMemoryType::Image {
            Some(&self.image_memory[handle.index()])
        } else {
            None
        }
    }
    fn buffer<'a>(&'a self, handle: ResourceMemoryHandle) -> Option<&'a Buffer> {
        if handle.ty() == ResourceMemoryType::Buffer {
            Some(&self.buffer_memory[handle.index()])
        } else {
            None
        }
    }
}
