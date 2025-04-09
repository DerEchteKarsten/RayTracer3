use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use anyhow::{Error, Result};
use ash::vk;
use glam::{UVec2, UVec3};
use gpu_allocator::MemoryLocation;
use winit::platform::x11;

use crate::backend::bindless::{BindlessDescriptorHeap, DescriptorResourceHandle};
use crate::backend::pipeline_cache::{
    ComputePipelineHandle, PipelineCache, RayTracingPipelineHandle,
};
use crate::raytracing::{RayTracingContext, RayTracingShaderCreateInfo, ShaderBindingTable};
use crate::vulkan::utils::BufferHandle;
use crate::vulkan::Context;

use crate::backend::vulkan::utils::{Buffer, Image, ImageResource};
use crate::WINDOW_SIZE;

use derivative::Derivative;

use super::{
    Execution, Node, NodeEdge, NodeHandle, RenderGraph, ResourceDescription, ResourceMemoryHandle, ResourceMemoryType, FRAMES_IN_FLIGHT, IMPORTED
};

pub struct NodeBuilder<'a, T>
where
    T: Execution + 'static,
{
    rg: &'a mut RenderGraph,
    execution: MaybeUninit<T>,
    reads: Vec<NodeEdge>,
    writes: Vec<NodeEdge>,
    handle: NodeHandle,
}

impl<'b, T> NodeBuilder<'b, T>
where
    T: Execution + 'static,
{
    fn new<E: Execution + 'static>(rg: &'b mut RenderGraph) -> NodeBuilder<'b, E> {
        let handle = rg.graph.len();
        NodeBuilder::<'b, E> {
            rg,
            handle,
            execution: MaybeUninit::uninit(),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    pub(crate) fn read(mut self, node: NodeHandle, name: &str) -> Self {
        let handle = self.rg.graph[node].writes.iter().find(|e| e.name == name).unwrap().clone();
        self.reads.push(handle);
        self
    }

    pub(crate) fn read_index(mut self, node: NodeHandle, index: usize) -> Self {
        let handle = self.rg.graph[node].writes[index].clone();
        self.reads.push(handle);
        self
    }

    pub(crate) fn read_external(mut self, memory: &ResourceMemoryHandle) -> Self {
        let handle = NodeEdge {
            output_of: IMPORTED,
            output_index: 0,
            description: ResourceDescription::PercistentBuffer {
                memory: memory.clone(),
            },
            name: "".to_string(),
        };
        self.reads.push(handle);
        self
    }

    pub(crate) fn read_external_image(
        mut self,
        memory: &ResourceMemoryHandle,
        layout: vk::ImageLayout,
    ) -> Self {
        let handle = NodeEdge {
            output_of: IMPORTED,
            output_index: 0,
            description: ResourceDescription::PercistentImage {
                layout,
                memory: memory.clone(),
            },
            name: "".to_string(),
        };
        self.reads.push(handle);
        self
    }

    fn write(mut self, description: ResourceDescription, name: &str) -> Self {
        let output_index = self.writes.len();
        self.writes.push(NodeEdge {
            output_of: self.handle,
            output_index,
            description: description,
            name: name.to_string(),
        });
        self
    }

    pub(crate) fn write_image(self, name: &str, size: ImageSize, format: vk::Format) -> Self {
        let mut description = ResourceDescription::Image {
            format,
            size: size.size(),
            usage: vk::ImageUsageFlags::STORAGE,
            layout: vk::ImageLayout::GENERAL,
            index: 0,
        };
        let i = self
            .writes
            .iter()
            .chain(self.writes.iter())
            .fold(0, |acc, e| {
                if description.eql(&e.description) {
                    acc + 1
                } else {
                    acc
                }
            });
        if let ResourceDescription::Image { index, .. } = &mut description {
            *index = i;
        }
        self.write(description, name)
    }
    pub(crate) fn write_buffer(self, name: &str, size: u64) -> Self {
        let mut description = ResourceDescription::Buffer {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            index: 0,
        };
        let i = self
            .writes
            .iter()
            .chain(self.reads.iter())
            .fold(0, |acc, e| {
                if description.eql(&e.description) {
                    acc + 1
                } else {
                    acc
                }
            });
        if let ResourceDescription::Buffer { index, .. } = &mut description {
            *index = i;
        }
        self.write(description, name)
    }
    pub(crate) fn write_to_buffer(self, handle: ResourceMemoryHandle, name: &str) -> Self {
        self.write(ResourceDescription::PercistentBuffer { memory: handle }, name)
    }
    pub(crate) fn write_to_image(
        self,
        handle: ResourceMemoryHandle,
        layout: vk::ImageLayout,
        name: &str
    ) -> Self {
        if handle.ty() == ResourceMemoryType::ExternalReadonly {
            panic!("attempting to write to readonly descriptor")
        }
        self.write(ResourceDescription::PercistentImage {
            memory: handle,
            layout,
        }, name)
    }
}

pub enum ImageSize {
    FullScreen,
    FractionalFullScreen(u32, u32),
    XY(u32, u32),
}

impl ImageSize {
    pub fn size(self) -> UVec2 {
        match self {
            Self::FullScreen => UVec2::new(WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            Self::FractionalFullScreen(dx, dy) => UVec2::new(
                (WINDOW_SIZE.x as u32).div_ceil(dx),
                (WINDOW_SIZE.y as u32).div_ceil(dy),
            ),
            Self::XY(x, y) => UVec2::new(x, y),
        }
    }
}

#[derive(Default, Clone, Copy)]
pub enum DispatchSize {
    #[default]
    FullScreen,
    FractionalFullScreen(u32, u32),
    X(u32),
    XY(u32, u32),
    XYZ(u32, u32, u32),
    Custom(fn() -> UVec3),
}

impl DispatchSize {
    fn size(&self) -> (u32, u32, u32) {
        match self {
            DispatchSize::Custom(func) => {
                let res = func();
                (res.x, res.y, res.z)
            }
            DispatchSize::FractionalFullScreen(x, y) => (
                (WINDOW_SIZE.x as u32).div_ceil(*x),
                (WINDOW_SIZE.y as u32).div_ceil(*y),
                1,
            ),
            DispatchSize::FullScreen => (
                (WINDOW_SIZE.x as u32).div_ceil(8),
                (WINDOW_SIZE.y as u32).div_ceil(8),
                1,
            ),
            DispatchSize::X(x) => (*x, 1, 1),
            DispatchSize::XY(x, y) => (*x, *y, 1),
            DispatchSize::XYZ(x, y, z) => (*x, *y, *z),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ComputePass {
    pipeline: ComputePipelineHandle,
    dispatch: DispatchSize,
}

impl ComputePass {
    pub(crate) fn new<'a>(rg: &'a mut RenderGraph) -> NodeBuilder<'a, ComputePass> {
        let mut builder = NodeBuilder::<ComputePass>::new::<ComputePass>(rg);
        builder.execution = MaybeUninit::new(ComputePass {
            pipeline: ComputePipelineHandle {
                path: "".to_owned(),
                entry: "main".to_owned(),
            },
            dispatch: DispatchSize::FullScreen,
        });
        builder
    }
}

impl<'b> NodeBuilder<'b, ComputePass> {
    pub(crate) fn shader(mut self, path: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.path = path.to_string();
        self
    }
    pub(crate) fn entry(mut self, entry: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.entry = entry.to_string();
        self
    }
    pub(crate) fn dispatch(mut self, dispatch: DispatchSize) -> NodeHandle {
        unsafe { self.execution.assume_init_mut() }.dispatch = dispatch;
        self.rg.graph.push(Node {
            reads: self.reads,
            writes: self.writes,
            descriptor_buffer: BufferHandle::default(),
            execution: Box::new(unsafe { self.execution.assume_init() }),
        });
        self.handle
    }
}

impl Execution for ComputePass {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()> {
        let (x, y, z) = self.dispatch.size();
        self.pipeline.dispatch(cmd, x, y, z);
        Ok(())
    }
    fn get_stages(&self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::COMPUTE_SHADER
    }
}

#[derive(Default)]
pub enum LaunchSize {
    #[default]
    FullScreen,
    FractionalFullScreen(u32, u32),
    X(u32),
    XY(u32, u32),
    Custom(fn() -> UVec2),
}

impl LaunchSize {
    fn size(&self) -> (u32, u32) {
        match self {
            LaunchSize::Custom(func) => {
                let res = func();
                (res.x, res.y)
            }
            LaunchSize::FractionalFullScreen(x, y) => (
                (WINDOW_SIZE.x as u32).div_ceil(*x),
                (WINDOW_SIZE.y as u32).div_ceil(*y),
            ),
            LaunchSize::FullScreen => (WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            LaunchSize::X(x) => (*x, 1),
            LaunchSize::XY(x, y) => (*x, *y),
        }
    }
}

pub(crate) struct RayTracingPass {
    launch: LaunchSize,
    pipeline: RayTracingPipelineHandle,
}

impl RayTracingPass {
    pub(crate) fn new<'a>(rg: &'a mut RenderGraph) -> NodeBuilder<'a, RayTracingPass> {
        let mut builder = NodeBuilder::<RayTracingPass>::new::<RayTracingPass>(rg);
        builder.execution = MaybeUninit::new(RayTracingPass {
            launch: LaunchSize::FullScreen,
            pipeline: RayTracingPipelineHandle {
                path: "".to_string(),
                entry: "main".to_string(),
            },
        });
        builder
    }
}

impl<'b> NodeBuilder<'b, RayTracingPass> {
    pub(crate) fn shader(mut self, path: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.path = path.to_string();
        self
    }
    pub(crate) fn entry(mut self, entry: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.entry = entry.to_string();
        self
    }
    pub(crate) fn launch(mut self, launch: LaunchSize) -> NodeHandle {
        unsafe { self.execution.assume_init_mut() }.launch = launch;
        self.rg.graph.push(Node {
            reads: self.reads,
            writes: self.writes,
            descriptor_buffer: BufferHandle::default(),
            execution: Box::new(unsafe { self.execution.assume_init() }),
        });
        self.handle
    }
}

impl Execution for RayTracingPass {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()> {
        let (x, y) = self.launch.size();
        self.pipeline.launch(cmd, x, y);
        Ok(())
    }
    fn get_stages(&self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
    }
}
