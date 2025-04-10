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
    Execution, Node, NodeEdge, NodeHandle, RenderGraph, ResourceHandle, ResourceMemoryType,
    FRAMES_IN_FLIGHT, IMPORTED,
};

use std::ffi;

pub struct NodeBuilder<'a, T>
where
    T: Execution + 'static,
{
    rg: &'a mut RenderGraph,
    execution: MaybeUninit<T>,
    reads: Vec<NodeEdge>,
    writes: Vec<NodeEdge>,
    handle: NodeHandle,
    constants: Option<*const ffi::c_void>,
    constants_size: usize,
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
            constants: None,
            constants_size: 0,
        }
    }

    pub fn constants<C>(mut self, constants: &'b C) -> Self {
        self.constants = Some(constants as *const C as *const ffi::c_void);
        self.constants_size = size_of::<C>();
        self
    }

    pub fn read_array(mut self, output: NodeHandle, handle: ResourceHandle) -> Self {
        self.reads.push(NodeEdge {
            layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            output_of: output,
            output_index: self.rg.graph[output]
                .writes
                .iter()
                .position(|e| e.resource == handle)
                .unwrap(),
            resource: handle,
        });
        self
    }

    pub fn read(mut self, output: NodeHandle, handle: ResourceHandle) -> Self {
        self.reads.push(NodeEdge {
            layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            output_of: output,
            output_index: self.rg.graph[output]
                .writes
                .iter()
                .position(|e| e.resource == handle)
                .unwrap(),
            resource: handle,
        });
        self
    }

    pub fn write(mut self, handle: ResourceHandle) -> Self {
        let output_index = self.writes.len();
        self.writes.push(NodeEdge {
            layout: Some(vk::ImageLayout::GENERAL),
            output_of: self.handle,
            output_index,
            resource: handle,
        });
        self
    }

    pub fn read_write(mut self, output: NodeHandle, handle: ResourceHandle) -> Self {
        let output_index = self.writes.len();
        self.reads.push(NodeEdge {
            layout: Some(vk::ImageLayout::GENERAL),
            output_of: output,
            output_index: self.rg.graph[output]
                .writes
                .iter()
                .position(|e| e.resource == handle)
                .unwrap(),
            resource: handle,
        });
        self.writes.push(NodeEdge {
            layout: Some(vk::ImageLayout::GENERAL),
            output_of: self.handle,
            output_index,
            resource: handle,
        });
        self
    }
    fn build(self) -> NodeHandle {
        let constants_buffer = if let Some(constants) = self.constants {
            let handle = self.rg.internal_buffer(self.constants_size as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu);
            let buffer = &self.rg.internal_buffers[handle.index()];
            unsafe { buffer.allocation.mapped_ptr().unwrap().as_ptr().copy_from(constants, self.constants_size) };

            Some(handle)
        }else {
            None
        };
        
        let mut node = Node {
            reads: self.reads,
            writes: self.writes,
            descriptor_buffer: ResourceHandle(0),
            constants_buffer,
            execution: Box::new(unsafe { self.execution.assume_init() }),
            constants: self.constants,
            constants_size: self.constants_size
        };

        node.descriptor_buffer = self.rg.internal_buffer((node.num_bindings() * size_of::<usize>()) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu);
        self.rg.graph.push(node);
        self.handle
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
        self.build()
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
        self.build()
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
