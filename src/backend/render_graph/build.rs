use std::collections::HashMap;

use anyhow::Result;
use ash::vk;
use glam::{UVec2, UVec3};
use gpu_allocator::MemoryLocation;
use winit::platform::x11;

use crate::backend::bindless::{BindlessDescriptorHeap, DescriptorResourceHandle};
use crate::backend::pipeline_cache::{ComputePipelineHandle, PipelineCache, RayTracingPipelineHandle};
use crate::raytracing::{RayTracingContext, RayTracingShaderCreateInfo, ShaderBindingTable};
use crate::vulkan_context::Context;

use crate::backend::utils::{Buffer, Image, ImageResource};
use crate::WINDOW_SIZE;

use derivative::Derivative;

use super::{
    Execution, Node, NodeEdge, NodeHandle, RenderGraph, ResourceDescription, ResourceOrigin,
    FRAMES_IN_FLIGHT,
};

struct NodeBuilder<'a> {
    rg: &'a mut RenderGraph,
    handle: NodeHandle,
}

impl<'b> NodeBuilder<'b> {
    pub fn new(rg: &'b mut RenderGraph, name: &str, execution: impl Execution + 'static) -> Self {
        let handle = rg.graph.len();
        rg.graph.push(Node {
            name: name.to_string(),
            reads: Vec::new(),
            writes: Vec::new(),
            execution: Box::new(execution),
        });

        Self { rg, handle }
    }

    pub fn read(self, node: NodeHandle, index: usize) -> Self {
        let handle = self.rg.graph[node].writes[index].clone();
        self.read_from(handle)
    }
    pub fn read_from(self, handle: NodeEdge) -> Self {
        self.rg.graph[self.handle].reads.push(handle);
        self
    }

    pub fn write(self, description: ResourceDescription) -> Self {
        let s = &mut self.rg.graph[self.handle];
        let handle = NodeEdge {
            origin: ResourceOrigin::RenderGraphTransient {
                output_of: self.handle,
                output_index: s.writes.len(),
            },
            description,
        };
        self.write_to(handle)
    }
    pub fn write_to(self, handle: NodeEdge) -> Self {
        self.rg.graph[self.handle].writes.push(handle);
        self
    }
}

#[derive(Default)]
enum DispatchSize {
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

struct ComputePass {
    pub pipeline: ComputePipelineHandle,
    pub dispatch: DispatchSize,
}

impl ComputePass {
    fn new_with_entry(path: &str, entry: &str, dispatch: DispatchSize) -> Self {
        Self { pipeline: ComputePipelineHandle { path: path.to_string(), entry: entry.to_string() }, dispatch }
    }
    fn new(path: &str, dispatch: DispatchSize) -> Self {
        Self::new_with_entry(path, "main", dispatch)
    }
}

impl Execution for ComputePass {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()> {
        let (x, y, z) = self.dispatch.size();
        self.pipeline.dispatch(cmd, x, y, z);        
        Ok(())
    }
}

#[derive(Default)]
enum LaunchSize {
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

struct RayTracingPass {
    launch: LaunchSize,
    pipeline: RayTracingPipelineHandle,
}

impl RayTracingPass {
    pub fn new_entry(path: &str, entry: &str, launch: LaunchSize) -> Self {
        RayTracingPass { launch, pipeline: RayTracingPipelineHandle { path: path.to_string(), entry: entry.to_string(), } }
    }
    pub fn new(path: &str, launch: LaunchSize) -> Self {
        RayTracingPass { launch, pipeline: RayTracingPipelineHandle { path: path.to_string(), entry: "main".to_string(), } }
    }
}

impl Execution for RayTracingPass {
    fn execute(&self, cmd: &vk::CommandBuffer) -> Result<()> {
        let (x, y) = self.launch.size();
        self.pipeline.launch(cmd, x, y);
        Ok(())
    }
}
