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
    ComputePipelineHandle, PipelineCache, RasterPipelineHandle, RayTracingPipelineHandle,
};
use crate::raytracing::{RayTracingContext, RayTracingShaderCreateInfo, ShaderBindingTable};
use crate::vulkan::Context;

use crate::{GConst, WINDOW_SIZE};

use derivative::Derivative;

use super::{
    EdgeType, Execution, ExecutionTrait, Node, NodeEdge, NodeHandle, RenderGraph,
    ResourceHandle, IMPORTED,
};

use std::ffi::{self, c_void};

pub struct NodeBuilder<'a, T>
where
    T: ExecutionTrait + 'static,
{
    name: &'static str,
    rg: &'a mut RenderGraph,
    execution: MaybeUninit<T>,
    edges: Vec<NodeEdge>,
    constants: Option<*const ffi::c_void>,
    constants_size: usize,
}

impl<'b, T> NodeBuilder<'b, T>
where
    T: ExecutionTrait + 'static,
    Execution: From<T>,
{
    fn new<E: ExecutionTrait + 'static>(rg: &'b mut RenderGraph, name: &'static str) -> NodeBuilder<'b, E> {
        NodeBuilder::<'b, E> {
            name,
            rg,
            execution: MaybeUninit::uninit(),
            edges: Vec::new(),
            constants: None,
            constants_size: 0,
        }
    }

    pub fn constants<C>(mut self, constants: &'b C) -> Self {
        self.constants = Some(constants as *const C as *const ffi::c_void);
        self.constants_size = size_of::<C>();
        self
    }

    pub fn read(mut self, origin: NodeHandle, handle: ResourceHandle) -> Self {
        if origin != IMPORTED {
            let prev = self.rg.nodes[origin]
                .edges
                .iter()
                .find(|e| e.resource == handle)
                .ok_or(Error::msg("Origin doesnt write to handle".to_owned()))
                .unwrap();
            if prev.edge_type == EdgeType::ShaderRead {
                panic!("Origin contains handle but does not write to it")
            }
        }
        let image_handle = self.rg.image_handle(handle);
        self.edges.push(NodeEdge {
            layout: image_handle.and_then(|image| {
                Some(if image.usage.contains(vk::ImageUsageFlags::STORAGE) {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                })
            }),
            origin: if origin == IMPORTED {
                None
            } else {
                Some(origin)
            },
            edge_type: EdgeType::ShaderRead,
            resource: handle,
        });
        self
    }

    pub fn write(mut self, last_read: NodeHandle, handle: ResourceHandle) -> Self {
        self.edges.push(NodeEdge {
            layout: Some(vk::ImageLayout::GENERAL),
            origin: if last_read == IMPORTED {
                None
            } else {
                Some(last_read)
            },
            edge_type: EdgeType::ShaderWrite,
            resource: handle,
        });
        self
    }

    pub fn read_write(mut self, origin: NodeHandle, handle: ResourceHandle) -> Self {
        self.edges.push(NodeEdge {
            layout: Some(vk::ImageLayout::GENERAL),
            origin: if origin == IMPORTED {
                None
            } else {
                Some(origin)
            },
            edge_type: super::EdgeType::ShaderReadWrite,
            resource: handle,
        });
        self
    }
    fn build(self) -> NodeHandle {
        let offset = if let Some(constants) = self.constants {
            if let Some(found_constants) = self
                .rg
                .constants
                .iter()
                .position(|co| co.0 == constants)
            {
                let offset: usize = self
                    .rg
                    .constants[..found_constants]
                    .iter()
                    .map(|c| c.1)
                    .sum();
                Some(offset as u32)
            } else {
                let offset = self
                    .rg
                    .constants
                    .iter()
                    .map(|c| c.1)
                    .sum();
                self.rg.constants.push((constants.clone(), self.constants_size));

                let buffer_ptr = self.rg.constants_buffer
                    .ty
                    .buffer()
                    .allocation
                    .as_ref()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut c_void;
                unsafe {
                    constants
                        .copy_to(buffer_ptr.add(offset), self.constants_size)
                };
                Some(offset as u32)
            }
        }else {
            None
        };  

        match self.rg.nodes.get_mut(self.name) {
            None => {self.rg.nodes.insert(self.name, Node {
                constant_offset: offset,
                edges: self.edges,
                execution: Execution::from(unsafe { self.execution.assume_init() }),
            });},
            Some(node) => {node.edges = self.edges;}
        }

        self.rg.graph.insert(self.name);
        self.name
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

#[derive(Default, Clone, Copy, PartialEq)]
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

#[derive(Clone, PartialEq)]
pub(crate) struct ComputePass {
    pipeline: ComputePipelineHandle,
    dispatch: DispatchSize,
}

impl ComputePass {
    pub(crate) fn new<'a>(rg: &'a mut RenderGraph, name: &'static str) -> NodeBuilder<'a, ComputePass> {
        let mut builder = NodeBuilder::<ComputePass>::new::<ComputePass>(rg, name);
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

impl ExecutionTrait for ComputePass {
    fn execute(
        &self,
        cmd: &vk::CommandBuffer,
        _rg: &RenderGraph,
        _edges: &[NodeEdge],
    ) -> Result<()> {
        let (x, y, z) = self.dispatch.size();
        self.pipeline.dispatch(cmd, x, y, z);
        Ok(())
    }
    fn get_stages(&self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::COMPUTE_SHADER
    }
}

#[derive(Default, PartialEq)]
pub enum WorkSize2D {
    #[default]
    FullScreen,
    FractionalFullScreen(u32, u32),
    X(u32),
    XY(u32, u32),
    Custom(fn() -> UVec2),
}

impl WorkSize2D {
    fn size(&self) -> (u32, u32) {
        match self {
            WorkSize2D::Custom(func) => {
                let res = func();
                (res.x, res.y)
            }
            WorkSize2D::FractionalFullScreen(x, y) => (
                (WINDOW_SIZE.x as u32).div_ceil(*x),
                (WINDOW_SIZE.y as u32).div_ceil(*y),
            ),
            WorkSize2D::FullScreen => (WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            WorkSize2D::X(x) => (*x, 1),
            WorkSize2D::XY(x, y) => (*x, *y),
        }
    }
}

#[derive(PartialEq)]
pub(crate) struct RayTracingPass {
    launch: WorkSize2D,
    pipeline: RayTracingPipelineHandle,
}

impl RayTracingPass {
    pub(crate) fn new<'a>(rg: &'a mut RenderGraph, name: &'static str) -> NodeBuilder<'a, RayTracingPass> {
        let mut builder = NodeBuilder::<RayTracingPass>::new::<RayTracingPass>(rg, name);
        builder.execution = MaybeUninit::new(RayTracingPass {
            launch: WorkSize2D::FullScreen,
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
    pub(crate) fn launch(mut self, launch: WorkSize2D) -> NodeHandle {
        unsafe { self.execution.assume_init_mut() }.launch = launch;
        self.build()
    }
}

impl ExecutionTrait for RayTracingPass {
    fn execute(&self, cmd: &vk::CommandBuffer, _: &RenderGraph, _: &[NodeEdge]) -> Result<()> {
        let (x, y) = self.launch.size();
        self.pipeline.launch(cmd, x, y);
        Ok(())
    }
    fn get_stages(&self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
    }
}

#[derive(PartialEq)]
pub(crate) struct RasterPass {
    dispatch: DispatchSize,
    render_area: WorkSize2D,
    pipeline: RasterPipelineHandle,
}

impl ExecutionTrait for RasterPass {
    fn execute(&self, cmd: &vk::CommandBuffer, rg: &RenderGraph, edges: &[NodeEdge]) -> Result<()> {
        let (x, y, z) = self.dispatch.size();
        let (width, height) = self.render_area.size();
        let color_attachments = edges
            .iter()
            .filter_map(|e| {
                if e.edge_type == EdgeType::ColorAttachment {
                    Some(rg.image_handle(e.resource).unwrap())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let depth_attachment = edges
            .iter()
            .find(|e| e.edge_type == EdgeType::DepthAttachment)
            .and_then(|e| rg.image_handle(e.resource));
        let stencil_attachment = edges
            .iter()
            .find(|e| e.edge_type == EdgeType::StencilAttachment)
            .and_then(|e| rg.image_handle(e.resource));
        self.pipeline.dispatch(
            *cmd,
            &color_attachments,
            &depth_attachment,
            &stencil_attachment,
            width,
            height,
            x,
            y,
            z,
        );
        Ok(())
    }
    fn get_stages(&self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::ALL_GRAPHICS
    }
}

impl RasterPass {
    pub(crate) fn new<'a>(rg: &'a mut RenderGraph, name: &'static str) -> NodeBuilder<'a, RasterPass> {
        let mut builder = NodeBuilder::<RasterPass>::new::<RasterPass>(rg, name);
        builder.execution = MaybeUninit::new(RasterPass {
            dispatch: DispatchSize::FullScreen,
            render_area: WorkSize2D::FullScreen,
            pipeline: RasterPipelineHandle {
                fragment_entry: "main".to_string(),
                mesh_entry: "main".to_string(),
                fragment_path: "".to_string(),
                mesh_path: "".to_string(),
            },
        });
        builder
    }
}

impl<'b> NodeBuilder<'b, RasterPass> {
    pub(crate) fn mesh_shader(mut self, path: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .mesh_path = path.to_string();
        self
    }
    pub(crate) fn mesh_entry(mut self, entry: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .mesh_entry = entry.to_string();
        self
    }
    pub(crate) fn fragment_shader(mut self, path: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .fragment_path = path.to_string();
        self
    }
    pub(crate) fn fragment_entry(mut self, entry: &str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .fragment_entry = entry.to_string();
        self
    }

    pub(crate) fn render_area(mut self, render_area: WorkSize2D) -> Self {
        unsafe { self.execution.assume_init_mut() }.render_area = render_area;
        self
    }
    pub(crate) fn draw(mut self, dispatch_size: DispatchSize) -> NodeHandle {
        {
            let pipeline = unsafe { self.execution.assume_init_mut() };
            pipeline.dispatch = dispatch_size;
        }
        self.build()
    }

    pub(crate) fn color_attachment(
        mut self,
        last_read: NodeHandle,
        handle: ResourceHandle,
    ) -> Self {
        self.edges.push(NodeEdge {
            layout: Some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            origin: if last_read == IMPORTED {
                None
            } else {
                Some(last_read)
            },
            edge_type: EdgeType::ColorAttachment,
            resource: handle,
        });
        self
    }
    pub(crate) fn depth_attachment(mut self, origin: NodeHandle, handle: ResourceHandle) -> Self {
        self.edges.push(NodeEdge {
            layout: Some(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
            origin: if origin == IMPORTED {
                None
            } else {
                Some(origin)
            },
            edge_type: EdgeType::DepthAttachment,
            resource: handle,
        });
        self
    }
    pub(crate) fn stencil_attachment(mut self, origin: NodeHandle, handle: ResourceHandle) -> Self {
        self.edges.push(NodeEdge {
            layout: Some(vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL),
            origin: if origin == IMPORTED {
                None
            } else {
                Some(origin)
            },
            edge_type: EdgeType::StencilAttachment,
            resource: handle,
        });
        self
    }
}
