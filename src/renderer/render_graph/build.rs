use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use anyhow::{Error, Result};
use ash::vk::{self, BufferUsageFlags, ImageUsageFlags};
use bevy_ecs::world::World;
use glam::{UVec2, UVec3};
use gpu_allocator::MemoryLocation;
use winit::platform::x11;

use crate::raytracing::{RayTracingContext, RayTracingShaderCreateInfo, ShaderBindingTable};
use crate::renderer::bindless::{BindlessDescriptorHeap, DescriptorResourceHandle};
use crate::renderer::pipeline_cache::{
    ComputePipelineHandle, PipelineCache, RasterPipelineHandle, RayTracingPipelineHandle,
};
use crate::vulkan::Context;

use crate::WINDOW_SIZE;

use derivative::Derivative;

use super::executions::{ComputePass, RasterPass, RayTracingPass, WorkSize2D};
use super::{
    EdgeType, Execution, ExecutionTrait, Node, NodeEdge, NodeHandle, RenderGraph,
    ResourceDescriptionType, ResourceHandle, ResourceType, IMPORTED,
};

use std::ffi::{self, c_void};

pub(crate) struct NodeBuilder<'a, T>
where
    T: ExecutionTrait + 'static,
{
    pub(super) name: &'static str,
    pub(super) rg: &'a mut RenderGraph,
    pub(super) execution: MaybeUninit<T>,
    pub(super) edges: Vec<NodeEdge>,
    pub(super) constants_offset: Option<u32>,
    pub(super) constants_size: usize,
}

impl<'b, T> NodeBuilder<'b, T>
where
    T: ExecutionTrait + 'static,
    Execution: From<T>,
{
    pub(super) fn new<E: ExecutionTrait + 'static>(
        rg: &'b mut RenderGraph,
        name: &'static str,
    ) -> NodeBuilder<'b, E> {
        if rg.nodes.iter().find(|e| e.name == name).is_some() {
            panic!("Node name {name} allready used")
        }
        NodeBuilder::<'b, E> {
            name,
            rg,
            execution: MaybeUninit::uninit(),
            edges: Vec::new(),
            constants_offset: None,
            constants_size: 0,
        }
    }

    pub fn constants<C>(mut self, constants: &'b C) -> Self {
        self.constants_size = size_of::<C>();
        let constants_ptr = constants as *const C as *const ffi::c_void;

        if let Some(offset) = self.rg.constants.get(&(constants_ptr as usize)) {
            self.constants_offset = Some(*offset);
        } else {
            let offset = self.rg.constants_offset;
            self.rg.constants_offset += self.constants_size as u32;

            let buffer_ptr = self
                .rg
                .constants_buffer
                .ty
                .buffer()
                .allocation
                .as_ref()
                .unwrap()
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            unsafe {
                constants_ptr.copy_to(buffer_ptr, self.constants_size);
            };
            self.constants_offset = Some(offset);
            self.rg.constants.insert(constants_ptr as usize, offset);
        }
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
        if let ResourceType::Uninitilized(index) = self.rg.resources[handle].ty {
            match &mut self.rg.resource_cache[index].ty {
                super::ResourceDescriptionType::Buffer { size, usage } => {
                    *usage |= BufferUsageFlags::STORAGE_BUFFER
                }
                ResourceDescriptionType::Image {
                    size,
                    usage,
                    format,
                } => {
                    if !usage.contains(ImageUsageFlags::STORAGE) {
                        *usage |= ImageUsageFlags::SAMPLED;
                    } else {
                        *usage |= ImageUsageFlags::STORAGE;
                        *usage &= ImageUsageFlags::SAMPLED;
                    }
                }
            }
        }
        self.edges.push(NodeEdge {
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
        if let ResourceType::Uninitilized(index) = self.rg.resources[handle].ty {
            match &mut self.rg.resource_cache[index].ty {
                super::ResourceDescriptionType::Buffer { size, usage } => {
                    *usage |= BufferUsageFlags::STORAGE_BUFFER
                }
                ResourceDescriptionType::Image {
                    size,
                    usage,
                    format,
                } => {
                    *usage |= ImageUsageFlags::STORAGE;
                    *usage &= !ImageUsageFlags::SAMPLED;
                }
            }
        }
        self.edges.push(NodeEdge {
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
        if let ResourceType::Uninitilized(index) = self.rg.resources[handle].ty {
            match &mut self.rg.resource_cache[index].ty {
                super::ResourceDescriptionType::Buffer { size, usage } => {
                    *usage |= BufferUsageFlags::STORAGE_BUFFER
                }
                ResourceDescriptionType::Image {
                    size,
                    usage,
                    format,
                } => {
                    *usage |= ImageUsageFlags::STORAGE;
                    *usage &= !ImageUsageFlags::SAMPLED;
                }
            }
        }
        self.edges.push(NodeEdge {
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
        let mut seen = HashSet::new();
        if let Some(edge) = self.edges.iter().find(|x| !seen.insert(x.resource.clone())) {
            panic!("resource: {:?} is duplicate", edge.resource);
        }

        let handle = self.rg.nodes.len();
        self.rg.nodes.push(Node {
            name: self.name,
            execution: unsafe { self.execution.assume_init().into() },
            constant_offset: self.constants_offset,
            edges: self.edges,
        });
        handle
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
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
    pub(super) fn size(&self) -> (u32, u32, u32) {
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

impl<'b> NodeBuilder<'b, RasterPass> {
    pub(crate) fn mesh_shader(mut self, path: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .mesh_path = path;
        self
    }
    pub(crate) fn mesh_entry(mut self, entry: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .mesh_entry = entry;
        self
    }
    pub(crate) fn fragment_shader(mut self, path: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .fragment_path = path;
        self
    }
    pub(crate) fn fragment_entry(mut self, entry: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }
            .pipeline
            .fragment_entry = entry;
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
        clear_color: Option<[f32; 4]>,
    ) -> Self {
        if let ResourceType::Uninitilized(index) = self.rg.resources[handle].ty {
            match &mut self.rg.resource_cache[index].ty {
                super::ResourceDescriptionType::Buffer { size, usage } => {
                    panic!("Exspected Image")
                }
                ResourceDescriptionType::Image {
                    size,
                    usage,
                    format,
                } => *usage |= ImageUsageFlags::COLOR_ATTACHMENT,
            }
        }
        self.edges.push(NodeEdge {
            origin: if last_read == IMPORTED {
                None
            } else {
                Some(last_read)
            },
            edge_type: EdgeType::ColorAttachmentOutput { clear_color },
            resource: handle,
        });
        self
    }
    pub(crate) fn depth_attachment(mut self, origin: NodeHandle, handle: ResourceHandle) -> Self {
        if let ResourceType::Uninitilized(index) = self.rg.resources[handle].ty {
            match &mut self.rg.resource_cache[index].ty {
                super::ResourceDescriptionType::Buffer { size, usage } => {
                    panic!("Exspected Image")
                }
                ResourceDescriptionType::Image {
                    size,
                    usage,
                    format,
                } => *usage |= ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            }
        }
        self.edges.push(NodeEdge {
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

impl<'b> NodeBuilder<'b, RayTracingPass> {
    pub(crate) fn shader(mut self, path: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.path = path;
        self
    }
    pub(crate) fn entry(mut self, entry: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.entry = entry;
        self
    }
    pub(crate) fn launch(mut self, launch: WorkSize2D) -> NodeHandle {
        unsafe { self.execution.assume_init_mut() }.launch = launch;
        self.build()
    }
}

impl<'b> NodeBuilder<'b, ComputePass> {
    pub(crate) fn shader(mut self, path: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.path = path;
        self
    }
    pub(crate) fn entry(mut self, entry: &'static str) -> Self {
        unsafe { self.execution.assume_init_mut() }.pipeline.entry = entry;
        self
    }
    pub(crate) fn dispatch(mut self, dispatch: DispatchSize) -> NodeHandle {
        unsafe { self.execution.assume_init_mut() }.dispatch = dispatch;
        self.build()
    }
}
