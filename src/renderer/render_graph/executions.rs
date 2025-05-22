use std::mem::MaybeUninit;

use anyhow::Result;
use ash::vk;

use crate::raytracing::RayTracingContext;
use crate::renderer::pipeline_cache::{ComputePipelineHandle, RasterPipelineHandle};
use crate::vulkan::Context;
use crate::PipelineCache;
use crate::{renderer::pipeline_cache::RayTracingPipelineHandle, WINDOW_SIZE};

use super::build::{DispatchSize, NodeBuilder};
use super::{EdgeType, ExecutionTrait, NodeEdge, RenderGraph};

#[derive(Clone, PartialEq)]
pub(crate) struct ComputePass {
    pub(super) pipeline: ComputePipelineHandle,
    pub(super) dispatch: DispatchSize,
}

impl ComputePass {
    pub(crate) fn new<'a>(
        rg: &'a mut RenderGraph,
        name: &'static str,
    ) -> NodeBuilder<'a, ComputePass> {
        let mut builder = NodeBuilder::<ComputePass>::new::<ComputePass>(rg, name);
        builder.execution = MaybeUninit::new(ComputePass {
            pipeline: ComputePipelineHandle {
                path: "",
                entry: "main",
            },
            dispatch: DispatchSize::FullScreen,
        });
        builder
    }
}

impl ExecutionTrait for ComputePass {
    fn execute(
        &self,
        cmd: &vk::CommandBuffer,
        rg: &RenderGraph,
        ctx: &mut Context,
        raytracing_ctx: Option<&RayTracingContext>,
        cache: &mut PipelineCache,
        _edges: &[NodeEdge],
    ) -> Result<()> {
        let (x, y, z) = self.dispatch.size();
        self.pipeline.dispatch(ctx, cache, cmd, x, y, z);
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
}

impl WorkSize2D {
    fn size(&self) -> (u32, u32) {
        match self {
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
    pub(super) launch: WorkSize2D,
    pub(super) pipeline: RayTracingPipelineHandle,
}

impl RayTracingPass {
    pub(crate) fn new<'a>(
        rg: &'a mut RenderGraph,
        name: &'static str,
    ) -> NodeBuilder<'a, RayTracingPass> {
        let mut builder = NodeBuilder::<RayTracingPass>::new::<RayTracingPass>(rg, name);
        builder.execution = MaybeUninit::new(RayTracingPass {
            launch: WorkSize2D::FullScreen,
            pipeline: RayTracingPipelineHandle {
                path: "",
                entry: "main",
            },
        });
        builder
    }
}

impl ExecutionTrait for RayTracingPass {
    fn execute(
        &self,
        cmd: &vk::CommandBuffer,
        _: &RenderGraph,
        ctx: &mut Context,
        raytracing_ctx: Option<&RayTracingContext>,
        cache: &mut PipelineCache,
        _: &[NodeEdge],
    ) -> Result<()> {
        let (x, y) = self.launch.size();
        self.pipeline
            .launch(ctx, cache, raytracing_ctx.unwrap(), cmd, x, y);
        Ok(())
    }
    fn get_stages(&self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
    }
}

#[derive(PartialEq)]
pub(crate) struct RasterPass {
    pub(super) dispatch: DispatchSize,
    pub(super) render_area: WorkSize2D,
    pub(super) pipeline: RasterPipelineHandle,
}

impl ExecutionTrait for RasterPass {
    fn execute(
        &self,
        cmd: &vk::CommandBuffer,
        rg: &RenderGraph,
        ctx: &mut Context,
        _: Option<&RayTracingContext>,
        cache: &mut PipelineCache,
        edges: &[NodeEdge],
    ) -> Result<()> {
        let (x, y, z) = self.dispatch.size();
        let (width, height) = self.render_area.size();
        let color_attachments = edges
            .iter()
            .filter_map(|e| {
                if let EdgeType::ColorAttachmentOutput { clear_color } = e.edge_type {
                    Some((rg.image_handle(e.resource).unwrap(), clear_color))
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
            ctx,
            cache,
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
        vk::PipelineStageFlags2::FRAGMENT_SHADER
    }
}

impl RasterPass {
    pub(crate) fn new<'a>(
        rg: &'a mut RenderGraph,
        name: &'static str,
    ) -> NodeBuilder<'a, RasterPass> {
        let mut builder = NodeBuilder::<RasterPass>::new::<RasterPass>(rg, name);
        builder.execution = MaybeUninit::new(RasterPass {
            dispatch: DispatchSize::FullScreen,
            render_area: WorkSize2D::FullScreen,
            pipeline: RasterPipelineHandle {
                fragment_entry: "main",
                mesh_entry: "main",
                fragment_path: "",
                mesh_path: "",
            },
        });
        builder
    }
}
