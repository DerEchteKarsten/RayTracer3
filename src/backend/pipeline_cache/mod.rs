use std::{collections::HashMap, ffi::CStr, mem::MaybeUninit, sync::Once};

use anyhow::Result;
use ash::vk;

use crate::WINDOW_SIZE;

use super::{
    bindless::BindlessDescriptorHeap,
    vulkan::{
        image::ImageHandle,
        raytracing::{RayTracingContext, RayTracingShaderCreateInfo, ShaderBindingTable},
        Context,
    },
};

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ComputePipelineHandle {
    pub path: String,
    pub entry: String,
}

impl ComputePipelineHandle {
    pub fn dispatch(&self, cmd: &vk::CommandBuffer, x: u32, y: u32, z: u32) {
        let ctx = Context::get();
        let pipeline = PipelineCache::get_compute_pipeline(self);
        unsafe {
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            ctx.device.cmd_dispatch(*cmd, x, y, z);
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct RayTracingPipelineHandle {
    pub path: String,
    pub entry: String,
}

impl RayTracingPipelineHandle {
    pub fn launch(&self, cmd: &vk::CommandBuffer, x: u32, y: u32) {
        let (pipeline, sbt) = PipelineCache::get_raytracing_pipeline(self);
        unsafe {
            let raytracing = RayTracingContext::get().unwrap();
            let ctx = Context::get();
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, *pipeline);
            let call_region = vk::StridedDeviceAddressRegionKHR::default();
            raytracing.pipeline_fn.cmd_trace_rays(
                *cmd,
                &sbt.raygen_region,
                &sbt.miss_region,
                &sbt.hit_region,
                &call_region,
                x,
                y,
                1,
            );
        };
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct RasterPipelineHandle {
    pub mesh_path: String,
    pub mesh_entry: String,
    pub fragment_path: String,
    pub fragment_entry: String,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct RasterPipelineHash {
    handle: RasterPipelineHandle,
    color_formats: Vec<vk::Format>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
}

impl RasterPipelineHandle {
    pub fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        color_attachments: &[ImageHandle],
        depth_attachment: &Option<ImageHandle>,
        stencil_attachment: &Option<ImageHandle>,
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        z: u32,
    ) {
        let color_formats = color_attachments.iter().map(|e| e.format).collect::<Vec<_>>();
        let depth_format = depth_attachment.and_then(|d| Some(d.format)).unwrap_or(vk::Format::UNDEFINED);
        let stencil_format = stencil_attachment.and_then(|d| Some(d.format)).unwrap_or(vk::Format::UNDEFINED);

        let ctx = Context::get();
        let pipeline = PipelineCache::get_raster_pipeline(self, color_formats, depth_format, stencil_format);

        let color_attachments = color_attachments
            .iter()
            .map(|e| {
                vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(e.view)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 1.0, 0.0],
                        },
                    })
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
            })
            .collect::<Vec<_>>();
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(color_attachments.as_slice())
            .layer_count(1)
            .render_area(
                vk::Rect2D::default()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
            )
            .view_mask(0);

        let render_info1;
        let render_info2;

        if let Some(depth_attachment) = &depth_attachment {
            render_info1 = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image_view(depth_attachment.view)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE);
            rendering_info = rendering_info.depth_attachment(&render_info1);
        }
        if let Some(stencil_attachment) = &stencil_attachment {
            render_info2 = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(stencil_attachment.view)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE);
            rendering_info = rendering_info.stencil_attachment(&render_info2);
        }

        unsafe {
            ctx.device.cmd_begin_rendering(cmd, &rendering_info);
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

            ctx.device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: WINDOW_SIZE.x as f32,
                    height: WINDOW_SIZE.y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            ctx.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    extent: vk::Extent2D {
                        width: WINDOW_SIZE.x as u32,
                        height: WINDOW_SIZE.y as u32,
                    },
                    offset: vk::Offset2D { x: 0, y: 0 },
                }],
            );
            // ctx.mesh_fn.cmd_draw_mesh_tasks(cmd, x, y, z);
            ctx.device.cmd_draw(cmd, 3, 1, 0, 0);
            ctx.device.cmd_end_rendering(cmd);
        };
    }
}

pub(crate) struct PipelineCache {
    compute_pipelines: HashMap<ComputePipelineHandle, vk::Pipeline>,
    raster_pipelines: HashMap<RasterPipelineHash, vk::Pipeline>,
    raytracing_pipelines: HashMap<RayTracingPipelineHandle, (vk::Pipeline, ShaderBindingTable)>,
    shader_cache: HashMap<String, vk::ShaderModule>,
}

static mut CACHE: MaybeUninit<PipelineCache> = MaybeUninit::uninit();

impl PipelineCache {
    pub(crate) fn create_shader_module(&mut self, code_path: &str) -> Result<vk::ShaderModule> {
        match self.shader_cache.get(code_path) {
            Some(module) => Ok(*module),
            None => {
                let mut code = std::fs::File::open(code_path)?;
                let decoded_code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);

                let module = unsafe {
                    Context::get()
                        .device
                        .create_shader_module(&create_info, None)?
                };
                self.shader_cache.insert(code_path.to_string(), module);
                Ok(module)
            }
        }
    }

    fn create_shader_stage<'a>(
        &mut self,
        code_path: &str,
        main: &'a str,
        stage: vk::ShaderStageFlags,
    ) -> Result<vk::PipelineShaderStageCreateInfo<'a>> {
        let module = self.create_shader_module(code_path)?;
        Ok(vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(module)
            .name(CStr::from_bytes_with_nul(main.as_bytes())?))
    }

    pub fn init() {
        unsafe {
            CACHE.write(PipelineCache::new());
        }
    }

    pub fn get() -> &'static PipelineCache {
        unsafe { CACHE.assume_init_ref() }
    }

    pub fn get_mut() -> &'static mut PipelineCache {
        unsafe { CACHE.assume_init_mut() }
    }

    pub fn new() -> Self {
        Self {
            compute_pipelines: HashMap::new(),
            shader_cache: HashMap::new(),
            raytracing_pipelines: HashMap::new(),
            raster_pipelines: HashMap::new(),
        }
    }

    pub fn get_compute_pipeline(handle: &ComputePipelineHandle) -> vk::Pipeline {
        let s = Self::get_mut();
        match s.compute_pipelines.get(handle) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let entry = format!("{}\0", handle.entry);
                let path = format!("./shaders/bin/{}.slang.spv", handle.path);

                let ctx = Context::get();
                let create_info = vk::ComputePipelineCreateInfo::default()
                    .layout(BindlessDescriptorHeap::get().layout)
                    .stage(
                        Self::get_mut()
                            .create_shader_stage(&path, &entry, vk::ShaderStageFlags::COMPUTE)
                            .unwrap(),
                    );
                let pipeline = unsafe {
                    ctx.device
                        .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                        .unwrap()
                }[0];
                ctx.set_debug_name(&handle.path, pipeline);
                Self::get_mut()
                    .compute_pipelines
                    .insert(handle.clone(), pipeline.clone());
                pipeline
            }
        }
    }

    pub fn get_raytracing_pipeline(
        handle: &RayTracingPipelineHandle,
    ) -> &(vk::Pipeline, ShaderBindingTable) {
        let s = Self::get_mut();
        match s.raytracing_pipelines.get(handle) {
            Some(pipeline) => pipeline,
            None => {
                let entry = format!("{}\0", handle.entry);
                let path = format!("./shaders/bin/{}.slang.spv", handle.path,);

                let raytracing_ctx = RayTracingContext::get().unwrap();
                let (pipeline, shader_binding_table) = raytracing_ctx
                    .create_raytracing_pipeline(
                        BindlessDescriptorHeap::get().layout,
                        &[
                            RayTracingShaderCreateInfo {
                                group: crate::raytracing::RayTracingShaderGroup::RayGen,
                                source: &[(&path, &entry, vk::ShaderStageFlags::RAYGEN_KHR)],
                            },
                            RayTracingShaderCreateInfo {
                                group: crate::raytracing::RayTracingShaderGroup::Hit,
                                source: &[(
                                    "shaders/bin/default_hit",
                                    "main\0",
                                    vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                                )],
                            },
                            RayTracingShaderCreateInfo {
                                group: crate::raytracing::RayTracingShaderGroup::Miss,
                                source: &[(
                                    "shaders/bin/default_miss",
                                    "main\0",
                                    vk::ShaderStageFlags::MISS_KHR,
                                )],
                            },
                        ],
                    )
                    .unwrap();
                Context::get().set_debug_name(&handle.path, pipeline);
                Self::get_mut()
                    .raytracing_pipelines
                    .insert(handle.clone(), (pipeline, shader_binding_table));
                Self::get().raytracing_pipelines.get(handle).unwrap()
            }
        }
    }

    pub fn get_raster_pipeline(handle: &RasterPipelineHandle, color_formats: Vec<vk::Format>, depth_format: vk::Format, stencil_format: vk::Format) -> vk::Pipeline {
        let s = Self::get_mut();
        let hash = RasterPipelineHash {handle: handle.clone(), color_formats, depth_format, stencil_format};
        match s.raster_pipelines.get(&hash) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let fragment_entry = format!("{}\0", handle.fragment_entry);
                let fragment_path = format!("./shaders/bin/{}.slang.spv", handle.fragment_path);
                let mesh_entry = format!("{}\0", handle.mesh_entry);
                let mesh_path = format!("./shaders/bin/{}.slang.spv", handle.mesh_path);

                let ctx = Context::get();

                let mut rendering = vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(&hash.color_formats)
                    .depth_attachment_format(depth_format)
                    .stencil_attachment_format(stencil_format)
                    .view_mask(0);

                let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                    .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

                let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                    .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                    .sample_shading_enable(false)
                    .min_sample_shading(1.0)
                    .alpha_to_coverage_enable(false)
                    .alpha_to_one_enable(false)
                    .sample_mask(&[]);

                let color_blend_attachments = hash
                    .color_formats
                    .iter()
                    .map(|e| {
                        vk::PipelineColorBlendAttachmentState::default()
                            .blend_enable(false)
                            .color_write_mask(vk::ColorComponentFlags::RGBA)
                            .alpha_blend_op(vk::BlendOp::ADD)
                            .color_blend_op(vk::BlendOp::ADD)
                            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .src_color_blend_factor(vk::BlendFactor::SRC_COLOR)
                    })
                    .collect::<Vec<_>>();
                let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_attribute_descriptions(&[])
                    .vertex_attribute_descriptions(&[]);
                let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                    .attachments(color_blend_attachments.as_slice())
                    .logic_op_enable(false)
                    .logic_op(vk::LogicOp::COPY)
                    .blend_constants([0.0, 0.0, 0.0, 0.0]);
                let stages = [
                    Self::get_mut()
                        .create_shader_stage(&mesh_path, &mesh_entry, vk::ShaderStageFlags::VERTEX)
                        .unwrap(),
                    Self::get_mut()
                        .create_shader_stage(
                            &fragment_path,
                            &fragment_entry,
                            vk::ShaderStageFlags::FRAGMENT,
                        )
                        .unwrap(),
                ];
                let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                    .scissor_count(1)
                    .viewport_count(1);
                let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                    .primitive_restart_enable(false)
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
                let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                    .depth_clamp_enable(false)
                    .rasterizer_discard_enable(false)
                    .polygon_mode(vk::PolygonMode::FILL)
                    .line_width(1.0)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .front_face(vk::FrontFace::CLOCKWISE)
                    .depth_bias_enable(false);
                let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_bounds_test_enable(false)
                    .depth_compare_op(vk::CompareOp::LESS)
                    .depth_test_enable(false)
                    .depth_write_enable(false)
                    .min_depth_bounds(0.0)
                    .max_depth_bounds(1.0)
                    .stencil_test_enable(false);
                let create_info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .layout(BindlessDescriptorHeap::get().layout)
                    .dynamic_state(&dynamic_state)
                    .multisample_state(&multisampling)
                    .color_blend_state(&color_blend_state)
                    .rasterization_state(&rasterization_state)
                    .input_assembly_state(&input_assembly_state)
                    .vertex_input_state(&vertex_input_info)
                    .viewport_state(&viewport_state)
                    .depth_stencil_state(&depth_stencil_state)
                    .base_pipeline_handle(vk::Pipeline::null())
                    .base_pipeline_index(-1)
                    .push_next(&mut rendering);

                let pipeline = unsafe {
                    ctx.device
                        .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                        .unwrap()
                }[0];

                Self::get_mut()
                    .raster_pipelines
                    .insert(hash, pipeline.clone());
                pipeline
            }
        }
    }
}
