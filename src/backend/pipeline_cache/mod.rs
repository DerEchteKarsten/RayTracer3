use std::{collections::HashMap, ffi::CStr, mem::MaybeUninit, sync::Once};

use anyhow::Result;
use ash::vk;

use super::{
    bindless::BindlessDescriptorHeap,
    vulkan::raytracing::{RayTracingContext, RayTracingShaderCreateInfo, ShaderBindingTable},
    vulkan::Context,
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
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
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
            let raytracing = RayTracingContext::get();
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

pub(crate) struct PipelineCache {
    compute_pipelines: HashMap<ComputePipelineHandle, vk::Pipeline>,
    // raster_pipelines: HashMap<ComputeShaderPipelineHandle<'static>, vk::Pipeline>,
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
        }
    }

    pub fn get_compute_pipeline<'a>(handle: &ComputePipelineHandle) -> &'a vk::Pipeline {
        let s = Self::get_mut();
        match s.compute_pipelines.get(handle) {
            Some(pipeline) => pipeline,
            None => {
                let entry = format!("{}\0", handle.entry);
                let path = format!("./shaders/bin/{}.slang.spv", handle.path,);

                let ctx = Context::get();
                println!("{}", path);
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
                    .insert(handle.clone(), pipeline);
                Self::get().compute_pipelines.get(handle).unwrap()
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
                let path = format!("./../../../shader/bin/{}.slang.spv", handle.path,);

                let raytracing_ctx = RayTracingContext::get();
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
}
