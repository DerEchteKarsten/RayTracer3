use std::{collections::HashSet, ffi::CStr, slice::from_ref};

use crate::{
    backend::raytracing::{RayTracingContext, RayTracingShaderCreateInfo, RayTracingShaderGroup},
    GConst,
};

use super::*;
use build::*;
use run::*;

impl RenderGraph {
    pub fn swpachain_format(&self) -> vk::Format {
        self.static_resources.swapchain.format
    }

    pub fn num_swpachain_images(&self) -> u32 {
        self.static_resources.swapchain.images.len() as u32
    }

    pub fn new(
        scene_resources: SceneResources,
        imgui: &mut ImGui,
    ) -> RenderGraph {
        let ctx = Context::get();
        let static_resources = StaticResources::new(scene_resources, imgui).unwrap();
        let static_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_count(static_resources.scene.texture_samplers.len() as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let descriptor_setlayout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&static_bindings);
        let static_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&descriptor_setlayout_info, None)
                .unwrap()
        };

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(static_resources.scene.texture_samplers.len() as u32 + 2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);

        let pool = unsafe { ctx.device.create_descriptor_pool(&pool_info, None).unwrap() };

        let binding = [static_descriptor_set_layout];
        let allocation_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&binding);
        let static_descriptor_set = unsafe {
            ctx.device
                .allocate_descriptor_sets(&allocation_info)
                .unwrap()
        }[0];

        let mut write_set_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&static_resources.scene.tlas.accel));

        let mut write = vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .dst_binding(0)
            .dst_set(static_descriptor_set)
            .push_next(&mut write_set_as);
        write.descriptor_count = 1;

        let blue_noise_image = image::open("./assets/bluenoise.png").unwrap();

        let blue_noise_image =
            ImageResource::new_from_data(blue_noise_image, vk::Format::R8G8B8A8_SRGB).unwrap();

        let geometry_infos = [vk::DescriptorBufferInfo::default()
            .buffer(static_resources.scene.geometry_infos.buffer)
            .offset(0)
            .range(static_resources.scene.geometry_infos.size)];
        let vertex_buffer = [vk::DescriptorBufferInfo::default()
            .buffer(static_resources.scene.vertex_buffer.buffer)
            .offset(0)
            .range(static_resources.scene.vertex_buffer.size)];
        let index_buffer = [vk::DescriptorBufferInfo::default()
            .buffer(static_resources.scene.index_buffer.buffer)
            .offset(0)
            .range(static_resources.scene.index_buffer.size)];
        let skybox = [vk::DescriptorImageInfo::default()
            .image_view(static_resources.scene.skybox.as_ref().unwrap().view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(static_resources.scene.skybox_sampler.unwrap())];
        let uniform = [vk::DescriptorBufferInfo::default()
            .buffer(static_resources.scene.uniform_buffer.buffer)
            .offset(0)
            .range(static_resources.scene.uniform_buffer.size)];
        let bluenoise_image_info = [vk::DescriptorImageInfo::default()
            .image_view(blue_noise_image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(static_resources.scene.skybox_sampler.unwrap())];

        let writes = [
            write,
            vk::WriteDescriptorSet::default()
                .buffer_info(&geometry_infos)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_binding(1)
                .dst_set(static_descriptor_set),
            vk::WriteDescriptorSet::default()
                .buffer_info(&vertex_buffer)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_binding(2)
                .dst_set(static_descriptor_set),
            vk::WriteDescriptorSet::default()
                .buffer_info(&index_buffer)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .dst_binding(3)
                .dst_set(static_descriptor_set),
            vk::WriteDescriptorSet::default()
                .image_info(&skybox)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(5)
                .dst_set(static_descriptor_set),
            vk::WriteDescriptorSet::default()
                .buffer_info(&uniform)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .dst_binding(6)
                .dst_set(static_descriptor_set),
            vk::WriteDescriptorSet::default()
                .image_info(&bluenoise_image_info)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(7)
                .dst_set(static_descriptor_set),
        ];

        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

        for (i, (image_index, sampler_index)) in
            static_resources.scene.texture_samplers.iter().enumerate()
        {
            let view = &static_resources.scene.texture_images[*image_index].view;
            let sampler = &static_resources.scene.samplers[*sampler_index];
            let img_info = vk::DescriptorImageInfo::default()
                .image_view(*view)
                .sampler(*sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            unsafe {
                ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_array_element(i as u32)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_binding(4)
                        .dst_set(static_descriptor_set)
                        .image_info(&[img_info])],
                    &[],
                )
            };
        }

        let mut frame_data: [FrameData; FRAMES_IN_FLIGHT as usize] =
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

            frame_data[i as usize] = FrameData {
                cmd: command_buffers[0],
                frame_number: i,
                command_pool,
            }
        }

        RenderGraph {
            resources: HashMap::new(),
            passes: Vec::new(),
            static_descriptor_set,
            static_descriptor_set_layout,
            static_resources,
            frame_data,
            frame_number: 0,
            back_buffer: "".to_string(),
        }
    }

    pub fn update_uniform(&mut self, gconst: &GConst) {
        self.static_resources
            .scene
            .uniform_buffer
            .copy_data_to_aligned_buffer(from_ref(gconst), 16)
            .unwrap();
    }

    pub(super) fn build_sync_resources(&mut self) {
        let mut wirten_resources: HashSet<&str> = HashSet::new();
        let mut last_writes: HashMap<&str, vk::PipelineStageFlags2> = HashMap::new();

        for pass in &mut self.passes {
            let mut sync_resources = Vec::new();
            for i in pass.input_resources.iter() {
                if wirten_resources.contains(i.as_str()) {
                    sync_resources.push(ResourceSync {
                        last_write: last_writes[i.as_str()],
                        new_layout: None, //TODO
                        old_layout: None, //TODO
                        resource_key: i.clone(),
                    });
                    wirten_resources.remove(i.as_str());
                }
            }

            for i in &pass.output_resources {
                wirten_resources.insert(i.as_str());
                last_writes.insert(
                    i.as_str(),
                    match pass.ty {
                        RenderPassType::Compute => vk::PipelineStageFlags2::COMPUTE_SHADER,
                        RenderPassType::Raytracing => {
                            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                        }
                        RenderPassType::Rasterization => vk::PipelineStageFlags2::MESH_SHADER_EXT,
                    },
                );
            }
            pass.sync_resources = sync_resources;
        }
    }

    pub fn add_image_resource(
        &mut self,
        name: &str,
        format: vk::Format,
        size: ImageSize,
    ) -> &mut Self {
        let ctx = Context::get();

        let (width, height) = match size {
            ImageSize::Custom { x, y } => (x, y),
            ImageSize::Viewport => (WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            ImageSize::ViewportFraction { x, y } => (
                (WINDOW_SIZE.x as f32 * x).ceil() as u32,
                (WINDOW_SIZE.y as f32 * y).ceil() as u32,
            ),
        };
        let image = ImageResource::new_2d(
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::GpuOnly,
            format,
            width,
            height,
        )
        .unwrap();

        ctx.execute_one_time_commands(|cmd| {
            let binding = [image
                .image
                .memory_barrier(vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL)];
            let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&binding);
            unsafe { ctx.device.cmd_pipeline_barrier2(*cmd, &dependency_info) };
        })
        .unwrap();

        let label = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(image.image.image)
            .object_name(unsafe { CStr::from_ptr(name.to_string().as_ptr() as _) });
        unsafe { ctx.debug_utils.set_debug_utils_object_name(&label) }.unwrap();

        self.resources.insert(
            name.to_string(),
            Resource {
                handle: ResourceTemporal::Single(ResourceData::Image(image)),
                ty: ResourceType::Image,
            },
        );
        self
    }

    pub fn add_buffer_resource(&mut self, name: &str, size: u64) -> &mut Self{
        let ctx = Context::get();
        let buffer = Buffer::new(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::GpuOnly,
            size,
        )
        .unwrap();

        let label = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(buffer.buffer)
            .object_name(unsafe { CStr::from_ptr(name.to_string().as_ptr() as _) });
        unsafe { ctx.debug_utils.set_debug_utils_object_name(&label) }.unwrap();

        self.resources.insert(
            name.to_string(),
            Resource {
                handle: ResourceTemporal::Single(ResourceData::Buffer(buffer)),
                ty: ResourceType::Buffer,
            },
        );
        self
    }

    pub fn add_temporal_image_resource(
        &mut self,
        name: &str,
        format: vk::Format,
        size: ImageSize,
    ) -> &mut Self{
        let ctx = Context::get();

        let (width, height) = match size {
            ImageSize::Custom { x, y } => (x, y),
            ImageSize::Viewport => (WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            ImageSize::ViewportFraction { x, y } => (
                (WINDOW_SIZE.x as f32 * x) as u32,
                (WINDOW_SIZE.y as f32 * y) as u32,
            ),
        };
        let mut images: [ResourceData; 2] = Default::default();
        for i in 0..2 {
            let image = ImageResource::new_2d(
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::GpuOnly,
                format,
                width,
                height,
            )
            .unwrap();

            ctx.execute_one_time_commands(|cmd| {
                let binding = [image
                    .image
                    .memory_barrier(vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL)];
                let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&binding);
                unsafe { ctx.device.cmd_pipeline_barrier2(*cmd, &dependency_info) };
            })
            .unwrap();

            let label = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(image.image.image)
                .object_name(unsafe { CStr::from_ptr(name.to_string().as_ptr() as _) });
            unsafe { ctx.debug_utils.set_debug_utils_object_name(&label) }.unwrap();

            images[i] = ResourceData::Image(image);
        }

        self.resources.insert(
            name.to_string(),
            Resource {
                handle: ResourceTemporal::Temporal(images),
                ty: ResourceType::Image,
            },
        );
        self
    }

    pub fn compile(
        &mut self,
        frame: FrameBuilder,
    ) {
        let raytracing_ctx = RayTracingContext::get();
        let ctx = Context::get();
        let mut max_sets = 0;
        let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();

        for pass in &frame.passes {
            let mut descriptors_needed = 0;
            for resource in pass
                .input_resources
                .iter()
                .chain(pass.output_resources.iter())
            {
                if matches!(
                    self.resources[resource].handle,
                    ResourceTemporal::Temporal(_)
                ) {
                    descriptors_needed |= 0x2
                } else {
                    descriptors_needed |= 0x1
                }
                match self.resources[resource].ty {
                    ResourceType::Buffer => pool_sizes
                        .entry(vk::DescriptorType::STORAGE_BUFFER)
                        .and_modify(|e| *e += descriptors_needed)
                        .or_insert(descriptors_needed),
                    ResourceType::Image => pool_sizes
                        .entry(vk::DescriptorType::STORAGE_IMAGE)
                        .and_modify(|e| *e += descriptors_needed)
                        .or_insert(descriptors_needed),
                };
            }
            if !pass.temporal_resources.is_empty() {
                descriptors_needed += 2;
                for resource in pass.temporal_resources.iter() {
                    match self.resources[resource].ty {
                        ResourceType::Buffer => pool_sizes
                            .entry(vk::DescriptorType::STORAGE_BUFFER)
                            .and_modify(|e| *e += 2)
                            .or_insert(2),
                        ResourceType::Image => pool_sizes
                            .entry(vk::DescriptorType::STORAGE_IMAGE)
                            .and_modify(|e| *e += 2)
                            .or_insert(2),
                    };
                }
            }
            max_sets += descriptors_needed;
        }

        let binding = pool_sizes
            .iter()
            .map(|e| {
                vk::DescriptorPoolSize::default()
                    .descriptor_count(*e.1)
                    .ty(*e.0)
            })
            .collect::<Vec<_>>();

        let descriptor_pool = if max_sets > 0 {
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(max_sets)
                .pool_sizes(binding.as_slice());

            Some(unsafe { ctx.device.create_descriptor_pool(&pool_info, None).unwrap() })
        } else {
            None
        };

        for pass in &frame.passes {
            let mut layouts = Vec::new();
            if pass.bindless_descriptor {
                layouts.push(self.static_descriptor_set_layout);
            }

            let mut static_descriptor = Vec::new();
            let mut current_temporal_descriptor = Vec::new();
            let mut temporal_previous_descriptor = Vec::new();

            for res in pass.input_resources.iter() {
                let resource = &self.resources[res];
                if matches!(resource.handle, ResourceTemporal::Single(_)) {
                    if !static_descriptor.contains(&res.as_str()) {
                        static_descriptor.push(res);
                    }
                } else {
                    if !current_temporal_descriptor.contains(&res.as_str()) {
                        current_temporal_descriptor.push(res);
                    }
                }
            }

            for (i, res) in pass.temporal_resources.iter().enumerate() {
                if !temporal_previous_descriptor.contains(&res.as_str()) {
                    temporal_previous_descriptor.push(res);
                }
            }

            for (i, res) in pass.output_resources.iter().enumerate() {
                let resource = &self.resources[res];
                if matches!(resource.handle, ResourceTemporal::Single(_)) {
                    if !static_descriptor.contains(&res.as_str()) {
                        static_descriptor.push(res);
                    }
                } else {
                    if !current_temporal_descriptor.contains(&res.as_str()) {
                        current_temporal_descriptor.push(res);
                    }
                }
            }

            log::info!("Compiled pass: {}-------------------------", pass.name);
            log::info!("Static bindings: ");
            for (i, binding) in static_descriptor.iter().enumerate() {
                log::info!("{}: {}", i, binding);
            }
            log::info!("Current Temporal bindings: ");
            for (i, binding) in current_temporal_descriptor.iter().enumerate() {
                log::info!("{}: {}", i, binding);
            }
            log::info!("Temporal bindings: ");
            for (i, binding) in temporal_previous_descriptor.iter().enumerate() {
                log::info!("{}: {}", i, binding);
            }

            let static_resources = static_descriptor
                .into_iter()
                .map(|e| &self.resources[e])
                .collect::<Vec<_>>();
            let current_temporal_resources = current_temporal_descriptor
                .into_iter()
                .map(|e| &self.resources[e])
                .collect::<Vec<_>>();
            let temporal_previous_resources = temporal_previous_descriptor
                .into_iter()
                .map(|e| &self.resources[e])
                .collect::<Vec<_>>();

            let lamda = |(i, e): (usize, &&Resource)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(match e.ty {
                        ResourceType::Buffer => vk::DescriptorType::STORAGE_BUFFER,
                        ResourceType::Image => vk::DescriptorType::STORAGE_IMAGE,
                    })
            };

            let static_bindings = static_resources
                .iter()
                .enumerate()
                .map(lamda)
                .collect::<Vec<_>>();
            let static_temporal_bindings = current_temporal_resources
                .iter()
                .enumerate()
                .map(lamda)
                .collect::<Vec<_>>();
            let temporal_bindings = temporal_previous_resources
                .iter()
                .enumerate()
                .map(lamda)
                .collect::<Vec<_>>();

            let mut allocat_layouts = Vec::new();

            let mut static_temporal_offset = 0;
            let mut temporal_offset = 0;

            if !static_bindings.is_empty() {
                let create_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(&static_bindings);
                let layout = unsafe {
                    ctx.device
                        .create_descriptor_set_layout(&create_info, None)
                        .unwrap()
                };
                allocat_layouts.push(layout);
                layouts.push(layout);
                static_temporal_offset += 1;
                temporal_offset += 1;
            }
            if !static_temporal_bindings.is_empty() {
                let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&static_temporal_bindings);
                let layout = unsafe {
                    ctx.device
                        .create_descriptor_set_layout(&create_info, None)
                        .unwrap()
                };
                allocat_layouts.push(layout);
                allocat_layouts.push(layout);
                layouts.push(layout);
                temporal_offset += 2;
            }
            if !temporal_bindings.is_empty() {
                let create_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(&temporal_bindings);
                let layout = unsafe {
                    ctx.device
                        .create_descriptor_set_layout(&create_info, None)
                        .unwrap()
                };
                allocat_layouts.push(layout);
                allocat_layouts.push(layout);
                layouts.push(layout);
            }

            let layout_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(layouts.as_slice());
            let layout = unsafe {
                ctx.device
                    .create_pipeline_layout(&layout_info, None)
                    .unwrap()
            };

            let (command, pipeline) = match &pass.ty {
                PassBuilderType::Compute(compute_pass) => {
                    let create_infos = vk::ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(
                            ctx.create_shader_stage(
                                vk::ShaderStageFlags::COMPUTE,
                                format!("./shaders/bin/{}", compute_pass.shader_source).as_str(),
                            )
                            .unwrap(),
                        );
                    let pipeline = unsafe {
                        ctx.device
                            .create_compute_pipelines(
                                vk::PipelineCache::null(),
                                &[create_infos],
                                None,
                            )
                            .unwrap()
                    }[0];
                    match compute_pass.dispatch {
                        ComputePassBuilderCommand::Dispatch { x, y, z } => {
                            (RenderPassCommand::Compute { x, y, z }, pipeline)
                        }
                        ComputePassBuilderCommand::DispatchIndirect(indirect_buffer) => (
                            RenderPassCommand::ComputeIndirect { indirect_buffer },
                            pipeline,
                        ),
                    }
                }
                PassBuilderType::Raytracing(raytracing_pass) => {
                    let (pipeline, group_info) = raytracing_ctx
                        .create_raytracing_pipeline(
                            layout,
                            &[
                                RayTracingShaderCreateInfo {
                                    source: &[(
                                        raytracing_pass.ray_gen_source,
                                        vk::ShaderStageFlags::RAYGEN_KHR,
                                    )],
                                    group: RayTracingShaderGroup::RayGen,
                                },
                                RayTracingShaderCreateInfo {
                                    source: &[(
                                        raytracing_pass.override_miss_shader_source.unwrap_or("./../../../shaders/bin/default_miss.slang.spv"),
                                        vk::ShaderStageFlags::MISS_KHR,
                                    )],
                                    group: RayTracingShaderGroup::Miss,
                                },
                                RayTracingShaderCreateInfo {
                                    source: &[(
                                        raytracing_pass.override_miss_shader_source.unwrap_or("./../../../shaders/bin/default_hit.slang.spv"),
                                        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                                    )],
                                    group: RayTracingShaderGroup::Hit,
                                },
                            ],
                        )
                        .unwrap();
                    let shader_binding_table =
                        ShaderBindingTable::new(&pipeline, &group_info)
                            .unwrap();
                    (
                        RenderPassCommand::Raytracing {
                            shader_binding_table,
                            x: raytracing_pass.launch_size[0],
                            y: raytracing_pass.launch_size[1],
                        },
                        pipeline,
                    )
                }
                PassBuilderType::Rasterization(raster_pass) => {
                    let color_attachments = raster_pass
                        .attachments
                        .iter()
                        .map(|e| {
                            let resource = &self.resources[&e.0];
                            ColorAttachment {
                                clear: e.1,
                                format: resource.get_format().unwrap_or_else(|| {
                                    log::error!("Color Attachment Format not found");
                                    vk::Format::UNDEFINED
                                }),
                                resource: e.0.clone(),
                            }
                        })
                        .collect::<Vec<ColorAttachment>>();
                    
                    let formats = color_attachments.iter().map(|e| e.format).collect::<Vec<_>>();

                    let depth_attachment = if let Some(attachment) = &raster_pass.depth_attachment {
                        Some(DepthStencilAttachment {
                            clear: attachment.1,
                            format: self.resources[&attachment.0].get_format().unwrap_or_else(|| {
                                log::error!("Depth Attachment Format not found");
                                vk::Format::UNDEFINED
                            }),
                            resource: attachment.0.clone(),
                        })
                    }else {None};

                    let stencil_attachment = if let Some(attachment) = &raster_pass.stencil_attachment {
                        Some(DepthStencilAttachment {
                            clear: attachment.1,
                            format: self.resources[&attachment.0].get_format().unwrap_or_else(|| {
                                log::error!("Stencil Attachment Format not found");
                                vk::Format::UNDEFINED
                            }),
                            resource: attachment.0.clone(),
                        })
                    }else {None};

                    let mut rendering = vk::PipelineRenderingCreateInfo::default()
                        .color_attachment_formats(formats.as_slice())
                        .depth_attachment_format(if let Some(da) = &depth_attachment {da.format} else {vk::Format::UNDEFINED})
                        .stencil_attachment_format(if let Some(da) = &stencil_attachment {da.format} else {vk::Format::UNDEFINED});

                    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                        .dynamic_states(&[
                            vk::DynamicState::VIEWPORT_WITH_COUNT,
                            vk::DynamicState::SCISSOR_WITH_COUNT,
                            vk::DynamicState::DEPTH_CLAMP_ENABLE_EXT,
                            vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
                            vk::DynamicState::POLYGON_MODE_EXT,
                            vk::DynamicState::CULL_MODE,
                            vk::DynamicState::FRONT_FACE,
                            vk::DynamicState::DEPTH_BIAS_ENABLE,
                            vk::DynamicState::DEPTH_BIAS,
                            vk::DynamicState::LINE_WIDTH,
                            vk::DynamicState::LOGIC_OP_ENABLE_EXT,
                            vk::DynamicState::LOGIC_OP_EXT,
                            vk::DynamicState::BLEND_CONSTANTS,
                            vk::DynamicState::COLOR_BLEND_EQUATION_EXT,
                            vk::DynamicState::COLOR_WRITE_MASK_EXT,
                            vk::DynamicState::BLEND_CONSTANTS,
                            vk::DynamicState::DEPTH_TEST_ENABLE,
                            vk::DynamicState::DEPTH_WRITE_ENABLE,
                            vk::DynamicState::DEPTH_COMPARE_OP,
                            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
                            vk::DynamicState::STENCIL_TEST_ENABLE,
                            vk::DynamicState::STENCIL_OP,
                            vk::DynamicState::DEPTH_BOUNDS,
                        ]);

                    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

                    let stages = [
                        ctx.create_shader_stage(
                            vk::ShaderStageFlags::MESH_EXT,
                            format!("./shaders/bin/{}", raster_pass.mesh_shader).as_str(),
                        )
                        .unwrap(),
                        ctx.create_shader_stage(
                            vk::ShaderStageFlags::FRAGMENT,
                            format!("./shaders/bin/{}", raster_pass.fragment_shader).as_str(),
                        )
                        .unwrap()
                    ];
                    let create_info = vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .layout(layout)
                        .dynamic_state(&dynamic_state)
                        .multisample_state(&multisampling);
                    create_info.push_next(&mut rendering);

                    let pipeline = unsafe {
                        ctx.device
                            .create_graphics_pipelines(
                                vk::PipelineCache::null(),
                                &[create_info],
                                None,
                            )
                            .unwrap()
                    }[0];

                    (RenderPassCommand::Raster{
                        color_attachments,
                        depth_attachment,
                        render_area: raster_pass.render_area.unwrap_or(UVec2::new(WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32)),
                        stencil_attachment,
                    }, pipeline)
                }
            };

            let descriptor_sets = if let Some(descriptor_pool) = descriptor_pool {
                let descriptor_allocate_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(allocat_layouts.as_slice());
                unsafe {
                    ctx.device
                        .allocate_descriptor_sets(&descriptor_allocate_info)
                }
                .unwrap()
            } else {
                Vec::new()
            };

            if !static_bindings.is_empty() {
                for (binding, resource) in static_resources.iter().enumerate() {
                    match &resource.handle {
                        ResourceTemporal::Single(resource) => {
                            match resource {
                                ResourceData::Buffer(buffer) => {
                                    let buffer_info = vk::DescriptorBufferInfo::default()
                                        .buffer(buffer.buffer)
                                        .offset(0)
                                        .range(buffer.size);
                                    let buffer_infos = [buffer_info];
                                    let write = vk::WriteDescriptorSet::default()
                                        .descriptor_count(1)
                                        .dst_array_element(0)
                                        .dst_set(descriptor_sets[0])
                                        .dst_binding(binding as u32)
                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                        .buffer_info(&buffer_infos);
                                    unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                                }
                                ResourceData::Image(image) => {
                                    let image_info = vk::DescriptorImageInfo::default()
                                        .image_layout(vk::ImageLayout::GENERAL)
                                        .image_view(image.view);
                                    let image_infos = [image_info];
                                    let write = vk::WriteDescriptorSet::default()
                                        .descriptor_count(1)
                                        .dst_array_element(0)
                                        .dst_set(descriptor_sets[0])
                                        .dst_binding(binding as u32)
                                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                        .image_info(&image_infos);
                                    unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                                }
                            };
                        }
                        ResourceTemporal::Temporal(_) => {
                            unreachable!()
                        }
                    }
                }
            }
            if !static_temporal_bindings.is_empty() {
                for (binding, resource) in current_temporal_resources.iter().enumerate() {
                    match &resource.handle {
                        ResourceTemporal::Temporal(resource) => {
                            for i in 0..2 {
                                match &resource[i] {
                                    ResourceData::Buffer(buffer) => {
                                        let buffer_info = vk::DescriptorBufferInfo::default()
                                            .buffer(buffer.buffer)
                                            .offset(0)
                                            .range(buffer.size);
                                        let buffer_infos = [buffer_info];
                                        let write = vk::WriteDescriptorSet::default()
                                            .descriptor_count(1)
                                            .dst_array_element(0)
                                            .dst_set(descriptor_sets[static_temporal_offset + i])
                                            .dst_binding(binding as u32)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(&buffer_infos);
                                        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                                    }
                                    ResourceData::Image(image) => {
                                        let image_info = vk::DescriptorImageInfo::default()
                                            .image_layout(vk::ImageLayout::GENERAL)
                                            .image_view(image.view);
                                        let image_infos = [image_info];
                                        let write = vk::WriteDescriptorSet::default()
                                            .descriptor_count(1)
                                            .dst_array_element(0)
                                            .dst_set(descriptor_sets[static_temporal_offset + i])
                                            .dst_binding(binding as u32)
                                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                            .image_info(&image_infos);
                                        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                                    }
                                };
                            }
                        }
                        ResourceTemporal::Single(_) => {
                            unreachable!()
                        }
                    }
                }
            }
            if !temporal_bindings.is_empty() {
                for (binding, resource) in temporal_previous_resources.iter().enumerate() {
                    match &resource.handle {
                        ResourceTemporal::Temporal(resource) => {
                            for i in 0..2 {
                                match &resource[i] {
                                    ResourceData::Buffer(buffer) => {
                                        let buffer_info = vk::DescriptorBufferInfo::default()
                                            .buffer(buffer.buffer)
                                            .offset(0)
                                            .range(buffer.size);
                                        let buffer_infos = [buffer_info];
                                        let write = vk::WriteDescriptorSet::default()
                                            .descriptor_count(1)
                                            .dst_array_element(0)
                                            .dst_set(descriptor_sets[temporal_offset + i])
                                            .dst_binding(binding as u32)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(&buffer_infos);
                                        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                                    }
                                    ResourceData::Image(image) => {
                                        let image_info = vk::DescriptorImageInfo::default()
                                            .image_layout(vk::ImageLayout::GENERAL)
                                            .image_view(image.view);
                                        let image_infos = [image_info];
                                        let write = vk::WriteDescriptorSet::default()
                                            .descriptor_count(1)
                                            .dst_array_element(0)
                                            .dst_set(descriptor_sets[temporal_offset + i])
                                            .dst_binding(binding as u32)
                                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                            .image_info(&image_infos);
                                        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                                    }
                                };
                            }
                        }
                        ResourceTemporal::Single(_) => {
                            unreachable!()
                        }
                    }
                }
            }

            let mut input_resources = pass.input_resources.clone();
            input_resources.append(&mut pass.temporal_resources.clone());

            let compiled_pass = RenderPass {
                command,
                descriptor_set: if !static_bindings.is_empty() {
                    Some(descriptor_sets[0])
                } else {
                    None
                },
                layout,
                pipeline,
                sync_resources: vec![],
                temporal_descriptor_sets: if !static_temporal_bindings.is_empty() {
                    Some([
                        descriptor_sets[static_temporal_offset],
                        descriptor_sets[static_temporal_offset + 1],
                    ])
                } else {
                    None
                },
                temporal_descriptor_sets2: if !temporal_bindings.is_empty() {
                    Some([
                        descriptor_sets[temporal_offset],
                        descriptor_sets[temporal_offset + 1],
                    ])
                } else {
                    None
                },
                bindless_descriptor: pass.bindless_descriptor,
                name: pass.name.clone(),
                active: pass.active,
                input_resources,
                output_resources: pass.output_resources.clone(),
                ty: match pass.ty {
                    PassBuilderType::Compute(_) => RenderPassType::Compute,
                    PassBuilderType::Raytracing(_) => RenderPassType::Raytracing,
                    PassBuilderType::Rasterization(_) => RenderPassType::Rasterization,
                },
            };
            self.passes.push(compiled_pass);
        }
        self.build_sync_resources();
        self.back_buffer = frame.back_buffer;
    }
}
