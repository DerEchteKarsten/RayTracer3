use std::{
    collections::{HashMap, HashSet},
    ffi::CStr,
    mem::MaybeUninit,
    num::NonZero,
    ops::DerefMut,
    rc::Rc,
    slice::from_ref,
    sync::Arc,
};

use anyhow::Result;
use ash::vk::{self, ImageLayout};
use gltf::json::extensions::image::Image;
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocationSizes, AllocatorDebugSettings,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::cell::RefCell;

use crate::{assets::model::Model, imgui::ImGui, GConst, WINDOW_SIZE};

use super::{
    raytracing::{
        AccelerationStructure, RayTracingContext, RayTracingShaderCreateInfo,
        RayTracingShaderGroup, ShaderBindingTable,
    },
    swapchain::Swapchain,
    utils::{Buffer, ImageResource},
    vulkan_context::Context,
};

pub const FRAMES_IN_FLIGHT: u64 = 3;

#[derive(PartialEq, Eq)]
pub enum ResourceType {
    Buffer,
    Image,
}

enum ResourceTemporal {
    Single(ResourceData),
    Temporal([ResourceData; 2]),
}

enum ResourceData {
    Buffer(Buffer),
    Image(ImageResource),
}

impl Default for ResourceData {
    fn default() -> Self {
        Self::Image(ImageResource::default())
    }
}

pub struct Resource {
    pub ty: ResourceType,
    handle: ResourceTemporal,
}

impl Resource {
    fn get_current(&self, graph: &RenderGraph) -> &ResourceData {
        match &self.handle {
            ResourceTemporal::Single(data) => data,
            ResourceTemporal::Temporal(data) => &data[(graph.frame_number % 2) as usize],
        }
    }
}

pub struct SceneResources {
    //TODO
    pub(crate) vertex_buffer: Buffer,
    pub(crate) index_buffer: Buffer,
    pub(crate) geometry_infos: Buffer,
    pub(crate) samplers: Vec<vk::Sampler>,
    pub(crate) texture_images: Vec<ImageResource>,
    pub(crate) texture_samplers: Vec<(usize, usize)>,

    pub(crate) tlas: AccelerationStructure,
    pub(crate) uniform_buffer: Buffer,
    pub(crate) skybox: ImageResource,
    pub(crate) skybox_sampler: vk::Sampler,
}

struct ImguiResources {
    renderpass: vk::RenderPass,
    frame_buffers: Vec<vk::Framebuffer>,
    renderer: imgui_rs_vulkan_renderer::Renderer,
}

pub struct StaticResources {
    swapchain: Swapchain,
    imgui_resources: ImguiResources,
    scene: SceneResources,
    frame_timeline_semaphore: vk::Semaphore,
}

impl StaticResources {
    fn new(ctx: &Context, scene: SceneResources, imgui: &mut ImGui) -> Result<Self> {
        let swapchain = Swapchain::new(&ctx)?;

        let mut timeline_create_info = vk::SemaphoreTypeCreateInfo::default()
            .initial_value(FRAMES_IN_FLIGHT - 1)
            .semaphore_type(vk::SemaphoreType::TIMELINE);

        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_create_info);
        let frame_timeline_semaphore =
            unsafe { ctx.device.create_semaphore(&create_info, None).unwrap() };

        let attachments = [vk::AttachmentDescription::default()
            .initial_layout(ImageLayout::PRESENT_SRC_KHR)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .format(swapchain.format)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .samples(vk::SampleCountFlags::TYPE_1)];
        let attachment = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(ImageLayout::GENERAL)];
        let subpasses = [vk::SubpassDescription::default()
            .color_attachments(&attachment)
            .input_attachments(&[])];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let renderpass = unsafe {
            ctx.device
                .create_render_pass(&render_pass_create_info, None)
                .unwrap()
        };

        let mut frame_buffers = Vec::new();

        for (i, image) in swapchain.images.iter().enumerate() {
            let attachment = [image.view];
            let frame_buffer_info = vk::FramebufferCreateInfo::default()
                .attachments(&attachment)
                .height(WINDOW_SIZE.y as u32)
                .width(WINDOW_SIZE.x as u32)
                .layers(1)
                .render_pass(renderpass);
            frame_buffers.push(unsafe {
                ctx.device
                    .create_framebuffer(&frame_buffer_info, None)
                    .unwrap()
            })
        }

        let renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
            &ctx.instance,
            ctx.physical_device.handel,
            ctx.device.clone(),
            ctx.graphics_queue,
            ctx.command_pool,
            renderpass,
            &mut imgui.context,
            None,
        )
        .unwrap();

        Ok(Self {
            swapchain,
            scene,
            frame_timeline_semaphore,
            imgui_resources: ImguiResources {
                frame_buffers,
                renderpass,
                renderer,
            },
        })
    }
}

enum RenderPassCommand {
    Raytracing {
        shader_binding_table: ShaderBindingTable,
        x: u32,
        y: u32,
    },
    Raster(),
    Compute {
        x: u32,
        y: u32,
        z: u32,
    },
    ComputeIndirect {
        indirect_buffer: vk::Buffer,
    },
    Custom(Box<dyn Fn(&vk::CommandBuffer, &RenderPass) -> ()>),
}

enum RenderPassType {
    Compute,
    Raytracing,
}

struct RenderPass {
    command: RenderPassCommand,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    bindless_descriptor: bool,
    descriptor_set: Option<vk::DescriptorSet>,
    temporal_descriptor_sets: Option<[vk::DescriptorSet; 2]>,
    temporal_descriptor_sets2: Option<[vk::DescriptorSet; 2]>,
    input_resources: Vec<String>,
    output_resources: Vec<String>,
    sync_resources: Vec<ResourceSync>,
    name: String,
    active: bool,
    ty: RenderPassType,
}

struct ResourceSync {
    resource_key: String,
    last_write: vk::PipelineStageFlags2,
    old_layout: Option<vk::ImageLayout>,
    new_layout: Option<vk::ImageLayout>,
}

pub struct RenderGraph {
    pub resources: HashMap<String, Resource>,
    pub static_resources: StaticResources,
    passes: Vec<RenderPass>,
    pub back_buffer: String,

    static_descriptor_set: vk::DescriptorSet,
    static_descriptor_set_layout: vk::DescriptorSetLayout,

    frame_data: [FrameData; FRAMES_IN_FLIGHT as usize],

    pub frame_number: u64,
}

struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

pub enum ImageSize {
    Custom { x: u32, y: u32 },
    Viewport,
    ViewportFraction { x: f32, y: f32 },
}

impl RenderGraph {
    const RAYMISS_BYTES: &[u8] = include_bytes!("./../../shaders/bin/default_miss.slang.spv");
    const RAYHIT_BYTES: &[u8] = include_bytes!("./../../shaders/bin/default_hit.slang.spv");

    pub fn swpachain_format(&self) -> vk::Format {
        self.static_resources.swapchain.format
    }

    pub fn num_swpachain_images(&self) -> u32 {
        self.static_resources.swapchain.images.len() as u32
    }

    pub fn new(
        ctx: &mut Context,
        scene_resources: SceneResources,
        imgui: &mut ImGui,
    ) -> RenderGraph {
        let static_resources = StaticResources::new(&ctx, scene_resources, imgui).unwrap();
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
            ImageResource::new_from_data(ctx, blue_noise_image, vk::Format::R8G8B8A8_SRGB).unwrap();

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
            .image_view(static_resources.scene.skybox.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(static_resources.scene.skybox_sampler)];
        let uniform = [vk::DescriptorBufferInfo::default()
            .buffer(static_resources.scene.uniform_buffer.buffer)
            .offset(0)
            .range(static_resources.scene.uniform_buffer.size)];
        let bluenoise_image_info = [vk::DescriptorImageInfo::default()
            .image_view(blue_noise_image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(static_resources.scene.skybox_sampler)];

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

    fn build_sync_resources(&mut self) {
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
                    },
                );
            }
            pass.sync_resources = sync_resources;
        }
    }

    pub fn add_image_resource(
        &mut self,
        ctx: &mut Context,
        name: &str,
        format: vk::Format,
        size: ImageSize,
    ) {
        let (width, height) = match size {
            ImageSize::Custom { x, y } => (x, y),
            ImageSize::Viewport => (WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            ImageSize::ViewportFraction { x, y } => (
                (WINDOW_SIZE.x as f32 * x).ceil() as u32,
                (WINDOW_SIZE.y as f32 * y).ceil() as u32,
            ),
        };
        let image = ImageResource::new_2d(
            ctx,
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
    }

    pub fn add_buffer_resource(&mut self, ctx: &mut Context, name: &str, size: u64) {
        let buffer = Buffer::new(
            ctx,
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
    }

    pub fn add_temporal_image_resource(
        &mut self,
        ctx: &mut Context,
        name: &str,
        format: vk::Format,
        size: ImageSize,
    ) {
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
                ctx,
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
    }

    pub fn compile(
        &mut self,
        frame: FrameBuilder,
        ctx: &mut Context,
        raytracing_ctx: &RayTracingContext,
    ) {
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
                    let raygen_source =
                        std::fs::read(format!("./shaders/bin/{}", raytracing_pass.ray_gen_source))
                            .unwrap();
                    let miss_source = raytracing_pass
                        .override_miss_shader_source
                        .and_then(|e| Some(std::fs::read(format!("./shaders/bin/{}", e)).unwrap()))
                        .unwrap_or(Self::RAYMISS_BYTES.to_vec());
                    let chit_source = raytracing_pass
                        .override_hit_shader_source
                        .and_then(|e| Some(std::fs::read(format!("./shaders/bin/{}", e)).unwrap()))
                        .unwrap_or(Self::RAYHIT_BYTES.to_vec());

                    let (pipeline, group_info) = raytracing_ctx
                        .create_raytracing_pipeline(
                            ctx,
                            layout,
                            &[
                                RayTracingShaderCreateInfo {
                                    source: &[(
                                        raygen_source.as_slice(),
                                        vk::ShaderStageFlags::RAYGEN_KHR,
                                    )],
                                    group: RayTracingShaderGroup::RayGen,
                                },
                                RayTracingShaderCreateInfo {
                                    source: &[(
                                        miss_source.as_slice(),
                                        vk::ShaderStageFlags::MISS_KHR,
                                    )],
                                    group: RayTracingShaderGroup::Miss,
                                },
                                RayTracingShaderCreateInfo {
                                    source: &[(
                                        chit_source.as_slice(),
                                        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                                    )],
                                    group: RayTracingShaderGroup::Hit,
                                },
                            ],
                        )
                        .unwrap();
                    let shader_binding_table =
                        ShaderBindingTable::new(ctx, raytracing_ctx, &pipeline, &group_info)
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
                },
            };
            self.passes.push(compiled_pass);
        }
        self.build_sync_resources();
        self.back_buffer = frame.back_buffer;
    }

    fn begin_frame(&self, ctx: &Context, frame_in_flight: u64) -> u32 {
        let frame = &self.frame_data[frame_in_flight as usize];
        let semaphore = [self.static_resources.frame_timeline_semaphore];
        let values = [frame.frame_number];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphore)
            .values(&values);
        unsafe { ctx.device.wait_semaphores(&wait_info, 1000000000).unwrap() };
        unsafe {
            ctx.device
                .reset_command_pool(frame.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap()
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            ctx.device
                .begin_command_buffer(frame.cmd, &begin_info)
                .unwrap()
        };

        unsafe {
            self.static_resources
                .swapchain
                .ash_swapchain
                .acquire_next_image(
                    self.static_resources.swapchain.vk_swapchain,
                    1000000000,
                    self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                        .image_availible_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
        }
        .0
    }

    fn execute_passes(
        &self,
        ctx: &Context,
        raytracing_ctx: &RayTracingContext,
        frame_in_flight: u64,
        swapchain_image_index: u32,
    ) {
        let frame = &self.frame_data[frame_in_flight as usize];
        unsafe {
            for pass in &self.passes {
                if !pass.active {
                    continue;
                }
                let bind_point = match pass.command {
                    RenderPassCommand::Compute { .. }
                    | RenderPassCommand::ComputeIndirect { .. } => {
                        Some(vk::PipelineBindPoint::COMPUTE)
                    }
                    RenderPassCommand::Raytracing { .. } => {
                        Some(vk::PipelineBindPoint::RAY_TRACING_KHR)
                    }
                    RenderPassCommand::Raster { .. } => Some(vk::PipelineBindPoint::GRAPHICS),
                    RenderPassCommand::Custom { .. } => None,
                };

                if !pass.sync_resources.is_empty() {
                    let mut buffers = Vec::new();
                    let mut images = Vec::new();
                    for sync_resource in &pass.sync_resources {
                        let resource = &self.resources[&sync_resource.resource_key];
                        match &resource.get_current(self) {
                            ResourceData::Buffer(buffer) => {
                                buffers.push(
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.buffer)
                                        .size(buffer.size)
                                        .src_stage_mask(sync_resource.last_write)
                                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                                        .offset(0)
                                        .dst_stage_mask(match bind_point {
                                            Some(vk::PipelineBindPoint::COMPUTE) => {
                                                vk::PipelineStageFlags2::COMPUTE_SHADER
                                            }
                                            Some(vk::PipelineBindPoint::RAY_TRACING_KHR) => {
                                                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                                            }
                                            Some(vk::PipelineBindPoint::GRAPHICS) => {
                                                vk::PipelineStageFlags2::VERTEX_SHADER
                                            } //TODO
                                            _ => vk::PipelineStageFlags2::TOP_OF_PIPE,
                                        })
                                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                                );
                            }
                            ResourceData::Image(image) => {
                                let mut info = vk::ImageMemoryBarrier2::default()
                                    .image(image.image.image)
                                    .src_stage_mask(sync_resource.last_write)
                                    .subresource_range(
                                        vk::ImageSubresourceRange::default()
                                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                                            .base_array_layer(0)
                                            .base_mip_level(0)
                                            .layer_count(1)
                                            .level_count(1),
                                    )
                                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                                    .dst_stage_mask(match bind_point {
                                        Some(vk::PipelineBindPoint::COMPUTE) => {
                                            vk::PipelineStageFlags2::COMPUTE_SHADER
                                        }
                                        Some(vk::PipelineBindPoint::RAY_TRACING_KHR) => {
                                            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                                        }
                                        Some(vk::PipelineBindPoint::GRAPHICS) => {
                                            vk::PipelineStageFlags2::VERTEX_SHADER
                                        } //TODO
                                        _ => vk::PipelineStageFlags2::TOP_OF_PIPE,
                                    })
                                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

                                if let Some(old_layout) = sync_resource.old_layout
                                    && let Some(new_layout) = sync_resource.new_layout
                                {
                                    info = info.old_layout(old_layout).new_layout(new_layout);
                                }

                                images.push(info);
                            }
                        }
                    }

                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(&buffers)
                        .image_memory_barriers(&images)
                        .dependency_flags(vk::DependencyFlags::empty());

                    let lable_message = &format!(
                        "Barrier for {}\0",
                        pass.sync_resources
                            .iter()
                            .fold("".to_owned(), |acc, e| format!(
                                "{}, {}",
                                acc,
                                e.resource_key.as_str()
                            ))
                    );
                    let label = vk::DebugUtilsLabelEXT::default()
                        .label_name(CStr::from_bytes_with_nul(lable_message.as_bytes()).unwrap())
                        .color([1.0, 1.0, 1.0, 1.0]);
                    ctx.debug_utils
                        .cmd_insert_debug_utils_label(frame.cmd, &label);

                    ctx.device
                        .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
                }

                if let Some(bind_point) = bind_point {
                    let mut descriptor_sets = Vec::new();
                    if pass.bindless_descriptor {
                        descriptor_sets.push(self.static_descriptor_set);
                    }
                    if let Some(descriptor) = pass.descriptor_set {
                        descriptor_sets.push(descriptor);
                    }
                    if let Some(temporal_descriptors) = pass.temporal_descriptor_sets {
                        descriptor_sets.push(temporal_descriptors[self.frame_number as usize % 2]);
                    }
                    if let Some(temporal_descriptors2) = pass.temporal_descriptor_sets2 {
                        descriptor_sets.push(temporal_descriptors2[self.frame_number as usize % 2]);
                    }

                    if !descriptor_sets.is_empty() {
                        ctx.device.cmd_bind_descriptor_sets(
                            frame.cmd,
                            bind_point,
                            pass.layout,
                            0,
                            &descriptor_sets,
                            &[],
                        );
                    }
                    ctx.device
                        .cmd_bind_pipeline(frame.cmd, bind_point, pass.pipeline);
                }

                let label = vk::DebugUtilsLabelEXT::default()
                    .label_name(CStr::from_ptr(pass.name.as_ptr() as _))
                    .color([1.0, 1.0, 1.0, 1.0]);
                ctx.debug_utils
                    .cmd_insert_debug_utils_label(frame.cmd, &label);

                match &pass.command {
                    RenderPassCommand::Compute { x, y, z } => {
                        ctx.device.cmd_dispatch(frame.cmd, *x, *y, *z);
                    }
                    RenderPassCommand::Raster {} => {}
                    RenderPassCommand::Raytracing {
                        x,
                        y,
                        shader_binding_table,
                    } => {
                        let call_region = vk::StridedDeviceAddressRegionKHR::default();
                        raytracing_ctx.pipeline_fn.cmd_trace_rays(
                            frame.cmd,
                            &shader_binding_table.raygen_region,
                            &shader_binding_table.miss_region,
                            &shader_binding_table.hit_region,
                            &call_region,
                            *x,
                            *y,
                            1,
                        );
                    }
                    RenderPassCommand::ComputeIndirect { indirect_buffer } => {
                        ctx.device
                            .cmd_dispatch_indirect(frame.cmd, *indirect_buffer, 0);
                    }
                    RenderPassCommand::Custom(func) => {
                        func(&frame.cmd, &pass);
                    }
                }
            }

            let copy_image = |image: &ImageResource| {
                let image_barriers = [
                    self.static_resources.swapchain.images[swapchain_image_index as usize]
                        .image
                        .memory_barrier(
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        ),
                    image.image.memory_barrier(
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ),
                ];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info);

                image.image.blit(
                    &ctx,
                    &frame.cmd,
                    &self.static_resources.swapchain.images[swapchain_image_index as usize].image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                let image_barriers = [
                    self.static_resources.swapchain.images[swapchain_image_index as usize]
                        .image
                        .memory_barrier(
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::PRESENT_SRC_KHR,
                        ),
                    image.image.memory_barrier(
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                    ),
                ];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
            };

            if let Some(entry) = self.resources.get(&self.back_buffer) {
                match &entry.handle {
                    ResourceTemporal::Single(res) => match res {
                        ResourceData::Buffer(_) => {
                            panic!("Back Buffer is Buffer")
                        }
                        ResourceData::Image(image) => copy_image(image),
                    },
                    ResourceTemporal::Temporal(res) => match &res[(self.frame_number % 2) as usize]
                    {
                        ResourceData::Buffer(_) => {
                            panic!("Back Buffer is Buffer")
                        }
                        ResourceData::Image(image) => copy_image(image),
                    },
                }
            }
        }
    }

    fn submit_frame(&mut self, ctx: &Context, frame_in_flight: u64, swapchain_image_index: u32) {
        let frame = &mut self.frame_data[frame_in_flight as usize];

        unsafe { ctx.device.end_command_buffer(frame.cmd).unwrap() };

        let wait_semaphores = [vk::SemaphoreSubmitInfo::default()
            .semaphore(
                self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                    .image_availible_semaphore,
            )
            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)];

        let signal_frame_value = frame.frame_number + FRAMES_IN_FLIGHT;
        frame.frame_number = signal_frame_value;

        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                        .render_finished_semaphore,
                )
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.static_resources.frame_timeline_semaphore)
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .value(signal_frame_value),
        ];

        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.cmd)];

        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signal_semaphores)
            .wait_semaphore_infos(&wait_semaphores);

        unsafe {
            ctx.device
                .queue_submit2(ctx.graphics_queue, &[submit], vk::Fence::null())
                .unwrap()
        };

        let binding = [
            self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                .render_finished_semaphore,
        ];
        let swapchains = [self.static_resources.swapchain.vk_swapchain];
        let image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .image_indices(&image_indices)
            .swapchains(&swapchains)
            .wait_semaphores(&binding);
        unsafe {
            self.static_resources
                .swapchain
                .ash_swapchain
                .queue_present(ctx.graphics_queue, &present_info)
                .unwrap()
        };
        self.frame_number += 1;
    }

    fn render_imgui(
        &mut self,
        ctx: &Context,
        draw_data: &imgui::DrawData,
        cmd: vk::CommandBuffer,
        swapchain_image_index: u32,
    ) {
        unsafe {
            let begin_info = vk::RenderPassBeginInfo::default()
                .framebuffer(
                    self.static_resources.imgui_resources.frame_buffers
                        [swapchain_image_index as usize],
                )
                .render_pass(self.static_resources.imgui_resources.renderpass)
                .render_area(
                    vk::Rect2D::default()
                        .extent(vk::Extent2D {
                            width: WINDOW_SIZE.x as u32,
                            height: WINDOW_SIZE.y as u32,
                        })
                        .offset(vk::Offset2D { x: 0, y: 0 }),
                );
            ctx.device
                .cmd_begin_render_pass(cmd, &begin_info, vk::SubpassContents::INLINE);
            self.static_resources
                .imgui_resources
                .renderer
                .cmd_draw(cmd, draw_data)
                .unwrap();
            ctx.device.cmd_end_render_pass(cmd);
        }
    }

    //TODO Error Handeling
    pub fn draw(
        &mut self,
        ctx: &Context,
        raytracing_ctx: &RayTracingContext,
        draw_data: &imgui::DrawData,
    ) {
        let frame_in_flight = self.frame_number % FRAMES_IN_FLIGHT;
        let swapchain_image_index = self.begin_frame(ctx, frame_in_flight);

        self.execute_passes(ctx, raytracing_ctx, frame_in_flight, swapchain_image_index);
        self.render_imgui(
            ctx,
            draw_data,
            self.frame_data[frame_in_flight as usize].cmd,
            swapchain_image_index,
        );
        self.submit_frame(ctx, frame_in_flight, swapchain_image_index);
    }

    pub fn toggle_pass(&mut self, pass: &str, value: bool) {
        self.passes
            .iter_mut()
            .find(|e| e.name == pass)
            .unwrap()
            .active = value;
        self.build_sync_resources();
    }
}

#[derive(Default)]
pub struct FrameBuilder<'a> {
    back_buffer: String,
    passes: Vec<PassBuilder<'a>>,
}

impl<'b> FrameBuilder<'b> {
    pub fn add_pass(mut self, pass: PassBuilder<'b>) -> Self {
        self.passes.push(pass);
        self
    }
    pub fn set_back_buffer_source(mut self, back_buffer: &str) -> Self {
        self.back_buffer = back_buffer.to_owned();
        self
    }
}

pub struct PassBuilder<'a> {
    input_resources: Vec<String>,
    temporal_resources: Vec<String>,
    output_resources: Vec<String>,
    bindless_descriptor: bool,
    name: String,
    active: bool,
    ty: PassBuilderType<'a>,
}

enum PassBuilderType<'a> {
    Raytracing(RayTracingPassBuilder<'a>),
    Compute(ComputePassBuilder<'a>),
}

impl<'b> PassBuilder<'b> {
    pub fn read(mut self, value: &'b str) -> Self {
        if self.input_resources.contains(&value.to_owned()) {
            panic!("Resource already in input resources")
        }
        self.input_resources.push(value.to_owned());
        self
    }
    pub fn read_previous(mut self, value: &'b str) -> Self {
        if self.temporal_resources.contains(&value.to_owned()) {
            panic!("Resource already in temporal resources")
        }
        self.temporal_resources.push(value.to_owned());
        self
    }
    pub fn write(mut self, value: &'b str) -> Self {
        if self.output_resources.contains(&value.to_owned()) {
            panic!("Resource already in output resources")
        }
        self.output_resources.push(value.to_owned());
        self
    }
    pub fn bindles_descriptor(mut self, value: bool) -> Self {
        self.bindless_descriptor = value;
        self
    }
    pub fn toggle_active(mut self, value: bool) -> Self {
        self.active = value;
        self
    }

    pub fn new_compute(name: &str, builder: ComputePassBuilder<'b>) -> Self {
        Self {
            ty: PassBuilderType::Compute(builder),
            bindless_descriptor: false,
            input_resources: Vec::new(),
            output_resources: Vec::new(),
            temporal_resources: Vec::new(),
            name: name.to_owned(),
            active: true,
        }
    }
    pub fn new_raytracing(name: &str, builder: RayTracingPassBuilder<'b>) -> Self {
        Self {
            ty: PassBuilderType::Raytracing(builder),
            bindless_descriptor: false,
            input_resources: Vec::new(),
            output_resources: Vec::new(),
            temporal_resources: Vec::new(),
            name: name.to_owned(),
            active: true,
        }
    }
}

#[derive(Clone, Copy)]
enum ComputePassBuilderCommand {
    Dispatch { x: u32, y: u32, z: u32 },
    DispatchIndirect(vk::Buffer),
}

impl Default for ComputePassBuilderCommand {
    fn default() -> Self {
        ComputePassBuilderCommand::Dispatch { x: 0, y: 0, z: 0 }
    }
}

#[derive(Default)]
pub struct ComputePassBuilder<'a> {
    shader_source: &'a str,
    dispatch: ComputePassBuilderCommand,
}

impl<'b> ComputePassBuilder<'b> {
    pub fn shader(mut self, src: &'b str) -> Self {
        self.shader_source = src;
        self
    }
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        self.dispatch = ComputePassBuilderCommand::Dispatch { x, y, z };
        self
    }
    pub fn dispatch_indirect(mut self, buffer: Buffer) -> Self {
        self.dispatch = ComputePassBuilderCommand::DispatchIndirect(buffer.buffer);
        self
    }
}

#[derive(Default)]
pub struct RayTracingPassBuilder<'a> {
    launch_size: [u32; 2],
    ray_gen_source: &'a str,
    override_hit_shader_source: Option<&'a str>,
    override_miss_shader_source: Option<&'a str>,
}

impl<'b> RayTracingPassBuilder<'b> {
    pub fn raygen_shader(mut self, src: &'b str) -> Self {
        self.ray_gen_source = src;
        self
    }
    pub fn raymiss_shader(mut self, src: &'b str) -> Self {
        self.override_miss_shader_source = Some(src);
        self
    }
    pub fn rayhit_shader(mut self, src: &'b str) -> Self {
        self.override_hit_shader_source = Some(src);
        self
    }
    pub fn launch(mut self, x: u32, y: u32) -> Self {
        self.launch_size = [x, y];
        self
    }
    pub fn launch_fullscreen(mut self) -> Self {
        self.launch_size = [WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32];
        self
    }
}
