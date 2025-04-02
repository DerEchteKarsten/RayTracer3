use std::collections::HashMap;

use anyhow::Result;
use ash::vk::{self, ImageLayout};
use glam::UVec2;

use crate::{imgui::ImGui, WINDOW_SIZE};

use super::{
    raytracing::{AccelerationStructure, ShaderBindingTable},
    swapchain::Swapchain,
    utils::{Buffer, ImageResource},
    vulkan_context::Context,
};

pub(crate) mod build;
pub(crate) mod compile;
pub(crate) mod run;

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

impl ResourceData {
    fn get_image(&self) -> Option<&ImageResource> {
        match self {
            Self::Image(image) => Some(image),
            _ => None,
        }
    }
    fn get_buffer(&self) -> Option<&Buffer> {
        match self {
            Self::Buffer(buffer) => Some(buffer),
            _ => None,
        }
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
    fn get_format(&self) -> Option<vk::Format> {
        match &self.handle {
            ResourceTemporal::Single(resource) => match resource {
                ResourceData::Image(image) => Some(image.image.format),
                _ => None,
            },
            ResourceTemporal::Temporal(images) => {
                let f1 = match &images[0] {
                    ResourceData::Image(image) => Some(image.image.format),
                    _ => None,
                };
                let f2 = match &images[1] {
                    ResourceData::Image(image) => Some(image.image.format),
                    _ => None,
                };

                if let Some(f1) = f1
                    && let Some(f2) = f2
                {
                    if f1 == f2 {
                        Some(f1)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
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
    pub(crate) skybox: Option<ImageResource>,
    pub(crate) skybox_sampler: Option<vk::Sampler>,
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
    fn new(scene: SceneResources, imgui: &mut ImGui) -> Result<Self> {
        let ctx = Context::get();

        let swapchain = Swapchain::new()?;

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

pub struct ColorAttachment {
    pub(super) format: vk::Format,
    pub(super) clear: Option<[f32; 4]>,
    pub(super) resource: String,
}

struct DepthStencilAttachment {
    pub(super) format: vk::Format,
    pub(super) clear: Option<f32>,
    pub(super) resource: String,
}

impl DepthStencilAttachment {
    pub fn render_info(&self, graph: &RenderGraph) -> vk::RenderingAttachmentInfo {
        vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: self.clear.unwrap_or(1.0),
                    stencil: 0,
                },
            })
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image_view(graph.resources[&self.resource].get_current(&graph).get_image().unwrap().view)
            .load_op(if self.clear.is_some() {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .resolve_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::NONE)
    }
}


enum RenderPassCommand{
    Raytracing {
        shader_binding_table: ShaderBindingTable,
        x: u32,
        y: u32,
    },
    Raster{
        color_attachments: Vec<ColorAttachment>,
        depth_attachment: Option<DepthStencilAttachment>,
        stencil_attachment: Option<DepthStencilAttachment>,
        render_area: UVec2,
        x: u32,
        y: u32,
        z: u32,
    },
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
    Rasterization,
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

impl ImageSize {
    fn raw(self) -> (u32, u32) {
        match self {
            ImageSize::Custom { x, y } => (x, y),
            ImageSize::Viewport => (WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32),
            ImageSize::ViewportFraction { x, y } => (
                (WINDOW_SIZE.x as f32 * x).ceil() as u32,
                (WINDOW_SIZE.y as f32 * y).ceil() as u32,
            ),
        }
    }
}