use ash::vk::{self, ImageLayout, Offset2D};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::{
    backend::{render_graph::RenderGraph, vulkan_context::Context},
    WINDOW_SIZE,
};

pub struct ImGui {
    renderer: imgui_rs_vulkan_renderer::Renderer,
    context: imgui::Context,
    platform: imgui_winit_support::WinitPlatform,
    render_pass: vk::RenderPass,
    frame_buffers: Vec<vk::Framebuffer>,
}

impl ImGui {
    pub fn new(ctx: &Context, rendergraph: &RenderGraph, window: &winit::window::Window) -> Self {
        let mut imgui = imgui::Context::create();
        imgui.style_mut().anti_aliased_lines = true;
        imgui.style_mut().anti_aliased_fill = true;

        let attachments = [vk::AttachmentDescription::default()
            .initial_layout(ImageLayout::PRESENT_SRC_KHR)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .format(rendergraph.swpachain_format())
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

        let imgui_pass = unsafe {
            ctx.device
                .create_render_pass(&render_pass_create_info, None)
                .unwrap()
        };

        let mut imgui_frame_buffers = vec![];
        for image in rendergraph.static_resources.swapchain.images.iter() {
            let attachment = [image.view];
            let frame_buffer_info = vk::FramebufferCreateInfo::default()
                .attachments(&attachment)
                .height(WINDOW_SIZE.y as u32)
                .width(WINDOW_SIZE.x as u32)
                .layers(1)
                .render_pass(imgui_pass);
            imgui_frame_buffers.push(unsafe {
                ctx.device
                    .create_framebuffer(&frame_buffer_info, None)
                    .unwrap()
            })
        }

        let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
            &ctx.instance,
            ctx.physical_device.handel,
            ctx.device.clone(),
            ctx.graphics_queue,
            ctx.command_pool,
            imgui_pass,
            &mut imgui,
            None,
        )
        .unwrap();

        imgui.fonts().build_alpha8_texture();

        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);
        Self {
            context: imgui,
            platform,
            renderer: imgui_renderer,
            frame_buffers: imgui_frame_buffers,
            render_pass: imgui_pass,
        }
    }

    pub fn handel_events(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<()>,
    ) {
        self.platform
            .handle_event(self.context.io_mut(), window, event);
    }

    pub fn update_delta_time(&mut self, delta_time: std::time::Duration) {
        self.context.io_mut().update_delta_time(delta_time);
    }

    pub fn prepare_frame(&mut self, window: &winit::window::Window) {
        self.platform
            .prepare_frame(self.context.io_mut(), &window)
            .expect("Failed to prepare frame");
    }

    pub fn render<F>(
        &mut self,
        window: &winit::window::Window,
        ctx: &Context,
        cmd: &vk::CommandBuffer,
        image_index: u32,
        ui_builder: F,
    ) where
        F: FnOnce(&imgui::Ui),
    {
        let ui = self.context.frame();
        ui_builder(ui);
        self.platform.prepare_render(ui, window);
        let draw_data = self.context.render();
        unsafe {
            let begin_info = vk::RenderPassBeginInfo::default()
                .framebuffer(self.frame_buffers[image_index as usize])
                .render_pass(self.render_pass)
                .render_area(
                    vk::Rect2D::default()
                        .extent(vk::Extent2D {
                            width: WINDOW_SIZE.x as u32,
                            height: WINDOW_SIZE.y as u32,
                        })
                        .offset(Offset2D { x: 0, y: 0 }),
                );
            ctx.device
                .cmd_begin_render_pass(*cmd, &begin_info, vk::SubpassContents::INLINE);
            self.renderer.cmd_draw(*cmd, draw_data).unwrap();
            ctx.device.cmd_end_render_pass(*cmd);
        }
    }
}
