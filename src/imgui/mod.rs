use ash::vk::{self, ImageLayout, Offset2D};
use bevy_ecs::resource::Resource;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::{renderer::vulkan::Context, WINDOW_SIZE};

pub struct ImGui {
    pub context: imgui::Context,
    pub platform: imgui_winit_support::WinitPlatform,
}

impl ImGui {
    pub fn new(window: &winit::window::Window) -> Self {
        let mut imgui = imgui::Context::create();
        imgui.style_mut().anti_aliased_lines = true;
        imgui.style_mut().anti_aliased_fill = true;

        imgui.fonts().build_alpha8_texture();

        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);
        Self {
            context: imgui,
            platform,
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
}
