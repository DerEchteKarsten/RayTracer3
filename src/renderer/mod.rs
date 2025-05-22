use std::ops::Deref;

use ash::vk::{Format, ImageUsageFlags};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_winit::WinitWindows;
use bindless::BindlessDescriptorHeap;
use glam::Vec2;
use render_graph::{
    begin_frame, build::{DispatchSize, ImageSize}, draw_frame, executions::{ComputePass, RasterPass, WorkSize2D}, resources::ResourceHandle, RenderGraph, IMPORTED
};
use vulkan::Context;
use world::{apply_transform, extract_instances, removed_instances, WorldResources};

use crate::{
    components::camera::Camera, raytracing::RayTracingContext, PipelineCache, WINDOW_SIZE,
};

pub(crate) mod bindless;
pub(crate) mod pipeline_cache;
pub(crate) mod render_graph;
pub(crate) mod vulkan;
pub(crate) mod world;

//TODO Error Handeling

pub fn init(world: &mut World) {
    let windows = world.get_non_send_resource::<WinitWindows>().unwrap();
    let window = windows.windows.values().into_iter().last().unwrap().deref();
    let mut ctx = Context::new(&window, &window).unwrap();
    let raytracing = RayTracingContext::new(&ctx);
    let mut bindless = BindlessDescriptorHeap::new(&ctx, Some(&raytracing));
    let rg = RenderGraph::new(&mut ctx, &mut bindless);
    world.insert_resource(GConst::default());
    world.insert_resource(ctx);
    world.insert_resource(bindless);
    world.insert_resource(raytracing);
    world.insert_resource(rg);
    world.insert_resource(PipelineCache::new());
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Resource)]
struct GConst {
    pub proj: glam::Mat4,
    pub view: glam::Mat4,
    pub proj_inverse: glam::Mat4,
    pub view_inverse: glam::Mat4,
    pub window_size: glam::Vec2,
    pub frame: u32,
    pub blendfactor: f32,
    pub bounces: u32,
    pub samples: u32,
    pub proberng: u32,
    pub cell_size: f32,
    pub mouse: [u32; 2],
    pub pad: [u32; 2],
}



fn commands(mut rg: ResMut<RenderGraph>, world: Res<WorldResources>, mut gconst: ResMut<GConst>, query: Query<&Camera>) {
    let camera = query.single().unwrap();
    gconst.proj = camera.projection_matrix();
    gconst.proj_inverse = gconst.proj.inverse();
    gconst.view = camera.view_matrix();
    gconst.view_inverse = gconst.view.inverse();
    gconst.window_size = Vec2::new(WINDOW_SIZE.x as f32, WINDOW_SIZE.y as f32);
    gconst.frame = rg.frame_number as u32;

    let depth = rg.image(ImageSize::FullScreen, Format::D32_SFLOAT, "depth");
    let color = rg.image(ImageSize::FullScreen, Format::R32G32B32A32_SFLOAT, "color");
    let swapchain = rg.get_swapchain();

    let test2 = RasterPass::new(&mut rg, "test2")
        .fragment_shader("bindless_test2")
        .mesh_shader("bindless_test2")
        .fragment_entry("fragment")
        .mesh_entry("mesh")
        .constants(gconst.as_ref())
        .read(IMPORTED, world.instances)
        .depth_attachment(IMPORTED, depth)
        .color_attachment(IMPORTED, color, Some([0.1, 0.15, 0.3, 1.0]))
        .render_area(WorkSize2D::FullScreen)
        .draw(DispatchSize::X(1));

    ComputePass::new(&mut rg, "test")
        .shader("bindless_test")
        .read(test2, depth)
        .read(test2, color)
        .write(IMPORTED, swapchain)
        .dispatch(DispatchSize::FullScreen);
}

pub fn RenderPlugin(app: &mut App) {
    app
    .add_systems(PreStartup, init)
    .add_systems(Update, (apply_transform, extract_instances, removed_instances))
    .add_systems(
        PostUpdate,
        (begin_frame, (commands, draw_frame).chain()).chain(),
    );
}
