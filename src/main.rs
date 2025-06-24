#![feature(let_chains)]
#![feature(generic_const_exprs)]
#![feature(substr_range)]
#![feature(box_as_ptr)]
#![feature(int_roundings)]
#![feature(rustc_private)]
#![feature(map_try_insert)]
#![feature(f16)]
pub mod assets;
pub mod components;
pub mod imgui;
pub mod renderer;
use bevy_a11y::AccessibilityPlugin;
use bevy_app::prelude::*;
use bevy_asset::{AssetApp, AssetPlugin, AssetServer, Assets, Handle};
use bevy_ecs::{
    component::Component,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Local, Query, Res, ResMut},
};
use std::time::{Duration, Instant};

use ::imgui::Condition;
use ash::vk::{self, BufferUsageFlags, Format, ImageUsageFlags};
// use assets::{gltf, model::Model};
use bevy_input::InputPlugin;
use bevy_log::LogPlugin;
use bevy_time::TimePlugin;
use bevy_window::{ExitCondition, Window, WindowPlugin, WindowResolution};
use bevy_winit::WinitPlugin;
use components::camera::{Camera, CameraPlugin, Controls};
use glam::{vec3, IVec2, Mat4, Vec2, Vec3};
use gpu_allocator::MemoryLocation;
use imgui::ImGui;
use renderer::{
    bindless::BindlessDescriptorHeap,
    pipeline_cache::PipelineCache,
    render_graph::{RenderGraph, IMPORTED},
    vulkan::{
        self,
        raytracing::{self, RayTracingContext},
        swapchain::Swapchain,
        Context,
    },
    world::{Instance, Transform},
    RenderPlugin,
};
use simple_logger::SimpleLogger;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

use crate::assets::{GltfMesh, GltfMeshLoader, Mesh, MeshAssets};

const WINDOW_SIZE: IVec2 = IVec2::new(1920, 1088);

fn init(mut cmd: Commands, asset_server: Res<AssetServer>) {
    let controles = Controls {
        ..Default::default()
    };

    let camera = Camera::new(
        vec3(0.0, 0.0, -1.0),
        vec3(0.0, 0.0, 1.0),
        65.0_f32.to_radians(),
        WINDOW_SIZE.x as f32 / WINDOW_SIZE.y as f32,
        0.1,
        1000.0,
    );
    let model: Handle<Mesh> = asset_server.load("box.glb");
    cmd.insert_resource(controles);
    cmd.spawn(camera);
    // cmd.spawn((
    //     Instance { model },
    //     Transform::from_position(Vec3::new(0.0, 0.0, 0.0)),
    // ));
}

fn update(mut cmd: Commands, query: Query<&Camera>) {
    let camera = query.single().unwrap();
    // log::info!("{:#?}", camera);
}

fn main() {
    // let model_thread = std::thread::spawn(|| gltf::load_file("./assets/box.glb").unwrap());
    // let image_thread = std::thread::spawn(|| image::open("./assets/skybox2.exr").unwrap());

    App::new()
        .add_plugins((
            LogPlugin {
                filter: "".to_owned(),
                level: bevy_log::Level::DEBUG,
                ..Default::default()
            },
            AccessibilityPlugin,
            InputPlugin,
            WindowPlugin {
                close_when_requested: true,
                exit_condition: ExitCondition::OnPrimaryClosed,
                primary_window: Some(Window {
                    resolution: WindowResolution::new(WINDOW_SIZE.x as f32, WINDOW_SIZE.y as f32),
                    present_mode: bevy_window::PresentMode::AutoNoVsync,
                    title: "RayTracer".to_owned(),
                    ..Default::default()
                }),
            },
            AssetPlugin {
                mode: bevy_asset::AssetMode::Processed,
                ..Default::default()
            },
            WinitPlugin::<bevy_winit::WakeUp>::default(),
            TimePlugin,
            RenderPlugin,
            CameraPlugin,
            // ModelPlugin,
            TaskPoolPlugin::default(),
            // ScenePlugin,
            // UiPlugin::default(),
            MeshAssets,
        ))
        .add_systems(Startup, init.after(renderer::init))
        .add_systems(Update, update)
        .run();
}
