#![feature(let_chains)]
#![feature(generic_const_exprs)]
pub mod assets;
pub mod backend;
pub mod imgui;
pub mod scene;
use std::time::{Duration, Instant};

use ::imgui::Condition;
use ash::vk;
use assets::{gltf, model::Model};
use backend::{
    raytracing::{self, RayTracingContext},
    render_graph::{build::{ComputePassBuilder, FrameBuilder, PassBuilder, RasterizationPassBuilder, RayTracingPassBuilder}, ImageSize, RenderGraph, ResourceType, SceneResources},
    utils::{create_sampler, Buffer, Image, ImageResource, SamplerInfo},
    vulkan_context::{self, Context},
};
use glam::{vec3, IVec2, Mat4};
use gpu_allocator::MemoryLocation;
use imgui::ImGui;
use scene::camera::{Camera, Controls};
use simple_logger::SimpleLogger;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

const WINDOW_SIZE: IVec2 = IVec2::new(1920, 1088);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanarViewConstants {
    pub mat_world_to_view: glam::Mat4,
    pub mat_view_to_clip: glam::Mat4,
    pub mat_world_to_clip: glam::Mat4,
    pub mat_clip_to_view: glam::Mat4,
    pub mat_view_to_world: glam::Mat4,
    pub mat_clip_to_world: glam::Mat4,

    pub viewport_origin: glam::Vec2,
    pub viewport_size: glam::Vec2,

    pub viewport_size_inv: glam::Vec2,
    pub pixel_offset: glam::Vec2,

    pub clip_to_window_scale: glam::Vec2,
    pub clip_to_window_bias: glam::Vec2,

    pub window_to_clip_scale: glam::Vec2,
    pub window_to_clip_bias: glam::Vec2,

    pub camera_direction_or_position: glam::Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GConst {
    pub planar_view_constants: PlanarViewConstants,
    pub frame: u32,
    pub blendfactor: f32,
    pub bounces: u32,
    pub samples: u32,
    pub proberng: u32,
    pub cell_size: f32,
    pub mouse: [u32; 2],
}

fn main() {
    let model_thread = std::thread::spawn(|| gltf::load_file("./assets/box.glb").unwrap());
    let image_thread = std::thread::spawn(|| image::open("./assets/skybox2.exr").unwrap());

    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .with_colors(true)
        .init()
        .unwrap();

    let event_loop = EventLoop::new().unwrap();
    #[allow(deprecated)]
    let window = event_loop
        .create_window(WindowAttributes::default().with_inner_size(PhysicalSize {
            width: WINDOW_SIZE.x,
            height: WINDOW_SIZE.y,
        }))
        .unwrap();

    Context::init(&window, &window);
    RayTracingContext::init();

    let model = model_thread.join().unwrap();
    let model = Model::from_gltf(model).unwrap();

    let image = image_thread.join().unwrap();
    let skybox =
        ImageResource::new_from_data(image, vk::Format::R32G32B32A32_SFLOAT).unwrap();

    let sampler_info = SamplerInfo {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        max_anisotropy: 1.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
        compare_op: vk::CompareOp::NEVER,
        ..Default::default()
    };

    let model_mat = Mat4::IDENTITY;
    let tlas = {
        let instaces = &[model.instance(model_mat)];

        let instance_buffer = Buffer::from_data(
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            instaces,
        )
        .unwrap();

        let as_struct_geo = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer.address,
                    }),
            });

        let as_ranges = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(instaces.len() as _)
            .primitive_offset(0)
            .transform_offset(0);

        RayTracingContext::get()
            .create_acceleration_structure(
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                &[as_struct_geo],
                &[as_ranges],
                &[1],
            )
            .unwrap()
    };

    let uniform_buffer = Buffer::new_aligned(
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        MemoryLocation::CpuToGpu,
        size_of::<GConst>() as u64,
        Some(16),
    )
    .unwrap();

    let scene_resources = SceneResources {
        geometry_infos: model.geometry_info_buffer,
        index_buffer: model.index_buffer,
        samplers: model.samplers,
        skybox: Some(skybox),
        skybox_sampler: Some(create_sampler(&sampler_info).unwrap()),
        texture_images: model.images,
        texture_samplers: model.textures,
        tlas,
        uniform_buffer,
        vertex_buffer: model.vertex_buffer,
    };

    let mut imgui = ImGui::new(&window);

    let mut render_graph = RenderGraph::new(scene_resources, &mut imgui);
    let swapchain_format = render_graph.swpachain_format();

    render_graph.add_image_resource(
        "GBuffer",
        vk::Format::R32G32B32A32_UINT,
        ImageSize::Viewport,
    ).add_image_resource(
        "GBufferDepth",
        vk::Format::R32_SFLOAT,
        ImageSize::Viewport,
    ).add_image_resource(
        "Out",
        swapchain_format,
        ImageSize::Viewport,
    ).add_temporal_image_resource(
        "ProbeAtlas",
        vk::Format::R32G32B32A32_SFLOAT,
        ImageSize::ViewportFraction { x: 0.5, y: 0.5 },
    ).add_buffer_resource(
        "SphericalHarmonics",
        (WINDOW_SIZE.y as u64 * WINDOW_SIZE.x as u64 * (9 * 4 * 3)).div_ceil(16),
    ).add_temporal_image_resource(
        "Light",
        vk::Format::R32G32B32A32_SFLOAT,
        ImageSize::Viewport,
    ).add_image_resource(
        "TraceDirections",
        vk::Format::R16_UINT,
        ImageSize::ViewportFraction { x: 0.5, y: 0.5 },
    ).add_image_resource(
        "DebugPDFs",
        vk::Format::R32_SFLOAT,
        ImageSize::ViewportFraction { x: 0.5, y: 0.5 },
    );

    let frame = FrameBuilder::default()
        .add_pass(
            PassBuilder::new_raytracing(
                "GBuffer",
                RayTracingPassBuilder::default()
                    .raygen_shader("gbuffer.slang.spv")
                    .launch_fullscreen(),
            )
            .bindles_descriptor(true)
            .write("GBuffer")
            .write("GBufferDepth"),
        )
        .add_pass(
            PassBuilder::new_compute(
                "StructuredImportanceSampling",
                ComputePassBuilder::default()
                    .shader("structured_importance_sampling.slang.spv")
                    .dispatch(
                        (WINDOW_SIZE.x as u32).div_ceil(16),
                        (WINDOW_SIZE.y as u32).div_ceil(16),
                        1,
                    ),
            )
            .bindles_descriptor(true)
            .read_previous("ProbeAtlas")
            .read("GBuffer")
            .read("GBufferDepth")
            .write("TraceDirections")
            .write("DebugPDFs"),
        )
        .add_pass(
            PassBuilder::new_raytracing(
                "TraceProbes",
                RayTracingPassBuilder::default()
                    .raygen_shader("trace_probes.slang.spv")
                    .launch(WINDOW_SIZE.x as u32 / 2, WINDOW_SIZE.y as u32 / 2),
            )
            .bindles_descriptor(true)
            .write("ProbeAtlas")
            .read_previous("ProbeAtlas")
            .read("GBuffer")
            .read("GBufferDepth")
            .read("TraceDirections"),
        )
        .add_pass(
            PassBuilder::new_compute(
                "SphericalHarmonics",
                ComputePassBuilder::default()
                    .shader("spherical_harmonic_conversion.slang.spv")
                    .dispatch(
                        (WINDOW_SIZE.x as u32).div_ceil(16),
                        (WINDOW_SIZE.y as u32).div_ceil(16),
                        1,
                    ),
            )
            .write("SphericalHarmonics")
            .read("ProbeAtlas"),
        )
        .add_pass(
            PassBuilder::new_compute(
                "InterpolateProbes",
                ComputePassBuilder::default()
                    .shader("interpolate_probes.slang.spv")
                    .dispatch(WINDOW_SIZE.x as u32 / 8, WINDOW_SIZE.y as u32 / 8, 1),
            )
            .bindles_descriptor(true)
            .read("GBuffer")
            .read("GBufferDepth")
            .read("SphericalHarmonics")
            .write("Light"),
        )
        .add_pass(
            PassBuilder::new_raytracing(
                "RefrenceMode",
                RayTracingPassBuilder::default()
                    .raygen_shader("refrence_mode.slang.spv")
                    .launch_fullscreen(),
            )
            .bindles_descriptor(true)
            .read("GBuffer")
            .read("GBufferDepth")
            .read_previous("Light")
            .write("Light")
            .toggle_active(false),
        )
        .add_pass(
            PassBuilder::new_compute(
                "Postprocess",
                ComputePassBuilder::default()
                    .shader("postprocess.slang.spv")
                    .dispatch(WINDOW_SIZE.x as u32 / 8, WINDOW_SIZE.y as u32 / 8, 1),
            )
            .bindles_descriptor(true)
            .read("GBufferDepth")
            .read("Light")
            .write("Out"),
        )
        .add_pass(PassBuilder::new_raster("Gizzmos", RasterizationPassBuilder::default()
            .attachment("Out")
            .depth_attachment("GBufferDepth")
            .mesh_shader("gizzmos.slang.spv")
            .fragment_shader("gizzmos.slang.spv"))
        )
        .set_back_buffer_source("Out");

    render_graph.compile(frame);

    let mut controles = Controls {
        ..Default::default()
    };

    let mut camera = Camera::new(
        vec3(0.0, 0.0, 10.0),
        vec3(0.0, 0.0, 1.0),
        65.0,
        WINDOW_SIZE.x as f32 / WINDOW_SIZE.y as f32,
        0.1,
        1000.0,
    );

    let mut gconst = GConst::default();
    gconst.blendfactor = 1.0;
    gconst.samples = 1;
    gconst.bounces = 4;
    gconst.cell_size = f32::tan(
        camera.fov
            * 16.0
            * f32::max(
                1.0 / WINDOW_SIZE.y as f32,
                WINDOW_SIZE.y as f32 / (WINDOW_SIZE.x as f32 * WINDOW_SIZE.x as f32),
            ),
    );
    let mut frame_time = Duration::default();
    let mut last_time = Instant::now();
    let mut refrence_mode = false;

    #[allow(clippy::collapsible_match, clippy::single_match, deprecated)]
    event_loop
        .run(move |event, control_flow| {
            control_flow.set_control_flow(ControlFlow::Poll);
            controles = controles.handle_event(&event, &window);
            imgui.handel_events(&window, &event);
            match event {
                Event::NewEvents(_) => {
                    controles = controles.reset();
                    let now = Instant::now();
                    frame_time = now - last_time;
                    last_time = Instant::now();
                    imgui.update_delta_time(frame_time);
                }
                Event::AboutToWait => {
                    imgui.prepare_frame(&window);
                    window.request_redraw();
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => control_flow.exit(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        match (event.physical_key, event.state) {
                            (PhysicalKey::Code(KeyCode::Escape), ElementState::Released) => {
                                control_flow.exit()
                            }
                            _ => (),
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let new_cam = camera.update(&controles, frame_time);
                        camera = new_cam;
                        gconst.planar_view_constants = camera.planar_view_constants();
                        gconst.frame = render_graph.frame_number.try_into().unwrap_or(0);
                        gconst.mouse = [
                            controles.cursor_position[0] as u32,
                            controles.cursor_position[1] as u32,
                        ];
                        render_graph.update_uniform(&gconst);

                        let ui = imgui.context.frame();
                        ui.window("Constants Editor")
                            .size([300.0, WINDOW_SIZE.y as f32], Condition::FirstUseEver)
                            .position([0.0, 0.0], Condition::FirstUseEver)
                            .build(|| {
                                ui.text(format!(
                                    "FPS: {:.0}, {:?}",
                                    1000.0 / frame_time.as_millis() as f64,
                                    frame_time
                                ));
                                ui.slider("Blendfactor", 0.0, 1.0, &mut gconst.blendfactor);
                                ui.input_scalar("Samples", &mut gconst.samples)
                                    .step(1)
                                    .build();
                                ui.input_scalar("Bounces", &mut gconst.bounces)
                                    .step(1)
                                    .build();
                                ui.checkbox_flags("ProbeRng", &mut gconst.proberng, 0x1);

                                if let Some(token) =
                                    ui.begin_combo("Backbuffer", render_graph.back_buffer.as_str())
                                {
                                    for (i, res) in &render_graph.resources {
                                        if res.ty == ResourceType::Image
                                            && ui.selectable(i.as_str())
                                        {
                                            render_graph.back_buffer = i.clone();
                                        }
                                    }
                                    token.end();
                                }

                                if ui.checkbox("Refrence Mode", &mut refrence_mode) {
                                    render_graph.toggle_pass("RefrenceMode", refrence_mode);
                                    render_graph.toggle_pass("InterpolateProbes", !refrence_mode);
                                    render_graph.toggle_pass("SphericalHarmonics", !refrence_mode);
                                    render_graph.toggle_pass("TraceProbes", !refrence_mode);
                                    render_graph.toggle_pass(
                                        "StructuredImportanceSampling",
                                        !refrence_mode,
                                    );
                                }
                            });
                        imgui.platform.prepare_render(ui, &window);
                        let draw_data = imgui.context.render();

                        render_graph.draw(draw_data);
                    }
                    _ => (),
                },
                Event::LoopExiting => {}
                _ => (),
            }
        })
        .unwrap();
}
