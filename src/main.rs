#![feature(let_chains)]
#![feature(generic_const_exprs)]
#![feature(substr_range)]
#![feature(box_as_ptr)]
#![feature(int_roundings)]
#![feature(rustc_private)]
pub mod assets;
pub mod backend;
pub mod imgui;
pub mod scene;
use std::time::{Duration, Instant};

use ::imgui::Condition;
use ash::vk::{self, BufferUsageFlags, Format, ImageUsageFlags};
use assets::{gltf, model::Model};
use backend::{
    bindless::BindlessDescriptorHeap,
    pipeline_cache::PipelineCache,
    render_graph::{
        build::{ComputePass, DispatchSize, ImageSize, RasterPass, RayTracingPass, WorkSize2D},
        RenderGraph, IMPORTED,
    },
    vulkan::{
        self,
        raytracing::{self, RayTracingContext},
        swapchain::Swapchain,
        Context,
    },
};
use glam::{vec3, IVec2, Mat4, Vec2};
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
struct GConst {
    pub proj: glam::Mat4,
    pub view: glam::Mat4,
    pub proj_inverse: glam::Mat4,
    pub view_inverse: glam::Mat4,
    pub window_size: Vec2,
    pub frame: u32,
    pub blendfactor: f32,
    pub bounces: u32,
    pub samples: u32,
    pub proberng: u32,
    pub cell_size: f32,
    pub mouse: [u32; 2],
}

fn main() {
    // let model_thread = std::thread::spawn(|| gltf::load_file("./assets/box.glb").unwrap());
    // let image_thread = std::thread::spawn(|| image::open("./assets/skybox2.exr").unwrap());

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

    Context::init(&window, &window).unwrap();
    // RayTracingContext::init();
    BindlessDescriptorHeap::init();
    PipelineCache::init();

    // let model = model_thread.join().unwrap();
    // let model = Model::from_gltf(model).unwrap();
    // let image = image_thread.join().unwrap();
    // let skybox = Image::new_from_data(image, vk::Format::R32G32B32A32_SFLOAT).unwrap();

    // let model_mat = Mat4::IDENTITY;
    // let tlas = {
    //     let instaces = &[model.instance(model_mat)];

    //     let instance_buffer = Buffer::from_data(
    //         vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    //             | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    //         instaces,
    //     )
    //     .unwrap();

    //     let as_struct_geo = vk::AccelerationStructureGeometryKHR::default()
    //         .geometry_type(vk::GeometryTypeKHR::INSTANCES)
    //         .flags(vk::GeometryFlagsKHR::OPAQUE)
    //         .geometry(vk::AccelerationStructureGeometryDataKHR {
    //             instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
    //                 .array_of_pointers(false)
    //                 .data(vk::DeviceOrHostAddressConstKHR {
    //                     device_address: instance_buffer.address,
    //                 }),
    //         });

    //     let as_ranges = vk::AccelerationStructureBuildRangeInfoKHR::default()
    //         .first_vertex(0)
    //         .primitive_count(instaces.len() as _)
    //         .primitive_offset(0)
    //         .transform_offset(0);

    //     RayTracingContext::get()
    //         .create_acceleration_structure(
    //             vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    //             &[as_struct_geo],
    //             &[as_ranges],
    //             &[1],
    //         )
    //         .unwrap()
    // };

    // let vertecies = rg.import_buffer(&model.vertex_buffer.handle());
    // let tlas = rg.import_tlas(&tlas);

    // let gbuffer = rg.image(
    //     ImageSize::FullScreen,
    //     vk::ImageUsageFlags::STORAGE,
    //     vk::Format::R32G32B32A32_UINT,
    // );
    // let gbuffer_depth = rg.image(
    //     ImageSize::FullScreen,
    //     vk::ImageUsageFlags::STORAGE,
    //     vk::Format::R32_SFLOAT,
    // );
    // let diffuse_lighting = rg.image(
    //     ImageSize::FullScreen,
    //     vk::ImageUsageFlags::STORAGE,
    //     vk::Format::R32G32B32A32_SFLOAT,
    // );
    // let output = rg.image(
    //     ImageSize::FullScreen,
    //     vk::ImageUsageFlags::STORAGE,
    //     swapchain_format,
    // );

    // let gbuffer_pass = RayTracingPass::new(&mut rg)
    //     .shader("gbuffer")
    //     .read(IMPORTED, vertecies)
    //     .read(IMPORTED, tlas)
    //     .write(gbuffer)
    //     .write(gbuffer_depth)
    //     .launch(LaunchSize::FullScreen);

    // let brdf_rays = RayTracingPass::new(&mut rg)
    //     .shader("refrence_mode")
    //     .read(gbuffer_pass, gbuffer)
    //     .read(gbuffer_pass, gbuffer_depth)
    //     .write(diffuse_lighting)
    //     .launch(LaunchSize::FullScreen);

    // let tonemap = ComputePass::new(&mut rg)
    //     .shader("postprocess")
    //     .read(brdf_rays, diffuse_lighting)
    //     .write(output)
    //     .dispatch(DispatchSize::FullScreen);

    // let test_texture = rg.image(ImageSize::FullScreen, ImageUsageFlags::STORAGE, Format::R32G32B32A32_SFLOAT);
    let mut controles = Controls {
        ..Default::default()
    };

    let mut camera = Camera::new(
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        65.0_f32.to_radians(),
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

    let mut imgui = ImGui::new(&window);
    let mut rg = RenderGraph::new();

    // let tlas = rg.import_tlas(&tlas);
    // let vertecies = rg.import_buffer(&model.vertex_buffer.handle());
    // let indicies = rg.import_buffer(&model.index_buffer.handle());
    let depth = rg.image(
        ImageSize::FullScreen,
        ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
        Format::D32_SFLOAT,
    );
    let color = rg.image(
        ImageSize::FullScreen,
        ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
        Format::R8G8B8A8_SRGB,
    );

    let mut frame_time = Duration::default();
    let mut last_time = Instant::now();
    let mut refrence_mode = false;

    #[allow(clippy::collapsible_match, clippy::single_match, deprecated)]
    event_loop
        .run(|event, control_flow| {
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
                        gconst.proj = camera.projection_matrix();
                        gconst.proj_inverse = camera.projection_matrix().inverse();
                        gconst.view = camera.view_matrix().inverse();
                        gconst.view_inverse = camera.view_matrix().inverse();
                        gconst.window_size = Vec2::new(WINDOW_SIZE.x as f32, WINDOW_SIZE.y as f32);
                        gconst.frame = rg.frame_number.try_into().unwrap_or(0);
                        gconst.mouse = [
                            controles.cursor_position[0] as u32,
                            controles.cursor_position[1] as u32,
                        ];
                        rg.begin_frame();

                        let swapchain = rg.get_swapchain();
                        let test2 = RasterPass::new(&mut rg)
                            .fragment_shader("bindless_test2")
                            .mesh_shader("bindless_test2")
                            .fragment_entry("fragment")
                            .mesh_entry("vertex")
                            .constants(&gconst)
                            .depth_attachment(IMPORTED, depth)
                            .color_attachment(IMPORTED, swapchain)
                            .render_area(WorkSize2D::FullScreen)
                            .draw(DispatchSize::X(1));

                        // let test = ComputePass::new(&mut rg)
                        //     .shader("bindless_test")
                        //     .read(test2, depth)
                        //     .read(test2, color)
                        //     .write(IMPORTED, swapchain)
                        //     .dispatch(DispatchSize::FullScreen);
                        rg.draw_frame(test2);
                        // let ui = imgui.context.frame();
                        // ui.window("Constants Editor")
                        //     .size([300.0, WINDOW_SIZE.y as f32], Condition::FirstUseEver)
                        //     .position([0.0, 0.0], Condition::FirstUseEver)
                        //     .build(|| {
                        //         ui.text(format!(
                        //             "FPS: {:.0}, {:?}",
                        //             1000.0 / frame_time.as_millis() as f64,
                        //             frame_time
                        //         ));
                        //         ui.slider("Blendfactor", 0.0, 1.0, &mut gconst.blendfactor);
                        //         ui.input_scalar("Samples", &mut gconst.samples)
                        //             .step(1)
                        //             .build();
                        //         ui.input_scalar("Bounces", &mut gconst.bounces)
                        //             .step(1)
                        //             .build();
                        //         ui.checkbox_flags("ProbeRng", &mut gconst.proberng, 0x1);

                        //         if ui.checkbox("Refrence Mode", &mut refrence_mode) {}
                        //     });
                        // imgui.platform.prepare_render(ui, &window);
                        // let draw_data = imgui.context.render();
                    }
                    _ => (),
                },
                Event::LoopExiting => {}
                _ => (),
            }
        })
        .unwrap();
}
