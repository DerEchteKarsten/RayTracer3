use ash::vk;
use glam::{UVec2, Vec4};

use crate::WINDOW_SIZE;

use crate::backend::utils::Buffer;

#[derive(Default)]
pub struct FrameBuilder<'a> {
    pub(super) back_buffer: String,
    pub(super) passes: Vec<PassBuilder<'a>>,
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
    pub(super) input_resources: Vec<String>,
    pub(super) temporal_resources: Vec<String>,
    pub(super) output_resources: Vec<String>,
    pub(super) bindless_descriptor: bool,
    pub(super) name: String,
    pub(super) active: bool,
    pub(super) ty: PassBuilderType<'a>,
}

pub(super) enum PassBuilderType<'a> {
    Raytracing(RayTracingPassBuilder<'a>),
    Compute(ComputePassBuilder<'a>),
    Rasterization(RasterizationPassBuilder<'a>),
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
    pub fn new_raster(name: &str, builder: RasterizationPassBuilder<'b>) -> Self {
        Self {
            ty: PassBuilderType::Rasterization(builder),
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
pub(super) enum ComputePassBuilderCommand {
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
    pub(super) shader_source: &'a str,
    pub(super) dispatch: ComputePassBuilderCommand,
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
    pub(super) launch_size: [u32; 2],
    pub(super) ray_gen_source: &'a str,
    pub(super) override_hit_shader_source: Option<&'a str>,
    pub(super) override_miss_shader_source: Option<&'a str>,
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

#[derive(Default)]
struct RasterShaderStages<'a> {
    pub(super) vertex: Option<&'a str>,
    pub(super) hullshader: Option<&'a str>,
    pub(super) domainshader: Option<&'a str>,
    pub(super) geometryshader: Option<&'a str>,
    pub(super) fragment: Option<&'a str>,
}

#[derive(Default)]
pub struct RasterizationPassBuilder<'a> {
    pub(super) mesh_shader: &'a str,
    pub(super) fragment_shader: &'a str,
    pub(super) fragment_main: &'static str,
    pub(super) mesh_main: &'static str,
    pub(super) attachments: Vec<(String, Option<[f32; 4]>)>,
    pub(super) depth_attachment: Option<(String, Option<f32>)>,
    pub(super) stencil_attachment: Option<(String, Option<f32>)>,
    pub(super) render_area: Option<UVec2>,
}

impl<'b> RasterizationPassBuilder<'b> {
    pub fn mesh_shader(mut self, src: &'b str) -> Self {
        self.mesh_shader = src;
        self
    }
    pub fn fragment_main(mut self, main: &'static str) -> Self {
        self.fragment_main = main;
        self
    }
    pub fn mesh_main(mut self, main: &'static str) -> Self {
        self.mesh_main = main;
        self
    }
    pub fn fragment_shader(mut self, src: &'b str) -> Self {
        self.fragment_shader = src;
        self
    }
    pub fn clear_attachment(mut self, attachment: &'b str, clear: Vec4) -> Self {
        self.attachments.push((attachment.to_owned(), Some(clear.into())));
        self
    }
    pub fn attachment(mut self, attachment: &'b str) -> Self {
        self.attachments.push((attachment.to_owned(), None));
        self
    }
    pub fn depth_attachment<'a>(mut self, v: &'a str) -> Self {
        self.depth_attachment = Some((v.to_owned(), None));
        self
    }
    pub fn stencil_attachment<'a>(mut self, v: &'a str) -> Self {
        self.stencil_attachment = Some((v.to_owned(), None));
        self
    }
    pub fn clear_depth_attachment<'a>(mut self, v: &'a str, clear: f32) -> Self {
        self.depth_attachment = Some((v.to_owned(), Some(clear)));
        self
    }
    pub fn clear_stencil_attachment<'a>(mut self, v: &'a str, clear: f32) -> Self {
        self.stencil_attachment = Some((v.to_owned(), Some(clear)));
        self
    }
    pub fn render_area<'a>(mut self, v: [u32; 2]) -> Self {
        self.render_area = Some(v.into());
        self
    }
}
