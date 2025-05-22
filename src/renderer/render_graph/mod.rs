use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    sync::Arc,
};

use anyhow::Result;
use ash::vk::{self, Format, ImageUsageFlags};
use bevy_ecs::{system::ResMut, world::World};
use build::ImageSize;
use derivative::Derivative;
use enum_dispatch::enum_dispatch;
use glam::{UVec2, Vec3};
use gpu_allocator::MemoryLocation;

use crate::{
    raytracing::{AccelerationStructure, RayTracingContext},
    PipelineCache,
};

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    vulkan::{
        buffer::{Buffer, BufferHandle},
        image::{Image, ImageHandle, ImageType},
        swapchain::{Swapchain, FRAMES_IN_FLIGHT},
        Context,
    },
};
pub mod bake;
pub mod build;
pub mod executions;
pub mod resources;
use bevy_ecs::system::Res;
use executions::*;
use resources::*;

pub const IMPORTED: NodeHandle = !0;

#[derive(Debug)]
struct Barrier {
    resource: ResourceHandle,
    layout: vk::ImageLayout,
    access: vk::AccessFlags2,
    stages: vk::PipelineStageFlags2,
}

impl Barrier {
    fn need_invalidate(&self, event: &resources::Event) -> bool {
        (0..64)
            .map(|i| {
                self.access.contains(
                    event.invalidated_in_stage[((self.stages.as_raw() >> i) & 1) as usize / 2],
                )
            })
            .fold(false, |acc, a| acc || a)
    }
}

impl Barrier {
    fn new(resource: ResourceHandle) -> Self {
        Self {
            resource,
            layout: vk::ImageLayout::UNDEFINED,
            access: vk::AccessFlags2::empty(),
            stages: vk::PipelineStageFlags2::empty(),
        }
    }
}

#[derive(Debug)]
struct Barriers {
    invalidates: Vec<Barrier>,
    flushes: Vec<Barrier>,
}

type NodeHandle = usize;

#[enum_dispatch]
pub(crate) trait ExecutionTrait {
    fn execute(
        &self,
        cmd: &vk::CommandBuffer,
        rg: &RenderGraph,
        ctx: &mut Context,
        raytracing_ctx: Option<&RayTracingContext>,
        cache: &mut PipelineCache,
        edges: &[NodeEdge],
    ) -> Result<()>;
    fn get_stages(&self) -> vk::PipelineStageFlags2;
}

#[enum_dispatch(ExecutionTrait)]
#[derive(PartialEq)]
enum Execution {
    RayTracingPass,
    ComputePass,
    RasterPass,
}

#[derive(PartialEq)]
struct Node {
    name: &'static str,
    execution: Execution,
    constant_offset: Option<u32>,
    edges: Vec<NodeEdge>,
}

impl Node {
    fn parents<'b>(&self) -> Vec<NodeHandle> {
        self.edges
            .iter()
            .filter_map(|r| r.origin)
            .collect::<Vec<_>>()
    }

    fn bindings<'b>(
        &'b self,
    ) -> std::iter::Filter<std::slice::Iter<'b, NodeEdge>, impl FnMut(&&'b NodeEdge) -> bool> {
        self.edges.iter().filter(|e| {
            std::mem::discriminant(&e.edge_type)
                != std::mem::discriminant(&EdgeType::ColorAttachmentOutput { clear_color: None })
                && e.edge_type != EdgeType::DepthAttachment
                && e.edge_type != EdgeType::StencilAttachment
        })
    }

    fn cmd_push_constants(
        &self,
        rg: &RenderGraph,
        ctx: &Context,
        bindless: &BindlessDescriptorHeap,
        frame: &FrameData,
        descriptor_offset: u32,
    ) {
        unsafe {
            let mut constants = [0u8; 16];
            constants[0..4].copy_from_slice(&self.constant_offset.unwrap_or(0).to_ne_bytes());

            constants[4..8].copy_from_slice(
                &if self.bindings().count() == 0 {
                    0
                } else {
                    descriptor_offset
                }
                .to_ne_bytes(),
            );
            constants[8..12].copy_from_slice(&rg.descriptor_buffer.descriptor.0.to_ne_bytes());
            constants[12..16].copy_from_slice(&rg.constants_buffer.descriptor.0.to_ne_bytes());

            ctx.device.cmd_push_constants(
                frame.cmd,
                bindless.layout,
                vk::ShaderStageFlags::ALL,
                0,
                &constants,
            )
        };
    }

    fn get_barriers(&self, rg: &RenderGraph) -> Barriers {
        let mut invalidates: HashMap<ResourceHandle, Barrier> = HashMap::new();
        let mut flushes: HashMap<ResourceHandle, Barrier> = HashMap::new();

        for edge in &self.edges {
            match edge.edge_type {
                EdgeType::ShaderRead => {
                    let barrier = invalidates
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    barrier.stages |= self.execution.get_stages();
                    if let Some(image) = rg.image_handle(edge.resource)
                        && image.usage.contains(vk::ImageUsageFlags::STORAGE)
                    {
                        barrier.access |= vk::AccessFlags2::SHADER_STORAGE_READ;
                        barrier.layout = vk::ImageLayout::GENERAL;
                    } else {
                        barrier.access |= vk::AccessFlags2::SHADER_SAMPLED_READ;
                        barrier.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    }
                }
                EdgeType::ColorAttachmentOutput { clear_color: _ } => {
                    let barrier = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    barrier.access |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
                    barrier.stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
                    barrier.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                }
                EdgeType::DepthAttachment | EdgeType::StencilAttachment => {
                    let src = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    let dst = invalidates
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    dst.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    dst.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    dst.stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;

                    src.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    src.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    dst.stages |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
                }
                EdgeType::ShaderReadWrite => {
                    let flush = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    flush.stages |= self.execution.get_stages();
                    flush.access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
                    flush.layout = vk::ImageLayout::GENERAL;

                    let invalidate = invalidates
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    invalidate.stages |= self.execution.get_stages();
                    if let Some(image) = rg.image_handle(edge.resource)
                        && image.usage.contains(vk::ImageUsageFlags::STORAGE)
                    {
                        invalidate.access |= vk::AccessFlags2::SHADER_STORAGE_READ;
                        invalidate.layout = vk::ImageLayout::GENERAL;
                    } else {
                        invalidate.access |= vk::AccessFlags2::SHADER_SAMPLED_READ;
                        invalidate.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    }
                }
                EdgeType::ShaderWrite => {
                    let flush = flushes
                        .entry(edge.resource)
                        .or_insert(Barrier::new(edge.resource));
                    flush.stages |= self.execution.get_stages();
                    flush.access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
                    flush.layout = vk::ImageLayout::GENERAL;
                }
                EdgeType::TransferDst => {
                    todo!();
                }
                EdgeType::TransferSrc => {
                    todo!();
                }
            }
            if !invalidates.contains_key(&edge.resource)
                && let Some(flush) = flushes.get(&edge.resource)
                && rg.resources[edge.resource].event.layout != flush.layout
            {
                invalidates.insert(
                    edge.resource,
                    Barrier {
                        resource: edge.resource,
                        layout: flush.layout,
                        access: vk::AccessFlags2::NONE,
                        stages: self.execution.get_stages(),
                    },
                );
            }
        }

        Barriers {
            invalidates: invalidates.into_values().collect::<Vec<_>>(),
            flushes: flushes.into_values().collect::<Vec<_>>(),
        }
    }
}

pub fn depends_on(rg: &RenderGraph, other: NodeHandle, s: NodeHandle) -> bool {
    other == s
        || rg.nodes[other]
            .edges
            .iter()
            .find(|e| {
                if let Some(origin) = e.origin
                    && origin == s
                {
                    true
                } else {
                    false
                }
            })
            .is_some()
}

#[derive(Clone)]
struct FrameData {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_number: u64,
}

#[derive(bevy_ecs::resource::Resource)]
pub struct RenderGraph {
    pub resources: Vec<Resource>,
    pub resource_cache: Vec<ResourceDescription>,

    constants_buffer: Resource,  //TODO
    descriptor_buffer: Resource, //TODO

    nodes: Vec<Node>,
    constants: HashMap<usize, u32>,
    constants_offset: u32,

    frame_data: [FrameData; FRAMES_IN_FLIGHT],
    frame_timeline_semaphore: vk::Semaphore,
    swapchain: Swapchain,
    swapchain_images: Vec<ResourceHandle>,
    swapchain_image_index: usize,
    pub frame_number: u64,
}

#[derive(Clone, PartialEq)]
enum EdgeType {
    ShaderRead,
    ShaderReadWrite,
    ShaderWrite,
    ColorAttachmentOutput { clear_color: Option<[f32; 4]> },
    DepthAttachment,
    StencilAttachment,
    TransferSrc,
    TransferDst,
}

#[derive(Clone, PartialEq)]
pub struct NodeEdge {
    edge_type: EdgeType,
    origin: Option<NodeHandle>,
    resource: ResourceHandle,
}

impl RenderGraph {
    pub fn new(ctx: &mut Context, bindless: &mut BindlessDescriptorHeap) -> Self {
        let mut frame_data: [FrameData; FRAMES_IN_FLIGHT] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        let mut timeline_create_info = vk::SemaphoreTypeCreateInfo::default()
            .initial_value(FRAMES_IN_FLIGHT as u64 - 1)
            .semaphore_type(vk::SemaphoreType::TIMELINE);

        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_create_info);
        let frame_timeline_semaphore =
            unsafe { ctx.device.create_semaphore(&create_info, None).unwrap() };

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

            frame_data[i] = FrameData {
                cmd: command_buffers[0],
                frame_number: i as u64,
                command_pool,
            }
        }
        let mut resources = Vec::new();
        let swapchain = Swapchain::new(&ctx).unwrap();
        let swapchain_images = swapchain
            .images
            .iter()
            .map(|image| {
                let descriptor = bindless.allocate_image_handle(ctx, image);
                let index = resources.len();
                resources.push(Resource::new(
                    descriptor,
                    ResourceType::Image(image.clone()),
                ));
                index
            })
            .collect::<Vec<_>>();
        let descriptor_buffer = {
            let buffer = Buffer::new(
                ctx,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                size_of::<u32>() as u64 * 256,
            )
            .unwrap();
            let descriptor = bindless.allocate_buffer_handle(ctx, &buffer);
            Resource::new(descriptor, ResourceType::Buffer(buffer))
        };

        let constants_buffer = {
            let buffer = Buffer::new(
                ctx,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                size_of::<u32>() as u64 * 1024,
            )
            .unwrap();
            let descriptor = bindless.allocate_buffer_handle(ctx, &buffer);
            Resource::new(descriptor, ResourceType::Buffer(buffer))
        };

        Self {
            resource_cache: Vec::new(),
            constants: HashMap::new(),
            constants_offset: 0,
            nodes: Vec::new(),
            swapchain,
            frame_data,
            frame_number: 0,
            resources,
            frame_timeline_semaphore,
            swapchain_images,
            constants_buffer,
            descriptor_buffer,
            swapchain_image_index: 0,
        }
    }

    pub fn get_swapchain(&self) -> ResourceHandle {
        self.swapchain_images[self.swapchain_image_index]
    }

    pub fn import<T>(
        &mut self,
        value: T,
        ctx: &Context,
        bindless: &mut BindlessDescriptorHeap,
    ) -> ResourceHandle
    where
        T: Importable,
    {
        let index = self.resources.len();
        self.resources.push(value.resource(ctx, bindless));
        index
    }

    pub fn buffer(&mut self, size: u64, name: &'static str) -> ResourceHandle {
        if let Some(resource) = self.resource_cache.iter().find(|e| e.name == name) {
            resource.handle
        } else {
            let index = self.resources.len();
            let index2 = self.resource_cache.len();
            self.resource_cache.push(ResourceDescription {
                name,
                handle: index,
                ty: ResourceDescriptionType::Buffer {
                    size,
                    usage: vk::BufferUsageFlags::empty(),
                },
            });
            self.resources.push(Resource::new(
                DescriptorResourceHandle(!0),
                ResourceType::Uninitilized(index2),
            ));
            index
        }
    }

    pub fn image(&mut self, size: ImageSize, format: Format, name: &'static str) -> ResourceHandle {
        if let Some(resource) = self.resource_cache.iter().find(|e| e.name == name) {
            resource.handle
        } else {
            let index = self.resources.len();
            let index2 = self.resource_cache.len();
            self.resource_cache.push(ResourceDescription {
                name,
                handle: index,
                ty: ResourceDescriptionType::Image {
                    format,
                    size,
                    usage: vk::ImageUsageFlags::empty(),
                },
            });
            self.resources.push(Resource::new(
                DescriptorResourceHandle(!0),
                ResourceType::Uninitilized(index2),
            ));
            index
        }
    }

    fn resource(
        &mut self,
        ctx: &mut Context,
        bindless: &mut BindlessDescriptorHeap,
        desc: &ResourceDescription,
    ) -> Resource {
        match &desc.ty {
            ResourceDescriptionType::Buffer { size, usage } => {
                let buffer = Buffer::new(ctx, *usage, MemoryLocation::GpuOnly, *size).unwrap();
                ctx.set_debug_name(desc.name, buffer.buffer);
                buffer.resource(&ctx, bindless)
            }
            ResourceDescriptionType::Image {
                size,
                usage,
                format,
            } => {
                let size = size.size();
                let image = Image::new_2d(
                    ctx,
                    *usage,
                    MemoryLocation::GpuOnly,
                    *format,
                    size.x,
                    size.y,
                )
                .unwrap();
                ctx.set_debug_name(desc.name, image.image);
                image.resource(ctx, bindless)
            }
        }
    }

    fn image_handle<'a>(&'a self, handle: ResourceHandle) -> Option<ImageHandle> {
        if let ResourceType::Image(image) = &self.resources[handle].ty {
            Some(image.handle())
        } else {
            None
        }
    }
    fn buffer_handle<'a>(&'a self, handle: ResourceHandle) -> Option<BufferHandle> {
        if let ResourceType::Buffer(buffer) = &self.resources[handle].ty {
            Some(buffer.handle())
        } else {
            None
        }
    }
}

pub fn draw_frame(
    mut rg: ResMut<RenderGraph>,
    mut ctx: ResMut<Context>,
    mut bindless: ResMut<BindlessDescriptorHeap>,
    mut cache: ResMut<PipelineCache>,
    raytracing: Option<Res<RayTracingContext>>,
) {
    for i in 0..rg.resources.len() {
        if let ResourceType::Uninitilized(index) = rg.resources[i].ty {
            let desc = rg.resource_cache[index].clone();
            rg.resources[i] = rg.resource(&mut ctx, &mut bindless, &desc);
        }
    }
    rg.resources.iter_mut().for_each(|resource| {
        resource.event.invalidated_in_stage = [vk::AccessFlags2::empty(); 25];
        resource.event.pipeline_barrier_src_stages = vk::PipelineStageFlags2::empty();
        resource.event.to_flush = vk::AccessFlags2::default();
    });

    let root_node = rg
        .nodes
        .iter()
        .position(|e| {
            e.edges
                .iter()
                .position(|e| e.resource == rg.get_swapchain())
                .is_some()
        })
        .unwrap();
    let frame_in_flight = rg.frame_number as usize % FRAMES_IN_FLIGHT;
    let frame = rg.frame_data[frame_in_flight].clone();
    bindless
        .bind(&ctx, raytracing.is_some(), &frame.cmd)
        .unwrap();
    let descriptor_offsets = rg.write_bindings().unwrap();
    let execution_order = if rg.nodes.len() > 2 {
        rg.bake(root_node).unwrap()
    } else if rg.nodes.len() == 2 {
        vec![0, 1]
    } else {
        vec![0]
    };

    let barriers = rg.create_barriers(&execution_order);
    for (pass_index, pass_handle) in execution_order.iter().enumerate() {
        let pass = &rg.nodes[*pass_handle];
        unsafe {
            ctx.cmd_start_label(&frame.cmd, &pass.name);
            pass.cmd_push_constants(
                &rg,
                &ctx,
                &bindless,
                &frame,
                descriptor_offsets[*pass_handle] as u32 * size_of::<u32>() as u32,
            );

            let barrier = &barriers[pass_index];
            // println!("{}:{:#?}", pass.name, barrier);
            if barrier.images.len() != 0 || barrier.buffers.len() != 0 {
                ctx.cmd_insert_label(&frame.cmd, &format!("Barrier for {}", pass.name));
                let dependency_info = vk::DependencyInfo::default()
                    .buffer_memory_barriers(&barrier.buffers)
                    .image_memory_barriers(&barrier.images);
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
            }

            pass.execution
                .execute(
                    &frame.cmd,
                    &rg,
                    &mut ctx,
                    raytracing.as_deref(),
                    &mut cache,
                    &pass.edges,
                )
                .unwrap();
            ctx.cmd_end_label(&frame.cmd);
        }
    }

    if let Some(barrier) = barriers.get(execution_order.len()) {
        // println!("{:#?}", barrier);
        ctx.cmd_insert_label(&frame.cmd, "Transitioning Swapchain Image");
        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&barrier.buffers)
            .image_memory_barriers(&barrier.images);
        unsafe {
            ctx.device
                .cmd_pipeline_barrier2(frame.cmd, &dependency_info)
        };
    }

    let end_stage = rg.nodes[root_node].execution.get_stages();
    unsafe { ctx.device.end_command_buffer(frame.cmd).unwrap() };

    let wait_semaphores = [vk::SemaphoreSubmitInfo::default()
        .semaphore(rg.swapchain.frame_resources[frame_in_flight].image_availible_semaphore)
        .stage_mask(end_stage)];

    let signal_frame_value = frame.frame_number + FRAMES_IN_FLIGHT as u64;
    rg.frame_data[frame_in_flight].frame_number = signal_frame_value;

    let signal_semaphores = [
        vk::SemaphoreSubmitInfo::default()
            .semaphore(
                rg.swapchain.frame_resources[frame_in_flight as usize].render_finished_semaphore,
            )
            .stage_mask(end_stage),
        vk::SemaphoreSubmitInfo::default()
            .semaphore(rg.frame_timeline_semaphore)
            .stage_mask(end_stage)
            .value(signal_frame_value),
    ];

    ctx.submit(&frame.cmd, &wait_semaphores, &signal_semaphores);

    rg.swapchain
        .present(&ctx, frame_in_flight, rg.swapchain_image_index);

    rg.frame_number += 1;
}
pub fn begin_frame(mut rg: ResMut<RenderGraph>, mut ctx: ResMut<Context>) {
    let frame_in_flight = rg.frame_number as usize % FRAMES_IN_FLIGHT;
    let frame = rg.frame_data[frame_in_flight].clone();

    let semaphore = [rg.frame_timeline_semaphore];
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

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        ctx.device
            .begin_command_buffer(frame.cmd, &begin_info)
            .unwrap()
    };

    rg.swapchain_image_index = rg.swapchain.next_image(frame_in_flight);

    rg.constants.clear();
    rg.nodes.clear();
    rg.constants_offset = 0;
}
