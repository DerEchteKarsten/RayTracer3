use std::{fmt::Debug, marker::PhantomData, os::raw::c_void};

use ash::vk::{self, BufferCopy, BufferUsageFlags, Packed24_8};
use bevy_app::prelude::*;
use bevy_asset::Handle;
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::Mesh, raytracing::{AccelerationStructure, RayTracingContext}
};

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    render_graph::{resources::ResourceHandle, RenderGraph},
    vulkan::{
        buffer::{Buffer, BufferType, DynamicBuffer},
        Context,
    },
};

const INSTANCE_BUFFER_CAPACITY: u64 = 65536; //TODO
const GEOMETRY_BUFFER_CAPACITY: u64 = 65536; //TODO
const TRANSFORM_BUFFER_CAPACITY: u64 = 65536; //TODO
const VERTEX_BUFFER_CAPACITY: u64 = 65536; //TODO
const ACCELERATION_STRUCTURE_SCRATCH_MEMORY: u64 = 65536; //TODO
const STAGING_BUFFER_SIZE: u64 = 65536; //TODO

#[derive(Clone, Copy)]
#[repr(C)]
struct InstanceInfo {
    mesh_index: DescriptorResourceHandle,
    transform: Mat4,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct InstanceInfo4Pack {
    infos: [InstanceInfo; 4],
}

#[derive(Component, Clone)]
pub(crate) struct Instance {
}

#[derive(Component, Clone)]
pub(crate) struct Transform {
    transform: Mat4,
}

impl Transform {
    pub fn from_position(pos: Vec3) -> Self {
        Self {
            transform: Mat4::from_translation(pos),
        }
    }
    pub fn new(mat: Mat4) -> Self {
        Self { transform: mat }
    }
}

impl Debug for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (position, rotation, scale) = self.transform.to_scale_rotation_translation();
        write!(
            f,
            "Position: {:#?}\nRotation: {:#?}\nScale: {:#?}",
            position, rotation, scale
        )
    }
}

#[derive(Resource)]
pub struct RenderWorld {
    loading: Vec<Handle<Mesh>>,
    global_vertex_buffer: DynamicBuffer,
    global_instance_buffer: DynamicBuffer,
    global_transform_buffer: DynamicBuffer,
    global_geometry_buffer: DynamicBuffer,
    staging_buffer: Buffer,
    acceleration_structure_scratch_memory: DynamicBuffer,
    acceleration_structure_memory: DynamicBuffer,
    tlas: AccelerationStructure,
    num_instances: usize,
}

#[derive(Resource)]
pub struct WorldResources {
    pub tlas: ResourceHandle,
    pub instances: ResourceHandle,
}

pub(super) fn init_world(
    mut cmd: Commands,
    mut ctx: ResMut<Context>,
    raytracing_ctx: Res<RayTracingContext>,
    mut bindless: ResMut<BindlessDescriptorHeap>,
    mut rg: ResMut<RenderGraph>,
) {
    let mut acceleration_structure_scratch_memory = DynamicBuffer::new(&mut ctx, &mut bindless, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, ACCELERATION_STRUCTURE_SCRATCH_MEMORY, None).unwrap();
    let acceleration_structure_memory = DynamicBuffer::new(&mut ctx, &mut bindless, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, ACCELERATION_STRUCTURE_SCRATCH_MEMORY, None).unwrap();

    let tlas = ctx.execute_one_time_commands(|cmd, ctx| {
        raytracing_ctx
            .create_acceleration_structure(
                ctx,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                &[],
                &[],
                &[],
                &acceleration_structure_memory,
                0,
                &mut acceleration_structure_scratch_memory,
                &cmd,
                &mut bindless,
            )
        .unwrap()
    }).unwrap();

    let tlas_descriptor = bindless.allocate_acceleration_structure_handle(&ctx, &tlas);
    let render_world = RenderWorld {
        loading: Vec::new(),
        global_geometry_buffer: DynamicBuffer::new(&mut ctx, &mut bindless, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, GEOMETRY_BUFFER_CAPACITY, None).unwrap(),
        global_instance_buffer: DynamicBuffer::new(&mut ctx, &mut bindless, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, INSTANCE_BUFFER_CAPACITY, None).unwrap(),
        global_transform_buffer: DynamicBuffer::new(&mut ctx, &mut bindless, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, TRANSFORM_BUFFER_CAPACITY, None).unwrap(),
        global_vertex_buffer: DynamicBuffer::new(&mut ctx, &mut bindless, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, VERTEX_BUFFER_CAPACITY, None).unwrap(),
        acceleration_structure_scratch_memory,
        acceleration_structure_memory,
        num_instances: 0,
        staging_buffer: Buffer::new(&mut ctx, BufferUsageFlags::empty(), MemoryLocation::CpuToGpu, STAGING_BUFFER_SIZE).unwrap(),
        tlas,
    };

    cmd.insert_resource(WorldResources {
        instances: rg.import(render_world.global_instance_buffer.bindless_handle, &ctx, &mut bindless),
        tlas: rg.import(tlas_descriptor, &ctx, &mut bindless),
    });
    cmd.insert_resource(render_world);
}