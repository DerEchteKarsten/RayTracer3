use std::{fmt::Debug, marker::PhantomData, os::raw::c_void};

use ash::vk::{self, BufferCopy, BufferUsageFlags, Packed24_8};
use bevy_app::prelude::*;
use bevy_asset::{AssetEvent, Assets, Handle};
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use glam::{Mat4, Vec3};

use crate::{assets::model::mat4_to_vk_transform, raytracing::{AccelerationStructure, RayTracingContext}, Model};

use super::{
    bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    render_graph::{resources::ResourceHandle, RenderGraph},
    vulkan::{
        buffer::{Buffer, BufferType},
        Context,
    },
};

#[derive(Clone, Copy)]
#[repr(C)]
struct InstanceInfo {
    // mesh_index: DescriptorResourceHandle,
    index_buffer: DescriptorResourceHandle, //TODO Remove these
    vertex_buffer: DescriptorResourceHandle, //TODO Remove these
    transform_buffer: DescriptorResourceHandle, //TODO Remove these
    geometry_buffer: DescriptorResourceHandle, //TODO Remove these
    transform: Mat4,
}

#[derive(Component, Clone)]
pub(crate) struct Instance {
    pub model: Handle<Model>,
}

#[derive(Component, Clone)]
struct Extracted(usize);

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
struct RenderWorld {
    scene_buffer: Buffer,
    scene_staging_buffer: Buffer,
    num_instances: u64,
    entitys: EntityHashMap<usize>,
    tlas: AccelerationStructure,
}

#[derive(Resource)]
pub struct WorldResources {
    pub tlas: ResourceHandle,
    pub instances: ResourceHandle,
}

pub(super) fn init_world(mut cmd: Commands, mut ctx: ResMut<Context>, raytracing_ctx: Res<RayTracingContext>, mut bindless: ResMut<BindlessDescriptorHeap>,  mut rg: ResMut<RenderGraph>) {
    let scene_buffer = Buffer::new(
        &mut ctx,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuOnly,
        size_of::<InstanceInfo>() as u64 * 10000,
    )
    .unwrap();
    let tlas = raytracing_ctx.create_acceleration_structure(&mut ctx, vk::AccelerationStructureTypeKHR::TOP_LEVEL, &[], &[], &[]).unwrap();
    let tlas_descriptor = bindless.allocate_acceleration_structure_handle(&ctx, &tlas);
    cmd.insert_resource(WorldResources {
        instances: rg.import(scene_buffer.handle(), &ctx, &mut bindless),
        tlas: rg.import(tlas_descriptor, &ctx, &mut bindless)
    });
    
    cmd.insert_resource(RenderWorld {
        entitys: EntityHashMap::with_capacity(10000),
        scene_buffer,
        tlas,
        scene_staging_buffer: Buffer::new(
            &mut ctx,
            BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::GpuToCpu,
            size_of::<InstanceInfo>() as u64 * 10000,
        )
        .unwrap(),
        num_instances: 0,
    });
}

pub(super) fn extract_instances(
    mut cmd: Commands,
    mut render_world: ResMut<RenderWorld>,
    models: Res<Assets<Model>>,
    query: Query<(Entity, &Transform, &Instance), Without<Extracted>>,
    ctx: Res<Context>,
    mut bindless: ResMut<BindlessDescriptorHeap>,
) {
    if query.is_empty() {return;}
    for (entity, transform, instance) in query {
        let Some(model) = models.get(&instance.model) else {
            log::error!("Model not Found");
            continue;
        };
        if let Some(render_model) = &model.render_model {
            let index = render_world.num_instances as usize;
            write(
                &mut render_world,
                InstanceInfo {
                    geometry_buffer: bindless
                        .allocate_buffer_handle(&ctx, &render_model.geometry_info_buffer),
                    index_buffer: bindless.allocate_buffer_handle(&ctx, &render_model.index_buffer),
                    transform_buffer: bindless
                        .allocate_buffer_handle(&ctx, &render_model.transform_buffer),
                    vertex_buffer: bindless
                        .allocate_buffer_handle(&ctx, &render_model.vertex_buffer),
                    transform: transform.transform,
                },
                index,
            );
            render_world.num_instances += 1;

            render_world.entitys.insert(entity, index);
            cmd.entity(entity).insert(Extracted(index));
        }
    }

    if let Err(e) = ctx.execute_one_time_commands(|cmd| {
        log::debug!("Copying");
        render_world
            .scene_staging_buffer
            .copy(&ctx, cmd, &render_world.scene_buffer);
    }) {
        log::error!("{e}");
    };
}

pub(super) fn apply_transform(
    mut render_world: ResMut<RenderWorld>,
    raytracing_ctx: Res<RayTracingContext>,
    query: Query<(&Transform, &Extracted), Changed<Transform>>,
    ctx: Res<Context>,
) {
    if query.is_empty() {return;}
    for (transform, index) in query {
        let mut instance = read(&render_world, index.0);
        instance.transform = transform.transform;
        write(&mut render_world, instance, index.0);
        let instaces = &[vk::AccelerationStructureInstanceKHR {
            transform: mat4_to_vk_transform(transform.transform),
            instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_CULL_DISABLE_NV.as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: self.blas.as_ref().unwrap().address,
            },
        }];
    }
    let tlas = {

        let instance_buffer = Buffer::from_data(
            &mut ctx,
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

    if let Err(e) = ctx.execute_one_time_commands(|cmd| {
        render_world
            .scene_staging_buffer
            .copy(&ctx, cmd, &render_world.scene_buffer);
        unsafe { raytracing_ctx.acceleration_structure_fn.cmd_build_acceleration_structures(*cmd, infos, build_range_infos) };
    }) {
        log::error!("{e}");
    };
}

fn write(render_world: &mut RenderWorld, instance_info: InstanceInfo, index: usize) {
    unsafe {
        render_world
            .scene_staging_buffer
            .allocation
            .as_ref()
            .unwrap()
            .mapped_ptr()
            .unwrap()
            .as_ptr()
            .add(index * size_of::<InstanceInfo>())
            .copy_from(
                &instance_info as *const InstanceInfo as *const c_void,
                size_of::<InstanceInfo>(),
            );
    };
}

fn read(render_world: &RenderWorld, index: usize) -> InstanceInfo {
    unsafe {
        *(render_world
            .scene_staging_buffer
            .allocation
            .as_ref()
            .unwrap()
            .mapped_ptr()
            .unwrap()
            .as_ptr()
            .add(index * size_of::<InstanceInfo>()) as *const InstanceInfo)
    }
}

fn swap_remove(render_world: &mut RenderWorld, index: usize) {
    let first = read(render_world, render_world.num_instances as usize - 1);
    write(render_world, first, index);
    render_world.num_instances -= 1;
}

pub(super) fn removed_instances(
    mut render_world: ResMut<RenderWorld>,
    mut events: RemovedComponents<Extracted>,
    ctx: Res<Context>,
) {
    if events.is_empty() {
        return;
    }
    let mut regions = vec![];
    for entity in events.read() {
        let Some(index) = render_world.entitys.get(&entity).cloned() else {
            continue;
        };
        swap_remove(&mut render_world, index);
        regions.push(BufferCopy {
            size: size_of::<InstanceInfo>() as u64,
            dst_offset: index as u64,
            src_offset: index as u64,
        });
    }
    if let Err(e) = ctx.execute_one_time_commands(|cmd| {
        render_world
            .scene_staging_buffer
            .copy(&ctx, cmd, &render_world.scene_buffer);
    }) {
        log::error!("{e}");
    };
}