use std::mem::{size_of, size_of_val};

use super::gltf::{self, Vertex};
use crate::{
    renderer::{
        bindless::{BindlessDescriptorHeap, DescriptorResourceHandle, ImmutableSampler},
        vulkan::{
            raytracing::{AccelerationStructure, RayTracingContext},
            Context,
        },
    },
    vulkan::{
        buffer::{Buffer, BufferType},
        image::{Image, ImageType},
    },
};
use anyhow::Result;
use ash::vk::{self, Filter, Packed24_8};
use bevy_asset::{io::Reader, Asset, AssetLoader, AsyncReadExt, LoadContext};
use bevy_reflect::TypePath;
use glam::{Mat4, Vec4};
use gpu_allocator::MemoryLocation;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GeometryInfo {
    pub color: [f32; 4],
    pub color_texture: DescriptorResourceHandle,
    pub sampler: ImmutableSampler,
    pub metallic_factor: f32,
    pub index_offset: u32,
    pub vertex_offset: u32,
    pub emission: [f32; 4],
    pub roughness: f32,
}

pub struct RenderModel {
    pub images: Vec<Image>,
    pub blas: Option<AccelerationStructure>,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub transform_buffer: Buffer,
    pub geometry_info_buffer: Buffer,
    pub geometry_infos: Vec<GeometryInfo>,
}

pub const IDENTITY_MATRIX: vk::TransformMatrixKHR = vk::TransformMatrixKHR {
    matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
};

pub fn mat4_to_vk_transform(mat: Mat4) -> vk::TransformMatrixKHR {
    let transform = mat.to_cols_array_2d();
    let r0 = transform[0];
    let r1 = transform[1];
    let r2 = transform[2];
    let r3 = transform[3];

    #[rustfmt::skip]
    let matrix = [
        r0[0], r1[0], r2[0], r3[0],
        r0[1], r1[1], r2[1], r3[1],
        r0[2], r1[2], r2[2], r3[2],
    ];

    vk::TransformMatrixKHR { matrix }
}

impl RenderModel {
    pub fn instance(&self, transform: Mat4) -> vk::AccelerationStructureInstanceKHR {
        vk::AccelerationStructureInstanceKHR {
            transform: mat4_to_vk_transform(transform),
            instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_CULL_DISABLE_NV.as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: self.blas.as_ref().unwrap().address,
            },
        }
    }

    // pub fn from_vertices(
    //     ctx: &mut Context,
    //     vertices: &[Vertex],
    //     indices: &[u32],
    //     transform: Mat4,
    // ) -> Result<Self> {
    //     //println!("{:?}, \n {:?}", indices.len() / 3, vertices.len());
    //     let vertex_buffer = ctx.create_gpu_only_buffer_from_data(
    //         vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    //             | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    //             | vk::BufferUsageFlags::STORAGE_BUFFER,
    //         vertices,
    //     )?;
    //     let index_buffer = ctx.create_gpu_only_buffer_from_data(
    //         vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    //             | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    //             | vk::BufferUsageFlags::STORAGE_BUFFER,
    //         indices,
    //     )?;
    //     let vk_transform = mat4_to_vk_transform(transform);
    //     let transform_buffer = ctx.create_gpu_only_buffer_from_data(
    //         vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    //             | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    //             | vk::BufferUsageFlags::STORAGE_BUFFER,
    //         from_ref(&vk_transform),
    //     )?;
    //     let blas = {
    //         let data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
    //             .vertex_data(vk::DeviceOrHostAddressConstKHR {
    //                 device_address: vertex_buffer.get_device_address(&ctx.device),
    //             })
    //             .index_type(vk::IndexType::UINT32)
    //             .index_data(vk::DeviceOrHostAddressConstKHR {
    //                 device_address: index_buffer.get_device_address(&ctx.device),
    //             })
    //             .max_vertex(vertices.len() as _)
    //             .transform_data(vk::DeviceOrHostAddressConstKHR {
    //                 device_address: transform_buffer.get_device_address(&ctx.device),
    //             })
    //             .vertex_format(vk::Format::R32G32B32_SFLOAT)
    //             .vertex_stride(size_of::<Vertex>() as _)
    //             ;

    //         let geometry = vk::AccelerationStructureGeometryKHR::default()
    //             .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data })
    //             .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
    //             ;

    //         log::debug!("{}, {}", indices.len(), vertices.len());

    //         let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
    //             .first_vertex(0)
    //             .primitive_count((vertices.len() as u32) / 3)
    //             .primitive_offset(0)
    //             .transform_offset(0)
    //             ;
    //         create_acceleration_structure(
    //             ctx,
    //             vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
    //             from_ref(&geometry),
    //             from_ref(&range),
    //             from_ref(&1),
    //         )?
    //     };

    //     let geometry_info = GeometryInfo {
    //         transform: transform,
    //         base_color: [1.0,1.0,1.0,1.0],
    //         base_color_texture_index: -1,
    //         metallic_factor: 0.0,
    //         vertex_offset: 0,
    //         index_offset: 0,
    //     };

    //     let geometry_info_buffer = ctx.create_gpu_only_buffer_from_data(
    //         vk::BufferUsageFlags::STORAGE_BUFFER,
    //         from_ref(&geometry_info),
    //     )?;

    //     Ok(Self {
    //         blas,
    //         index_buffer,
    //         transform_buffer,
    //         vertex_buffer,
    //         geometry_info_buffer,
    //     })
    // }

    fn map_gltf_sampler<'a>(sampler: &gltf::Sampler) -> ImmutableSampler {
        let mag_filter = match sampler.mag_filter {
            gltf::MagFilter::Linear => vk::Filter::LINEAR,
            gltf::MagFilter::Nearest => vk::Filter::NEAREST,
        };

        let min_filter = match sampler.min_filter {
            gltf::MinFilter::Linear
            | gltf::MinFilter::LinearMipmapLinear
            | gltf::MinFilter::LinearMipmapNearest => vk::Filter::LINEAR,
            gltf::MinFilter::Nearest
            | gltf::MinFilter::NearestMipmapLinear
            | gltf::MinFilter::NearestMipmapNearest => vk::Filter::NEAREST,
        };

        ImmutableSampler::new(mag_filter, min_filter).unwrap()
    }

    pub fn from_gltf(
        ctx: &mut Context,
        bindless: &mut BindlessDescriptorHeap,
        raytracing_ctx: Option<&mut RayTracingContext>,
        model: &gltf::GltfModel,
    ) -> Result<Self> {
        let vertices = model.vertices.as_slice();
        let indices = model.indices.as_slice();

        let transforms = model
            .nodes
            .iter()
            .map(|n| {
                let transform = n.transform;
                let r0 = transform[0];
                let r1 = transform[1];
                let r2 = transform[2];
                let r3 = transform[3];

                #[rustfmt::skip]
                let matrix = [
                    r0[0], r1[0], r2[0], r3[0],
                    r0[1], r1[1], r2[1], r3[1],
                    r0[2], r1[2], r2[2], r3[2],
                ];

                vk::TransformMatrixKHR { matrix }
            })
            .collect::<Vec<_>>();

        let transform_buffer = Buffer::from_data(
            ctx,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            transforms.as_slice(),
        )?;

        let mut images = vec![];

        model.images.iter().try_for_each::<_, Result<_>>(|i| {
            let width = i.width;
            let height = i.height;
            let pixels = i.pixels.as_slice();

            let staging = Buffer::new(
                ctx,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                size_of_val(pixels) as _,
            )?;

            staging.copy_data_to_buffer(pixels)?;

            let image = Image::new_2d(
                ctx,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                MemoryLocation::GpuOnly,
                vk::Format::R8G8B8A8_SRGB,
                width,
                height,
            )?;
            ctx.execute_one_time_commands(|cmd| {
                unsafe {
                    ctx.device.cmd_pipeline_barrier2(
                        *cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[image
                            .memory_barrier(
                                vk::ImageLayout::UNDEFINED,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            )]),
                    )
                };

                staging.copy_to_image(&ctx, cmd, &image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

                unsafe {
                    ctx.device.cmd_pipeline_barrier2(
                        *cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[image
                            .memory_barrier(
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            )]),
                    )
                };
            })?;

            images.push(image);

            Ok(())
        })?;

        if images.is_empty() {
            let image = Image::new_2d(
                ctx,
                vk::ImageUsageFlags::SAMPLED,
                MemoryLocation::GpuOnly,
                vk::Format::R8G8B8A8_SRGB,
                1,
                1,
            )?;

            ctx.execute_one_time_commands(|cmd| {
                unsafe {
                    ctx.device.cmd_pipeline_barrier2(
                        *cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[image
                            .memory_barrier(
                                vk::ImageLayout::UNDEFINED,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            )]),
                    )
                };
            })?;

            images.push(image);
        }

        let descriptor_handels = images
            .iter()
            .map(|e| bindless.allocate_texture_handle(&ctx, &e.handle()))
            .collect::<Vec<_>>();

        let vertex_buffer = Buffer::from_data(
            ctx,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
        )?;

        let index_buffer = Buffer::from_data(
            ctx,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER,
            indices,
        )?;

        let as_geo_triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_buffer.address,
            })
            .vertex_stride(size_of::<Vertex>() as _)
            .max_vertex(model.vertices.len() as _)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_buffer.address,
            })
            .transform_data(vk::DeviceOrHostAddressConstKHR {
                device_address: transform_buffer.address,
            });

        let mut geometry_infos = vec![];
        let mut as_geometries = vec![];
        let mut as_ranges = vec![];
        let mut max_primitive_counts = vec![];
        let mut index_counts = vec![];
        let mut lights = 0;
        for (node_index, node) in model.nodes.iter().enumerate() {
            let mesh = node.mesh;

            let primitive_count = (mesh.index_count / 3) as u32;

            let emission = [
                mesh.material.emission[0],
                mesh.material.emission[1],
                mesh.material.emission[2],
                1.0,
            ];
            if emission[0] != 0.0 || emission[1] != 0.0 || emission[2] != 0.0 {
                lights += primitive_count;
            }
            let texture = mesh
                .material
                .base_color_texture_index
                .and_then(|i| Some(model.textures[i as usize]));
            geometry_infos.push(GeometryInfo {
                color: mesh.material.base_color,
                emission,
                color_texture: texture
                    .and_then(|texture: gltf::Texture| {
                        Some(descriptor_handels[texture.image_index as usize])
                    })
                    .unwrap_or(DescriptorResourceHandle(!0)),
                sampler: Self::map_gltf_sampler(
                    &model.samplers[texture
                        .and_then(|texture| Some(texture.sampler_index as usize))
                        .unwrap_or(0)],
                ),
                metallic_factor: mesh.material.metallic_factor,
                vertex_offset: mesh.vertex_offset,
                index_offset: mesh.index_offset,
                roughness: mesh.material.roughness,
            });

            index_counts.push(mesh.index_count);

            as_geometries.push(
                vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .flags(vk::GeometryFlagsKHR::OPAQUE)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: as_geo_triangles_data,
                    }),
            );

            as_ranges.push(
                vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .first_vertex(mesh.vertex_offset)
                    .primitive_count(primitive_count)
                    .primitive_offset(mesh.index_offset * size_of::<u32>() as u32)
                    .transform_offset((node_index * size_of::<vk::TransformMatrixKHR>()) as u32),
            );

            max_primitive_counts.push(primitive_count)
        }
        let geometry_info_buffer = Buffer::from_data(
            ctx,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            geometry_infos.as_slice(),
        )?;

        let blas = raytracing_ctx.and_then(|raytracing_ctx| {
            Some(
                raytracing_ctx
                    .create_acceleration_structure(
                        ctx,
                        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                        &as_geometries,
                        &as_ranges,
                        &max_primitive_counts,
                    )
                    .unwrap(),
            )
        });

        Ok(Self {
            images,
            geometry_infos,
            blas,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            geometry_info_buffer,
        })
    }
}
