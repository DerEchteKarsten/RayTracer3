use std::mem::{size_of, size_of_val};

use super::gltf::{self, Vertex};
use crate::backend::{
    raytracing::{AccelerationStructure, RayTracingContext},
    utils::{Buffer, Image, ImageResource},
    vulkan_context::Context,
};
use anyhow::Result;
use ash::vk::{self, Packed24_8};
use glam::{Mat4, Vec4};
use gpu_allocator::MemoryLocation;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GeometryInfo {
    pub transform: Mat4,
    pub base_color: [f32; 4],
    pub base_color_texture_index: i32,
    pub metallic_factor: f32,
    pub index_offset: u32,
    pub vertex_offset: u32,
    pub emission: [f32; 4],
    pub roughness: f32,
}

pub struct Model {
    pub index_count: u32,
    pub images: Vec<ImageResource>,
    pub samplers: Vec<vk::Sampler>,
    pub textures: Vec<(usize, usize)>,
    pub blas: AccelerationStructure,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub transform_buffer: Buffer,
    pub geometry_info_buffer: Buffer,
    pub geometry_infos: Vec<GeometryInfo>,
    pub index_counts: Vec<u32>,
    pub lights: u32,
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

impl Model {
    pub fn instance(&self, transform: Mat4) -> vk::AccelerationStructureInstanceKHR {
        vk::AccelerationStructureInstanceKHR {
            transform: mat4_to_vk_transform(transform),
            instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_CULL_DISABLE_NV.as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: self.blas.address,
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

    fn map_gltf_sampler<'a>(sampler: &gltf::Sampler) -> vk::SamplerCreateInfo {
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

        vk::SamplerCreateInfo::default()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
    }

    pub fn from_gltf(
        model: gltf::Model,
    ) -> Result<Self> {
        let ctx = Context::get();

        let vertices = model.vertices.as_slice();
        let indices = model.indices.as_slice();
        for v in vertices {
            if v.color != Vec4::new(1.0, 1.0, 1.0, 1.0) {
                println!("{:?}", v);
            }
        }
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
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            transforms.as_slice(),
        )?;

        let mut images = vec![];

        model.images.iter().try_for_each::<_, Result<_>>(|i| {
            let width = i.width;
            let height = i.height;
            let pixels = i.pixels.as_slice();

            let staging = Buffer::new(
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                size_of_val(pixels) as _,
            )?;

            staging.copy_data_to_buffer(pixels)?;

            let image = ImageResource::new_2d(
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                MemoryLocation::GpuOnly,
                vk::Format::R8G8B8A8_SRGB,
                width,
                height,
            )?;
            let ctx = Context::get();
            ctx.execute_one_time_commands(|cmd| {
                unsafe {
                    ctx.device.cmd_pipeline_barrier2(
                        *cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[image
                            .image
                            .memory_barrier(
                                vk::ImageLayout::UNDEFINED,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            )]),
                    )
                };

                staging.copy_to_image(
                    cmd,
                    &image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                unsafe {
                    ctx.device.cmd_pipeline_barrier2(
                        *cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[image
                            .image
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
            let image = ImageResource::new_2d(
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
                            .image
                            .memory_barrier(
                                vk::ImageLayout::UNDEFINED,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            )]),
                    )
                };
            })?;

            images.push(image);
        }

        let mut samplers = model
            .samplers
            .iter()
            .map(|s| {
                let sampler_info = Self::map_gltf_sampler(s);
                unsafe { ctx.device.create_sampler(&sampler_info, None) }.unwrap()
            })
            .collect::<Vec<_>>();

        //Dummy sampler
        if samplers.is_empty() {
            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                max_anisotropy: 1.0,
                border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                compare_op: vk::CompareOp::NEVER,
                ..Default::default()
            };
            let sampler = unsafe { ctx.device.create_sampler(&sampler_info, None) }.unwrap();
            samplers.push(sampler);
        }

        let mut textures = model
            .textures
            .iter()
            .map(|t| (t.image_index, t.sampler_index))
            .collect::<Vec<_>>();

        // Dummy texture
        if textures.is_empty() {
            textures.push((0, 0));
        }

        let vertex_buffer = Buffer::from_data(
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
        )?;

        let index_buffer = Buffer::from_data(
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
                lights += mesh.index_count / 3;
            }
            geometry_infos.push(GeometryInfo {
                transform: Mat4::from_cols_array_2d(&node.transform),
                base_color: mesh.material.base_color,
                emission,
                base_color_texture_index: mesh
                    .material
                    .base_color_texture_index
                    .map_or(-1, |i| i as _),
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
            vk::BufferUsageFlags::STORAGE_BUFFER,
            geometry_infos.as_slice(),
        )?;

        let blas = RayTracingContext::get().create_acceleration_structure(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &as_geometries,
            &as_ranges,
            &max_primitive_counts,
        )?;

        Ok(Self {
            index_count: indices.len() as u32,
            images,
            samplers,
            textures,
            geometry_infos,
            blas,
            index_counts,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            geometry_info_buffer,
            lights,
        })
    }
}
