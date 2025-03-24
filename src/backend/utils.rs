use std::{collections::HashMap, ffi::CStr, mem::MaybeUninit, sync::{Arc, Once}};
use once_cell::sync::Lazy;
use anyhow::{Ok, Result};
use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc},
    MemoryLocation,
};
use image::{DynamicImage, GenericImageView};

use super::vulkan_context::Context;

pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: Allocation,
    pub(crate) address: vk::DeviceAddress,
    pub(crate) size: vk::DeviceSize,
}

#[derive(Default)]
pub struct Image {
    pub(crate) image: vk::Image,
    pub(crate) format: vk::Format,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) extent: vk::Extent2D,
}

#[derive(Default)]
pub struct ImageResource {
    pub(crate) image: Image,
    pub(crate) view: vk::ImageView,
    pub(crate) extent: vk::Extent2D,
}

impl Image {
    pub fn copy(
        &self,
        cmd: &vk::CommandBuffer,
        other: &Image,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
        let ctx = Context::get();
        unsafe {
            ctx.device.cmd_copy_image(
                *cmd,
                self.image,
                src_layout,
                other.image,
                dst_layout,
                &[vk::ImageCopy::default()
                    .extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    })
                    .src_offset(vk::Offset3D::default())
                    .dst_offset(vk::Offset3D::default())
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            )
        };
    }

    pub fn blit(
        &self,
        cmd: &vk::CommandBuffer,
        other: &Image,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
        let ctx = Context::get();
        unsafe {
            ctx.device.cmd_blit_image(
                *cmd,
                self.image,
                src_layout,
                other.image,
                dst_layout,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: self.extent.width as _,
                            y: self.extent.height as _,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: other.extent.width as _,
                            y: other.extent.height as _,
                            z: 1,
                        },
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
                vk::Filter::NEAREST,
            )
        };
    }

    pub fn get_pipeline_stage_acces_tuple(
        state: vk::ImageLayout,
    ) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        match state {
            vk::ImageLayout::UNDEFINED => {
                (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::NONE)
            }
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER
                    | vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS,
                vk::AccessFlags2::SHADER_READ,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::GENERAL => (
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::MEMORY_READ
                    | vk::AccessFlags2::MEMORY_WRITE
                    | vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::PRESENT_SRC_KHR => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::NONE,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            _ => {
                log::error!("Unsupported layout transition!");
                (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                )
            }
        }
    }

    pub fn subresource_range_memory_barrier<'a>(
        &self,
        subresource_range: vk::ImageSubresourceRange,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'a> {
        let (src_stage, src_access) = Self::get_pipeline_stage_acces_tuple(old_layout);
        let (dst_stage, dst_access) = Self::get_pipeline_stage_acces_tuple(new_layout);
        vk::ImageMemoryBarrier2::default()
            .dst_access_mask(dst_access)
            .dst_stage_mask(dst_stage)
            .src_access_mask(src_access)
            .src_stage_mask(src_stage)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(self.image)
            .subresource_range(subresource_range)
    }

    pub fn memory_barrier<'a>(
        &self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'a> {
        self.subresource_range_memory_barrier(
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                base_mip_level: 0,
                layer_count: 1,
                level_count: 1,
            },
            old_layout,
            new_layout,
        )
    }

    pub fn new_subresource(
        self,
        device: &ash::Device,
        extent: vk::Extent2D,
        subresource_range: vk::ImageSubresourceRange,
    ) -> ImageResource {
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(subresource_range);
        let view = unsafe { device.create_image_view(&image_view_info, None) }.unwrap();

        ImageResource {
            extent,
            image: self,
            view,
        }
    }
    pub fn new_resource(self, device: &ash::Device, extent: vk::Extent2D) -> ImageResource {
        self.new_subresource(
            device,
            extent,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
    }

    pub(crate) fn new_2d(
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let ctx = Context::get_mut();
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { ctx.device.create_image(&image_info, None)? };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };

        let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
            name: "image",
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            ctx.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?
        };

        Ok(Self {
            image,
            allocation: Some(allocation),
            format,
            extent: vk::Extent2D {
                height: extent.height,
                width: extent.width,
            },
        })
    }
}

impl Buffer {
    pub(crate) fn new_aligned(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        alignment: Option<u64>,
    ) -> Result<Self> {
        let ctx = Context::get_mut();
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buffer = unsafe { ctx.device.create_buffer(&create_info, None)? };
        let mut requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if let Some(a) = alignment {
            requirements.alignment = a;
        }

        let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
            name: "buffer",
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);

        Ok(Self {
            buffer,
            allocation,
            address: unsafe { ctx.device.get_buffer_device_address(&addr_info) },
            size,
        })
    }

    pub(crate) fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Self::new_aligned(usage, memory_location, size, None)
    }

    pub(crate) fn copy_data_to_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        self.copy_data_to_aligned_buffer(data, align_of::<T>() as _)
    }

    pub(crate) fn copy_data_to_aligned_buffer<T: Copy>(
        &self,
        data: &[T],
        alignment: u32,
    ) -> Result<()> {
        unsafe {
            let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr();
            let mut align = ash::util::Align::new(data_ptr, alignment as _, size_of_val(data) as _);
            align.copy_from_slice(data);
        };

        Ok(())
    }

    pub(crate) fn destroy(self) -> Result<()> {
        let ctx = Context::get_mut();
        unsafe { ctx.device.destroy_buffer(self.buffer, None) };
        ctx.allocator.free(self.allocation).unwrap();
        Ok(())
    }

    pub(crate) fn copy_to_image(
        &self,
        cmd: &vk::CommandBuffer,
        dst: &Image,
        layout: vk::ImageLayout,
    ) {
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: dst.extent.width,
                height: dst.extent.height,
                depth: 1,
            });

        unsafe {
            Context::get().device.cmd_copy_buffer_to_image(
                *cmd,
                self.buffer,
                dst.image,
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    pub(crate) fn copy(self, cmd: &vk::CommandBuffer, dst_buffer: &Buffer) {
        let ctx = Context::get();
        unsafe {
            let region = vk::BufferCopy::default().size(self.size);
            ctx.device.cmd_copy_buffer(
                *cmd,
                self.buffer,
                dst_buffer.buffer,
                std::slice::from_ref(&region),
            )
        };
    }

    pub(crate) fn from_data_with_size<T: Copy>(
        usage: vk::BufferUsageFlags,
        data: &[T],
        size: u64,
    ) -> Result<Buffer> {
        let ctx = Context::get();
        let staging_buffer = Self::new(
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            size,
        )?;
        staging_buffer.copy_data_to_buffer(data)?;

        let buffer = Self::new(
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            size,
        )?;

        ctx.execute_one_time_commands(|cmd_buffer| {
            staging_buffer.copy(cmd_buffer, &buffer);
        })?;

        Ok(buffer)
    }

    pub(crate) fn from_data<T: Copy>(
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Result<Buffer> {
        let size = size_of_val(data) as _;
        Self::from_data_with_size(usage, data, size)
    }
}

impl ImageResource {
    pub(crate) fn new_2d(
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let ctx = Context::get();
        let image = Image::new_2d(usage, memory_location, format, width, height)?;
        let extend = image.extent.clone();
        Ok(image.new_resource(&ctx.device, extend))
    }

    pub fn new_from_data(
        image: DynamicImage,
        format: vk::Format,
    ) -> Result<Self> {
        let (width, height) = image.dimensions();
        let image_buffer = if format != vk::Format::R8G8B8A8_SRGB {
            let image_data = image.to_rgba32f();
            let image_data_raw = image_data.as_raw();

            let image_buffer = Buffer::new(
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                (size_of::<f32>() * image_data.len()) as u64,
            )?;
            image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
            image_buffer
        } else {
            let image_data = image.to_rgba8();
            let image_data_raw = image_data.as_raw();

            let image_buffer = Buffer::new(
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                (size_of::<u8>() * image_data.len()) as u64,
            )?;

            image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
            image_buffer
        };

        let texture_image = Image::new_2d(
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            format,
            width,
            height,
        )?;
        let ctx = Context::get();
        ctx.execute_one_time_commands(|cmd| {
            let barrier = texture_image.memory_barrier(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            unsafe {
                ctx.device.cmd_pipeline_barrier2(
                    *cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
                )
            };

            image_buffer.copy_to_image(
                cmd,
                &texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let barrier = texture_image.memory_barrier(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            unsafe {
                ctx.device.cmd_pipeline_barrier2(
                    *cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
                )
            };
        })?;
        let extend = texture_image.extent;
        Ok(texture_image.new_resource(&ctx.device, extend))
    }
}

#[derive(Default, PartialEq, Clone, Copy)]
pub struct SamplerInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl SamplerInfo {
    fn to_vk<'a>(&self) -> vk::SamplerCreateInfo<'a> {
        vk::SamplerCreateInfo {
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_mode: self.mipmap_mode,
            address_mode_u: self.address_mode_u,
            address_mode_v: self.address_mode_v,
            address_mode_w: self.address_mode_w,
            mip_lod_bias: self.mip_lod_bias,
            anisotropy_enable: if self.anisotropy_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            max_anisotropy: self.max_anisotropy,
            compare_enable: if self.compare_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            compare_op: self.compare_op,
            min_lod: self.min_lod,
            max_lod: self.max_lod,
            border_color: self.border_color,
            unnormalized_coordinates: if self.unnormalized_coordinates {
                vk::TRUE
            } else {
                vk::FALSE
            },
            ..Default::default()
        }
    }
}

pub fn create_sampler(info: &SamplerInfo) -> Result<vk::Sampler> {
    static mut SAMPLER_CACHE: MaybeUninit<Vec<(SamplerInfo, vk::Sampler)>> = MaybeUninit::uninit();
    static ONCE: Once = Once::new();
    unsafe {
        ONCE.call_once(|| {
            SAMPLER_CACHE.write(Vec::new());
        });
        let cache = SAMPLER_CACHE.assume_init_mut();
        
        match cache.iter().find(|e| e.0 == *info) {
            Some(sampler) => Ok(sampler.1),
            None => {
                let sampler = Context::get().device.create_sampler(&info.to_vk(), None)?;
                cache.push((*info, sampler));
                Ok(sampler)
            }
        }
    }
}

pub fn alinged_size(size: u32, alignment: u32) -> u32 {
    (size + (alignment - 1)) & !(alignment - 1)
}
