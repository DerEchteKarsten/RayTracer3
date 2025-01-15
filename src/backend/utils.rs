use std::{ffi::CStr, sync::Arc};

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
        ctx: &Context,
        cmd: &vk::CommandBuffer,
        other: &Image,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
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
        ctx: &Context,
        cmd: &vk::CommandBuffer,
        other: &Image,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
        unsafe {
            ctx.device.cmd_blit_image(
                *cmd,
                self.image,
                src_layout,
                other.image,
                dst_layout,
                &[vk::ImageBlit::default()
                    .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: self.extent.width as _, y: self.extent.height as _, z: 1 }])
                    .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: other.extent.width as _, y: other.extent.height as _, z: 1 }])
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
        ctx: &mut Context,
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
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
        ctx: &mut Context,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        alignment: Option<u64>,
    ) -> Result<Self> {
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
        ctx: &mut Context,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Self::new_aligned(ctx, usage, memory_location, size, None)
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

    pub(crate) fn destroy(self, ctx: &mut Context) -> Result<()> {
        unsafe { ctx.device.destroy_buffer(self.buffer, None) };
        ctx.allocator.free(self.allocation).unwrap();
        Ok(())
    }

    pub(crate) fn copy_to_image(
        &self,
        ctx: &Context,
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
            ctx.device.cmd_copy_buffer_to_image(
                *cmd,
                self.buffer,
                dst.image,
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    pub(crate) fn copy(self, ctx: &Context, cmd: &vk::CommandBuffer, dst_buffer: &Buffer) {
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
        ctx: &mut Context,
        usage: vk::BufferUsageFlags,
        data: &[T],
        size: u64,
    ) -> Result<Buffer> {
        let staging_buffer = Self::new(
            ctx,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            size,
        )?;
        staging_buffer.copy_data_to_buffer(data)?;

        let buffer = Self::new(
            ctx,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            size,
        )?;

        ctx.execute_one_time_commands(|cmd_buffer| {
            staging_buffer.copy(ctx, cmd_buffer, &buffer);
        })?;

        Ok(buffer)
    }

    pub(crate) fn from_data<T: Copy>(
        ctx: &mut Context,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Result<Buffer> {
        let size = size_of_val(data) as _;
        Self::from_data_with_size(ctx, usage, data, size)
    }
}

impl ImageResource {
    pub(crate) fn new_2d(
        ctx: &mut Context,
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let image = Image::new_2d(ctx, usage, memory_location, format, width, height)?;
        let extend = image.extent.clone();
        Ok(image.new_resource(&ctx.device, extend))
    }

    pub fn new_from_data(
        ctx: &mut Context,
        image: DynamicImage,
        format: vk::Format,
    ) -> Result<Self> {
        let (width, height) = image.dimensions();
        let image_buffer = if format != vk::Format::R8G8B8A8_SRGB {
            let image_data = image.to_rgba32f();
            let image_data_raw = image_data.as_raw();

            let image_buffer = Buffer::new(
                ctx,
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
                ctx,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                (size_of::<u8>() * image_data.len()) as u64,
            )?;

            image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
            image_buffer
        };

        let texture_image = Image::new_2d(
            ctx,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            format,
            width,
            height,
        )?;

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
                &ctx,
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

pub fn alinged_size(size: u32, alignment: u32) -> u32 {
    (size + (alignment - 1)) & !(alignment - 1)
}

pub fn read_shader_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut cursor = std::io::Cursor::new(bytes);
    Ok(ash::util::read_spv(&mut cursor)?)
}

pub fn module_from_bytes(device: &ash::Device, source: &[u8]) -> Result<vk::ShaderModule> {
    let source = read_shader_from_bytes(source)?;

    let create_info = vk::ShaderModuleCreateInfo::default().code(&source);
    let res = unsafe { device.create_shader_module(&create_info, None) }?;
    Ok(res)
}
