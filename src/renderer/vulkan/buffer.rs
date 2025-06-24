use anyhow::{Error, Result};
use ash::vk;
use derivative::Derivative;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc},
    MemoryLocation,
};
use image::{DynamicImage, GenericImageView};

use crate::renderer::bindless::{BindlessDescriptorHeap, DescriptorResourceHandle};

use super::{
    image::{get_aspects, Image, ImageType},
    Context,
};

#[derive(Derivative)]
#[derivative(Eq, PartialEq, Debug)]
pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    #[derivative(PartialEq = "ignore")]
    pub(crate) allocation: Option<Allocation>,
    pub(crate) address: vk::DeviceAddress,
    pub(crate) size: vk::DeviceSize,
    pub(crate) usage: vk::BufferUsageFlags,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            usage: self.usage,
            address: self.address,
            size: self.size,
            allocation: None,
        }
    }
}

impl Buffer {
    pub(crate) fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.buffer,
            address: self.address,
            size: self.size,
            usage: self.usage,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct BufferHandle {
    pub(crate) buffer: vk::Buffer,
    pub(crate) address: vk::DeviceAddress,
    pub(crate) size: vk::DeviceSize,
    pub(crate) usage: vk::BufferUsageFlags,
}

impl BufferType for Buffer {
    fn get_address(&self) -> vk::DeviceAddress {
        self.address
    }
    fn get_size(&self) -> vk::DeviceSize {
        self.size
    }
    fn get_usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer
    }
}
impl BufferType for BufferHandle {
    fn get_address(&self) -> vk::DeviceAddress {
        self.address
    }
    fn get_size(&self) -> vk::DeviceSize {
        self.size
    }
    fn get_usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer
    }
}

pub(crate) trait BufferType {
    fn get_address(&self) -> vk::DeviceAddress;
    fn get_size(&self) -> vk::DeviceSize;
    fn to_vk(&self) -> vk::Buffer;
    fn get_usage(&self) -> vk::BufferUsageFlags;
    fn copy_to_image(
        &self,
        ctx: &Context,
        cmd: &vk::CommandBuffer,
        dst: &impl ImageType,
        layout: vk::ImageLayout,
        buffer_offset: u64,
    ) {
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: get_aspects(dst.get_format()),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: dst.get_extent().width,
                height: dst.get_extent().height,
                depth: 1,
            })
            .buffer_offset(buffer_offset);

        unsafe {
            ctx.device.cmd_copy_buffer_to_image(
                *cmd,
                self.to_vk(),
                dst.get_image(),
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    fn copy(&self, ctx: &Context, cmd: &vk::CommandBuffer, dst_buffer: &impl BufferType) {
        unsafe {
            let region = vk::BufferCopy::default().size(self.get_size());
            ctx.device.cmd_copy_buffer(
                *cmd,
                self.to_vk(),
                dst_buffer.to_vk(),
                std::slice::from_ref(&region),
            )
        };
    }
}

impl Buffer {
    pub fn new_aligned(
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
            allocation: Some(allocation),
            address: unsafe { ctx.device.get_buffer_device_address(&addr_info) },
            size,
            usage,
        })
    }

    pub fn new(
        ctx: &mut Context,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Self::new_aligned(ctx, usage, memory_location, size, None)
    }

    pub fn copy_data_to_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        self.copy_data_to_aligned_buffer(data, align_of::<T>() as _)
    }

    pub fn copy_data_to_aligned_buffer<T: Copy>(&self, data: &[T], alignment: u32) -> Result<()> {
        unsafe {
            let data_ptr = self
                .allocation
                .as_ref()
                .ok_or(Error::msg("Buffer not Owned"))?
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            let mut align = ash::util::Align::new(data_ptr, alignment as _, size_of_val(data) as _);
            align.copy_from_slice(data);
        };

        Ok(())
    }

    pub(crate) fn destroy(&mut self, ctx: &mut Context) -> Result<()> {
        unsafe { ctx.device.destroy_buffer(self.buffer, None) };
        ctx.allocator
            .free(self.allocation.take().ok_or(Error::msg("Buffer not Owned"))?)
            .unwrap();
        Ok(())
    }

    // pub(crate) fn from_data_with_size<T: Copy>(
    //     ctx: &mut Context,
    //     usage: vk::BufferUsageFlags,
    //     data: &[T],
    //     size: u64,
    // ) -> Result<Buffer> {
    //     let staging_buffer = Self::new(
    //         ctx,
    //         vk::BufferUsageFlags::TRANSFER_SRC,
    //         MemoryLocation::CpuToGpu,
    //         size,
    //     )?;
    //     staging_buffer.copy_data_to_buffer(data)?;

    //     let buffer = Self::new(
    //         ctx,
    //         usage | vk::BufferUsageFlags::TRANSFER_DST,
    //         MemoryLocation::GpuOnly,
    //         size,
    //     )?;

    //     ctx.execute_one_time_commands(|cmd_buffer| {
    //         staging_buffer.copy(&ctx, cmd_buffer, &buffer);
    //     })?;

    //     staging_buffer.destroy(ctx);

    //     Ok(buffer)
    // }

    // pub(crate) fn from_data<T: Copy>(
    //     ctx: &mut Context,
    //     usage: vk::BufferUsageFlags,
    //     data: &[T],
    // ) -> Result<Buffer> {
    //     let size = size_of_val(data) as _;
    //     Self::from_data_with_size(ctx, usage, data, size)
    // }
}

#[derive(Default)]
struct RawDynamicBuffer {
    buffer: vk::Buffer,
    address: vk::DeviceAddress,
    allocation: Option<Allocation>,
}

pub(crate) struct DynamicBuffer {
    buffer: RawDynamicBuffer,
    capacity: u64,
    usage: vk::BufferUsageFlags,
    size: u64,
    memory_location: MemoryLocation,
    override_alignment: Option<u64>,
    pub bindless_handle: DescriptorResourceHandle,
}

impl BufferType for DynamicBuffer {
    fn get_address(&self) -> vk::DeviceAddress {
        self.buffer.address
    }
    fn get_size(&self) -> vk::DeviceSize {
        self.size
    }
    fn get_usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer.buffer
    }
}

impl DynamicBuffer {
    pub(crate) fn new(ctx: &mut Context, bindless: &mut BindlessDescriptorHeap, usage: vk::BufferUsageFlags, memory_location: MemoryLocation, capacity: u64, override_alignment: Option<u64>) -> Result<Self> {
        let mut s = Self {
            memory_location,
            buffer: RawDynamicBuffer::default(),
            capacity,
            size: 0,
            usage: usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            override_alignment,
            bindless_handle: DescriptorResourceHandle(0),
        };
        let buffer = s.create_buffer(ctx)?;
        s.buffer = buffer;
        s.bindless_handle = bindless.allocate_buffer_handle(ctx, &s);
        Ok(s)
    }

    fn create_buffer(&mut self, ctx: &mut Context) -> Result<RawDynamicBuffer> {
        let create_info = vk::BufferCreateInfo::default()
            .size(self.capacity)
            .usage(self.usage);
        let buffer = unsafe { ctx.device.create_buffer(&create_info, None)? };
        let mut requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if let Some(a) = self.override_alignment {
            requirements.alignment = a;
        }

        let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
            name: "buffer",
            requirements,
            location: self.memory_location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { ctx.device.get_buffer_device_address(&addr_info) };
        Ok(RawDynamicBuffer{
            address,
            allocation: Some(allocation), 
            buffer,
        })
    }

    pub fn grow_to_size(&mut self, size: u64, ctx: &mut Context, bindless: &mut BindlessDescriptorHeap) -> Result<()> {
        let old_size = self.size;
        self.size = size;
        if size > self.capacity {
            self.capacity *= 1 << (64 - (self.capacity / self.size).leading_zeros() - 1);

            let buffer = self.create_buffer(ctx)?;
            ctx.execute_one_time_commands(|cmd, ctx | {
                unsafe { 
                    ctx.device.cmd_copy_buffer(*cmd, self.buffer.buffer, buffer.buffer, &[vk::BufferCopy{
                    size: old_size,
                    src_offset: 0,
                    dst_offset: 0,
                }]) };
            });
            unsafe { ctx.device.destroy_buffer(self.buffer.buffer, None) };
            ctx.allocator
                .free(self.buffer.allocation.take().ok_or(Error::msg("Buffer not Owned"))?)
                .unwrap();
            self.buffer = buffer;
            bindless.update_buffer_handle(ctx, self, self.bindless_handle);
        }
        Ok(())
    }

    pub fn cmd_grow_to_size(&mut self, size: u64, ctx: &mut Context, bindless: &mut BindlessDescriptorHeap, cmd: &vk::CommandBuffer) -> Result<()> {
        let old_size = self.size;
        self.size = size;
        if size > self.capacity {
            self.capacity *= 1 << (64 - (self.capacity / self.size).leading_zeros() - 1);

            let buffer = self.create_buffer(ctx)?;
            unsafe { 
                ctx.device.cmd_copy_buffer(*cmd, self.buffer.buffer, buffer.buffer, &[vk::BufferCopy{
                size: old_size,
                src_offset: 0,
                dst_offset: 0,
            }]) };
            unsafe { ctx.device.destroy_buffer(self.buffer.buffer, None) };
            ctx.allocator
                .free(self.buffer.allocation.take().ok_or(Error::msg("Buffer not Owned"))?)
                .unwrap();
            self.buffer = buffer;
            bindless.update_buffer_handle(ctx, self, self.bindless_handle);
        }
        Ok(())
    }

    pub(crate) fn copy_from(&mut self, ctx: &mut Context, bindless: &mut BindlessDescriptorHeap, src_buffer: &impl BufferType, offset: u64) -> Result<()>{
        if self.size < src_buffer.get_size() + offset {
            self.grow_to_size(src_buffer.get_size() + offset, ctx, bindless);
        }
        ctx.execute_one_time_commands(|cmd, ctx| {
            unsafe { ctx.device.cmd_copy_buffer(*cmd, src_buffer.to_vk(), self.buffer.buffer, &[vk::BufferCopy {
                            src_offset: 0,
                            size: src_buffer.get_size(),
                            dst_offset: offset,
                        }]) };
        })
    }
}
