use anyhow::{Error, Result};
use ash::vk;
use derivative::Derivative;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc},
    MemoryLocation,
};
use image::{DynamicImage, GenericImageView};

use super::{
    image::{get_aspects, Image, ImageType},
    Context,
};

#[derive(Derivative)]
#[derivative(Eq, PartialEq)]
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
        cmd: &vk::CommandBuffer,
        dst: &impl ImageType,
        layout: vk::ImageLayout,
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
            });

        unsafe {
            Context::get().device.cmd_copy_buffer_to_image(
                *cmd,
                self.to_vk(),
                dst.get_image(),
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    fn copy(&self, cmd: &vk::CommandBuffer, dst_buffer: &impl BufferType) {
        let ctx = Context::get();
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
            allocation: Some(allocation),
            address: unsafe { ctx.device.get_buffer_device_address(&addr_info) },
            size,
            usage,
        })
    }

    pub fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Self::new_aligned(usage, memory_location, size, None)
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

    pub(crate) fn destroy(self) -> Result<()> {
        let ctx = Context::get_mut();
        unsafe { ctx.device.destroy_buffer(self.buffer, None) };
        ctx.allocator
            .free(self.allocation.ok_or(Error::msg("Buffer not Owned"))?)
            .unwrap();
        Ok(())
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

    pub(crate) fn from_data<T: Copy>(usage: vk::BufferUsageFlags, data: &[T]) -> Result<Buffer> {
        let size = size_of_val(data) as _;
        Self::from_data_with_size(usage, data, size)
    }
}
