use std::mem::MaybeUninit;

use anyhow::{Ok, Result};
use ash::vk;

use crate::raytracing::AccelerationStructure;

use super::{
    vulkan::raytracing::RayTracingContext,
    vulkan::utils::{Buffer, BufferHandle, Image, ImageHandle, ImageResource},
    vulkan::Context,
};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct DescriptorResourceHandle(u32);

impl DescriptorResourceHandle {
    fn bump_version_and_update_tag(self, tag: RenderResourceTag) -> Self {
        self
    }

    fn index(&self) -> u32 {
        self.0
    }
}

enum AccessType {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy)]
#[repr(u32)]
pub enum RenderResourceTag {
    Buffer,
    Image,
    Texture,
    AccelerationStructure,
}

impl DescriptorResourceHandle {
    pub fn new(
        _version: u8,
        _tag: RenderResourceTag,
        index: u32,
        _access_type: AccessType,
    ) -> Self {
        Self(index)
    }
}

pub struct BindlessDescriptorHeap {
    available_recycled_descriptors: Vec<DescriptorResourceHandle>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_index: u32,
    set_layouts: [vk::DescriptorSetLayout; 4],
    sets: [vk::DescriptorSet; 4],
    pub layout: vk::PipelineLayout,
}

#[derive(PartialEq)]
enum BindlessTableType {
    Buffers,
    Images,
    Textures,
    AccelerationStructures,
}

static mut BINDLESS: MaybeUninit<BindlessDescriptorHeap> = MaybeUninit::uninit();

impl BindlessTableType {
    fn all_tables() -> [BindlessTableType; 4] {
        [
            BindlessTableType::Buffers,
            BindlessTableType::Images,
            BindlessTableType::Textures,
            BindlessTableType::AccelerationStructures,
        ]
    }

    fn to_vk(&self) -> vk::DescriptorType {
        match self {
            BindlessTableType::AccelerationStructures => {
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
            }
            BindlessTableType::Buffers => vk::DescriptorType::STORAGE_BUFFER,
            BindlessTableType::Images => vk::DescriptorType::STORAGE_IMAGE,
            BindlessTableType::Textures => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }

    fn set_index(&self) -> usize {
        match self {
            BindlessTableType::Buffers => 0,
            BindlessTableType::Images => 1,
            BindlessTableType::Textures => 2,
            BindlessTableType::AccelerationStructures => 3,
        }
    }

    fn table_size(&self) -> u32 {
        let ctx = Context::get();
        let raytracing_ctx = RayTracingContext::get();

        match self {
            BindlessTableType::Buffers => {
                ctx.physical_device
                    .limits
                    .max_descriptor_set_storage_buffers
            }
            BindlessTableType::Images => {
                ctx.physical_device.limits.max_descriptor_set_storage_images
            }
            BindlessTableType::Textures => {
                ctx.physical_device.limits.max_descriptor_set_sampled_images
            }
            BindlessTableType::AccelerationStructures => {
                raytracing_ctx
                    .acceleration_structure_properties
                    .max_descriptor_set_acceleration_structures
            }
        }
    }

    pub fn descriptor_pool_sizes(immutable_sampler_count: u32) -> Vec<vk::DescriptorPoolSize> {
        let mut type_histogram = std::collections::HashMap::new();

        for table in Self::all_tables().iter() {
            type_histogram
                .entry(table.to_vk())
                .and_modify(|v| *v += table.table_size())
                .or_insert_with(|| table.table_size());
        }

        type_histogram
            .entry(Self::Textures.to_vk())
            .and_modify(|v| *v += immutable_sampler_count);

        type_histogram
            .iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *descriptor_count,
            })
            .collect::<Vec<vk::DescriptorPoolSize>>()
    }
}

impl BindlessDescriptorHeap {
    pub(crate) fn retire_handle(&mut self, handle: DescriptorResourceHandle) {
        self.available_recycled_descriptors.push(handle);
    }

    fn increment_descriptor(&mut self) -> u32 {
        let index = self.descriptor_index;
        self.descriptor_index += 1;
        index
    }

    fn fetch_available_descriptor(
        &mut self,
        tag: RenderResourceTag,
        access_type: AccessType,
    ) -> DescriptorResourceHandle {
        self.available_recycled_descriptors.pop().map_or_else(
            || DescriptorResourceHandle::new(0, tag, self.increment_descriptor(), access_type),
            |recycled_handle| recycled_handle.bump_version_and_update_tag(tag),
        )
    }

    pub fn allocate_buffer_handle(&mut self, buffer: &BufferHandle) -> DescriptorResourceHandle {
        let ctx = Context::get();
        let handle = Self::fetch_available_descriptor(
            self,
            RenderResourceTag::Buffer,
            AccessType::ReadWrite,
        );

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buffer.buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Buffers.set_index()],
            dst_binding: 0,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buffer_info,
            ..Default::default()
        }];
        unsafe {
            ctx.device.update_descriptor_sets(&write, &[]);
        };

        handle
    }

    pub fn allocate_storage_image_handle(
        &mut self,
        image: &ImageHandle,
    ) -> DescriptorResourceHandle {
        let ctx = Context::get();
        let handle =
            Self::fetch_available_descriptor(self, RenderResourceTag::Image, AccessType::ReadWrite);

        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
            image_view: image.view,
        };

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Images.set_index()],
            dst_binding: 0,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &image_info,
            ..Default::default()
        }];
        unsafe {
            ctx.device.update_descriptor_sets(&write, &[]);
        };

        handle
    }

    pub fn allocate_texture_handle(&mut self, image: &ImageHandle) -> DescriptorResourceHandle {
        let ctx = Context::get();
        let handle =
            Self::fetch_available_descriptor(self, RenderResourceTag::Image, AccessType::ReadWrite);

        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler: vk::Sampler::null(),
            image_view: image.view,
        };

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Textures.set_index()],
            dst_binding: 0,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            p_image_info: &image_info,
            ..Default::default()
        }];
        unsafe {
            ctx.device.update_descriptor_sets(&write, &[]);
        };

        handle
    }


    pub fn allocate_acceleration_structure_handle(&mut self, tlas: AccelerationStructure) -> DescriptorResourceHandle {
        let ctx = Context::get();
        let handle =
            Self::fetch_available_descriptor(self, RenderResourceTag::AccelerationStructure, AccessType::ReadWrite);

        let mut write_set_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas.accel));

        let mut write = vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .dst_binding(0)
            .dst_array_element(handle.index())
            .dst_set(self.sets[BindlessTableType::AccelerationStructures.set_index()])
            .push_next(&mut write_set_as);
        write.descriptor_count = 1;
        unsafe {
            ctx.device.update_descriptor_sets(&[write], &[]);
        };

        handle
    }

    pub(crate) fn init(immutable_samplers: &[vk::Sampler]) {
        unsafe { BINDLESS.write(Self::new(immutable_samplers)) };
    }

    pub(crate) fn get() -> &'static BindlessDescriptorHeap {
        unsafe { BINDLESS.assume_init_ref() }
    }

    pub(crate) fn get_mut() -> &'static mut BindlessDescriptorHeap {
        unsafe { BINDLESS.assume_init_mut() }
    }

    pub(crate) fn new(immutable_samplers: &[vk::Sampler]) -> Self {
        let ctx = Context::get();
        let descriptor_sizes =
            BindlessTableType::descriptor_pool_sizes(immutable_samplers.len() as u32);

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(4);

        let descriptor_pool = unsafe {
            ctx.device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };
        let (set_layouts, layout) = Self::create_bindless_layout(immutable_samplers);

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let sets = unsafe { ctx.device.allocate_descriptor_sets(&allocate_info).unwrap() }
            .try_into()
            .unwrap();

        Self {
            descriptor_pool,
            available_recycled_descriptors: vec![],
            descriptor_index: 0,
            set_layouts,
            sets,
            layout,
        }
    }

    pub(super) fn create_bindless_layout(
        immutable_samplers: &[vk::Sampler],
    ) -> ([vk::DescriptorSetLayout; 4], vk::PipelineLayout) {
        let ctx = Context::get();
        let mut descriptor_layouts = [vk::DescriptorSetLayout::null(); 4];
        BindlessTableType::all_tables()
            .iter()
            .enumerate()
            .for_each(|(set_idx, table)| unsafe {
                assert_eq!(table.set_index(), set_idx);

                let mut descriptor_binding_flags = vec![
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                ];

                let mut set = vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: table.to_vk(),
                    descriptor_count: table.table_size(),
                    stage_flags: vk::ShaderStageFlags::ALL,
                    p_immutable_samplers: std::ptr::null(),
                    ..Default::default()
                }];

                if *table == BindlessTableType::Textures {
                    descriptor_binding_flags.push(vk::DescriptorBindingFlags::empty());

                    // Set texture binding start at the end of the immutable samplers.
                    set[0].binding = immutable_samplers.len() as u32;
                    set.push(vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        descriptor_count: immutable_samplers.len() as u32,
                        stage_flags: vk::ShaderStageFlags::ALL,
                        p_immutable_samplers: immutable_samplers.as_ptr(),
                        ..Default::default()
                    });
                }

                let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
                    .binding_flags(&descriptor_binding_flags);

                descriptor_layouts[set_idx] = ctx
                    .device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default()
                            .bindings(&set)
                            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                            .push_next(&mut ext_flags),
                        None,
                    )
                    .unwrap();
            });

        let num_push_constants = 1;
        let num_push_constants_sized = std::mem::size_of::<u32>() as u32 * num_push_constants;

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: num_push_constants_sized,
        };

        let push_constant_ranges = [push_constant_range];

        let layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout =
            unsafe { ctx.device.create_pipeline_layout(&layout_create_info, None) }
                .expect("Failed creating pipeline layout.");

        (descriptor_layouts, pipeline_layout)
    }

    fn bind(&self, cmd: &vk::CommandBuffer) -> Result<()> {
        let ctx = Context::get();

        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &self.sets,
                &[],
            )
        };
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.layout,
                0,
                &self.sets,
                &[],
            )
        };
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &self.sets,
                &[],
            )
        };

        Ok(())
    }
}
