use std::collections::HashMap;

use ash::vk::{self, Format};

use crate::{
    renderer::bindless::{BindlessDescriptorHeap, DescriptorResourceHandle},
    vulkan::{
        buffer::{Buffer, BufferHandle},
        image::{Image, ImageHandle},
        Context,
    },
};

use super::build::ImageSize;

pub(crate) type ResourceHandle = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Event {
    // event: vk::Event,
    pub(super) pipeline_barrier_src_stages: vk::PipelineStageFlags2,
    pub(super) to_flush: vk::AccessFlags2,
    pub(super) invalidated_in_stage: [vk::AccessFlags2; 25],
    pub(super) layout: vk::ImageLayout,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ResourceDescription {
    pub(super) name: &'static str,
    pub(super) handle: ResourceHandle,
    pub(super) ty: ResourceDescriptionType,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub(super) enum ResourceDescriptionType {
    Image {
        size: ImageSize,
        usage: vk::ImageUsageFlags,
        format: Format,
    },
    Buffer {
        size: u64,
        usage: vk::BufferUsageFlags,
    },
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Resource {
    pub(super) event: Event,
    pub(super) descriptor: DescriptorResourceHandle,
    pub(super) ty: ResourceType,
}

impl Resource {
    pub(super) fn new(descriptor: DescriptorResourceHandle, ty: ResourceType) -> Self {
        Self {
            descriptor,
            ty,
            event: Event {
                invalidated_in_stage: [vk::AccessFlags2::empty(); 25],
                pipeline_barrier_src_stages: vk::PipelineStageFlags2::empty(),
                to_flush: vk::AccessFlags2::default(),
                layout: vk::ImageLayout::UNDEFINED,
            },
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub(super) enum ResourceType {
    Image(Image),
    Buffer(Buffer),
    ExternalDescriptor,
    Uninitilized(usize),
}

impl ResourceType {
    pub(super) fn buffer(&self) -> &Buffer {
        match self {
            Self::Buffer(buffer) => buffer,
            _ => unreachable!(),
        }
    }
    pub(super) fn image(&self) -> &Image {
        match self {
            Self::Image(image) => image,
            _ => unreachable!(),
        }
    }
}

pub(super) trait Importable {
    fn resource(self, ctx: &Context, bindless: &mut BindlessDescriptorHeap) -> Resource;
}

impl Importable for Buffer {
    fn resource(self, ctx: &Context, bindless: &mut BindlessDescriptorHeap) -> Resource {
        let descriptor = bindless.allocate_buffer_handle(ctx, &self);
        Resource::new(descriptor, ResourceType::Buffer(self))
    }
}
impl Importable for BufferHandle {
    fn resource(self, ctx: &Context, bindless: &mut BindlessDescriptorHeap) -> Resource {
        let descriptor = bindless.allocate_buffer_handle(ctx, &self);
        Resource::new(
            descriptor,
            ResourceType::Buffer(Buffer {
                address: self.address,
                allocation: None,
                buffer: self.buffer,
                size: self.size,
                usage: self.usage,
            }),
        )
    }
}
impl Importable for Image {
    fn resource(self, ctx: &Context, bindless: &mut BindlessDescriptorHeap) -> Resource {
        let descriptor = if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            bindless.allocate_image_handle(ctx, &self)
        } else {
            bindless.allocate_texture_handle(ctx, &self)
        };
        Resource::new(descriptor, ResourceType::Image(self))
    }
}
impl Importable for ImageHandle {
    fn resource(self, ctx: &Context, bindless: &mut BindlessDescriptorHeap) -> Resource {
        let descriptor = if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            bindless.allocate_image_handle(ctx, &self)
        } else {
            bindless.allocate_texture_handle(ctx, &self)
        };
        Resource::new(
            descriptor,
            ResourceType::Image(Image {
                allocation: None,
                extent: self.extent,
                format: self.format,
                image: self.image,
                usage: self.usage,
                view: self.view,
            }),
        )
    }
}
impl Importable for DescriptorResourceHandle {
    fn resource(self, ctx: &Context, bindless: &mut BindlessDescriptorHeap) -> Resource {
        Resource::new(self, ResourceType::ExternalDescriptor)
    }
}
