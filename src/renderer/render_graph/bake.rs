use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::CStr,
    os::raw::c_void,
};

use anyhow::Result;
use ash::vk;
use bevy_ecs::{system::ResMut, world::World};

use crate::{
    raytracing::RayTracingContext,
    renderer::bindless::BindlessDescriptorHeap,
    vulkan::{buffer::BufferType, image::ImageType, swapchain::FRAMES_IN_FLIGHT, Context},
};

use super::{
    depends_on, Barrier, Barriers, EdgeType, Event, ExecutionTrait, Node, NodeEdge, NodeHandle,
    RenderGraph, ResourceHandle,
};

#[derive(Default, Debug)]
pub(super) struct PhysicalBarriers<'a> {
    pub(super) buffers: Vec<vk::BufferMemoryBarrier2<'a>>,
    pub(super) images: Vec<vk::ImageMemoryBarrier2<'a>>,
}

impl RenderGraph {
    fn flatten<'a>(&'a self, root_node: NodeHandle, execution_order: &mut Vec<NodeHandle>) {
        execution_order.push(root_node);
        for e in &self.nodes[root_node].edges {
            if let Some(origin) = e.origin {
                self.flatten(origin, execution_order);
            }
        }
    }

    pub fn bake<'a>(&'a self, root_node: NodeHandle) -> Result<Vec<NodeHandle>> {
        let mut execution_order = Vec::new();
        self.flatten(root_node, &mut execution_order);
        execution_order.reverse();
        let mut eo_deduped = Vec::new();
        for i in execution_order {
            if !eo_deduped.contains(&i) {
                eo_deduped.push(i);
            }
        }
        Ok(eo_deduped)
    }

    pub fn write_bindings(&mut self) -> Result<Vec<usize>> {
        let mut descriptor_buffer_offset = 0;
        let mut offsets = Vec::new();
        for pass in self.nodes.iter() {
            if pass.bindings().count() != 0 {
                let bindings = pass
                    .bindings()
                    .map(|e| self.resources[e.resource].descriptor.0)
                    .collect::<Vec<_>>();
                unsafe {
                    (self
                        .descriptor_buffer
                        .ty
                        .buffer()
                        .allocation
                        .as_ref()
                        .unwrap()
                        .mapped_ptr()
                        .unwrap()
                        .as_ptr()
                        .add(descriptor_buffer_offset * size_of::<u32>()))
                    .copy_from(
                        bindings.as_ptr() as *const c_void,
                        bindings.len() * size_of::<u32>(),
                    )
                };
            }
            offsets.push(descriptor_buffer_offset);
            descriptor_buffer_offset += pass.bindings().count();
        }

        Ok(offsets)
    }

    pub(super) fn create_barriers<'a, 'b>(
        &'b mut self,
        execution_order: &Vec<NodeHandle>,
    ) -> Vec<PhysicalBarriers<'a>>
    where
        'a: 'b,
    {
        execution_order
            .iter()
            .chain(vec![&!0])
            .map(|pass| {
                let mut need_pipeline_barrier = false;
                let mut barriers = PhysicalBarriers::default();

                let pass_barriers = if *pass == !0 {
                    Barriers {
                        invalidates: vec![Barrier {
                            access: vk::AccessFlags2::MEMORY_READ,
                            layout: vk::ImageLayout::PRESENT_SRC_KHR,
                            resource: self.swapchain_images[self.swapchain_image_index],
                            stages: vk::PipelineStageFlags2::TOP_OF_PIPE,
                        }],
                        flushes: Vec::new(),
                    }
                } else {
                    self.nodes[*pass].get_barriers(self)
                };

                for barrier in pass_barriers.invalidates {
                    if let Some(buffer) = self.buffer_handle(barrier.resource) {
                        let event = &mut self.resources[barrier.resource].event;

                        if (!event.to_flush.is_empty()) || barrier.need_invalidate(event) {
                            need_pipeline_barrier = !event.pipeline_barrier_src_stages.is_empty();
                        }

                        if need_pipeline_barrier {
                            barriers.buffers.push(vk::BufferMemoryBarrier2 {
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                src_access_mask: event.to_flush,
                                dst_access_mask: barrier.access,
                                src_stage_mask: event.pipeline_barrier_src_stages,
                                dst_stage_mask: barrier.stages,
                                buffer: buffer.buffer,
                                offset: 0,
                                size: buffer.get_size(),
                                ..Default::default()
                            });
                        };
                    };
                    let mut layout_change = false;
                    if let Some(image) = self.image_handle(barrier.resource) {
                        layout_change =
                            self.resources[barrier.resource].event.layout != barrier.layout;
                        let event = &mut self.resources[barrier.resource].event;
                        let mut image_barrier = vk::ImageMemoryBarrier2 {
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            old_layout: event.layout,
                            new_layout: barrier.layout,
                            src_access_mask: event.to_flush,
                            dst_access_mask: barrier.access,
                            dst_stage_mask: barrier.stages,
                            image: image.image,
                            subresource_range: image.subresource_range(),
                            ..Default::default()
                        };

                        if layout_change
                            || (!event.to_flush.is_empty())
                            || barrier.need_invalidate(&event)
                        {
                            if event.pipeline_barrier_src_stages.as_raw() > 0 {
                                image_barrier.src_stage_mask = event.pipeline_barrier_src_stages;
                                need_pipeline_barrier = true;
                            } else {
                                image_barrier.src_stage_mask = vk::PipelineStageFlags2::NONE;
                                image_barrier.src_access_mask = vk::AccessFlags2::NONE;
                            }
                            barriers.images.push(image_barrier);
                        };
                        event.layout = barrier.layout;
                    };

                    let event = &mut self.resources[barrier.resource].event;
                    if event.to_flush.as_raw() > 0 || layout_change {
                        for e in &mut event.invalidated_in_stage {
                            *e = vk::AccessFlags2::empty();
                        }
                    }
                    event.to_flush = vk::AccessFlags2::empty();
                    if need_pipeline_barrier && event.pipeline_barrier_src_stages.as_raw() != 0 {
                        for i in 0..64 {
                            event.invalidated_in_stage
                                [((barrier.stages.as_raw() >> i) & 1) as usize / 2] |=
                                barrier.access;
                        }
                    }
                }
                for barrier in pass_barriers.flushes.iter() {
                    if self.image_handle(barrier.resource).is_some() {
                        self.resources[barrier.resource].event.layout = barrier.layout;
                    }
                    let event = &mut self.resources[barrier.resource].event;
                    event.to_flush = barrier.access;
                    event.pipeline_barrier_src_stages = barrier.stages;
                }

                barriers
            })
            .collect::<Vec<_>>()
    }
}
