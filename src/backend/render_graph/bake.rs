use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::CStr,
    os::raw::c_void,
};

use anyhow::Result;
use ash::vk;

use crate::{
    backend::bindless::BindlessDescriptorHeap,
    vulkan::{image::ImageType, swapchain::FRAMES_IN_FLIGHT, Context},
};

use super::{
    depends_on, Barrier, Barriers, EdgeType, Event, ExecutionTrait, Node, NodeEdge, NodeHandle, RenderGraph, ResourceHandle
};

#[derive(Default, Debug)]
struct PhysicalBarriers<'a> {
    buffers: Vec<vk::BufferMemoryBarrier2<'a>>,
    images: Vec<vk::ImageMemoryBarrier2<'a>>,
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
                    .map(|e| self.resources[e.resource.0 as usize].descriptor)
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
    pub fn begin_frame(&mut self) {
        self.constants.clear();
        self.nodes.clear();
        self.resources.iter_mut().for_each(|resource| {
            resource.event.invalidated_in_stage = [vk::AccessFlags2::empty(); 25];
            resource.event.pipeline_barrier_src_stages = vk::PipelineStageFlags2::empty();
            resource.event.to_flush = vk::AccessFlags2::default();
            resource.event.layout = vk::ImageLayout::UNDEFINED;
        });

        let frame_in_flight = self.frame_number as usize % FRAMES_IN_FLIGHT;
        let frame = self.frame_data[frame_in_flight].clone();
        let ctx = Context::get();

        let semaphore = [self.frame_timeline_semaphore];
        let values = [frame.frame_number];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphore)
            .values(&values);
        unsafe { ctx.device.wait_semaphores(&wait_info, 1000000000).unwrap() };
        unsafe {
            ctx.device
                .reset_command_pool(frame.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap()
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            ctx.device
                .begin_command_buffer(frame.cmd, &begin_info)
                .unwrap()
        };

        self.swapchain_image_index = self.swapchain.next_image(frame_in_flight);
    }

    fn create_barriers<'a, 'b>(&'b mut self, execution_order: &Vec<NodeHandle>) -> Vec<PhysicalBarriers<'a>> 
    where 'a: 'b{
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
                }else {
                    self.nodes[*pass].get_barriers(self)
                };

                for barrier in pass_barriers.invalidates {
                    if let Some(buffer) = self.buffer_handle(barrier.resource) {
                        let event = &mut self.resources[barrier.resource.0 as usize].event;

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
                                size: vk::WHOLE_SIZE,
                                ..Default::default()
                            });
                        };
                    };
                    let mut layout_change = false;
                    if let Some(image) = self.image_handle(barrier.resource) {
                        layout_change = self.resources[barrier.resource.0 as usize].event.layout != barrier.layout;
                        let event = &mut self.resources[barrier.resource.0 as usize].event;
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
                            if event.pipeline_barrier_src_stages.as_raw() > 0
                            {
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

                    let event = &mut self.resources[barrier.resource.0 as usize].event;
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
                        self.resources[barrier.resource.0 as usize].event.layout = barrier.layout;
                    }
                    let event = &mut self.resources[barrier.resource.0 as usize].event;
                    event.to_flush = barrier.access;
                    event.pipeline_barrier_src_stages = barrier.stages;
                }

                barriers
            })
            .collect::<Vec<_>>()
    }

    //TODO Error Handeling
    pub fn draw_frame(&mut self, root_node: NodeHandle) {
        let frame_in_flight = self.frame_number as usize % FRAMES_IN_FLIGHT;
        let frame = self.frame_data[frame_in_flight].clone();
        let ctx = Context::get();
        // println!("ImportedBuffer: {:#?}, ImportedImages: {:#?}, Buffers: {:#?}, Images: {:#?}", self.descriptor_handles[0], self.descriptor_handles[1], self.descriptor_handles[2], self.descriptor_handles[3]);

        BindlessDescriptorHeap::get().bind(&frame.cmd).unwrap();

        let descriptor_offsets = self.write_bindings().unwrap();
        let execution_order = if self.nodes.len() > 2 {
            self.bake(root_node).unwrap()
        } else if self.nodes.len() == 1 {
            vec![0, 1]
        }else {
            vec![0]
        };

        let barriers = self.create_barriers(&execution_order);
        for (pass_index, pass_handle) in execution_order.iter().enumerate() {
            let pass = &self.nodes[*pass_handle];
            unsafe {
                pass.cmd_push_constants(self, &frame, descriptor_offsets[pass_index] as u32 * size_of::<u32>() as u32);

                let barrier = &barriers[pass_index];
                if barrier.images.len() != 0 || barrier.buffers.len() != 0 {
                    ctx.cmd_insert_label(&frame.cmd, &format!("Barrier for {}", pass.name));
                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(&barrier.buffers)
                        .image_memory_barriers(&barrier.images);
                    ctx.device
                        .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
                }

                ctx.cmd_start_label(&frame.cmd, &pass.name);
                pass.execution
                    .execute(&frame.cmd, self, &pass.edges)
                    .unwrap();
                ctx.cmd_end_label(&frame.cmd);
            }
        }

        if let Some(barrier) = barriers.get(execution_order.len()) {
            ctx.cmd_insert_label(&frame.cmd, "Transitioning Swapchain Image");
            let dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(&barrier.buffers)
                .image_memory_barriers(&barrier.images);
            unsafe {
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info)
            };
        }

        let end_stage = self.nodes[root_node].execution.get_stages();
        unsafe { ctx.device.end_command_buffer(frame.cmd).unwrap() };

        let wait_semaphores = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.swapchain.frame_resources[frame_in_flight].image_availible_semaphore)
            .stage_mask(end_stage)];

        let signal_frame_value = frame.frame_number + FRAMES_IN_FLIGHT as u64;
        self.frame_data[frame_in_flight].frame_number = signal_frame_value;

        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    self.swapchain.frame_resources[frame_in_flight as usize]
                        .render_finished_semaphore,
                )
                .stage_mask(end_stage),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.frame_timeline_semaphore)
                .stage_mask(end_stage)
                .value(signal_frame_value),
        ];

        ctx.submit(&frame.cmd, &wait_semaphores, &signal_semaphores);

        self.swapchain
            .present(frame_in_flight, self.swapchain_image_index);

        self.frame_number += 1;
    }
}
