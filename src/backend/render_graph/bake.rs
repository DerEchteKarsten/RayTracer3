use std::{
    collections::{HashMap, VecDeque},
    os::raw::c_void,
};

use anyhow::Result;
use ash::vk;

use crate::{
    backend::bindless::BindlessDescriptorHeap,
    vulkan::{image::ImageType, swapchain::FRAMES_IN_FLIGHT, Context},
};

use super::{
    ExecutionTrait, NodeHandle, RenderGraph,
};

impl RenderGraph {
    fn flatten<'a>(&'a self, root_node: NodeHandle) -> Vec<NodeHandle> {
        let mut flatt = vec![];
        let mut queue = VecDeque::new();
        let mut visited: HashMap<&'static str, bool> = HashMap::new();

        visited[root_node] = true;
        queue.push_front(root_node);

        while !queue.is_empty() {
            let current_index = queue.pop_front().unwrap();
            let current = &self.nodes[current_index];
            flatt.push(current_index);

            for c in current.parents() {
                if !visited[c] {
                    visited[c] = true;
                    queue.push_front(c);
                }
            }
        }
        flatt
    }

    pub fn bake(&mut self, root_node: NodeHandle) -> Result<Vec<NodeHandle>> {
        let mut unschedueld_passes = self.flatten(root_node);
        let mut execution_order = Vec::new();
        execution_order.push(unschedueld_passes[0]);
        unschedueld_passes.remove(0);

        while !unschedueld_passes.is_empty() {
            let mut best_candidate = 0;
            let mut best_overlap_factor = 0;

            for i in 0..unschedueld_passes.len() {
                let mut overlap_factor = 0;
                for j in execution_order.iter().rev() {
                    if self.graph[unschedueld_passes[i]].depends_on(self, &self.graph[*j]) {
                        break;
                    }
                    overlap_factor += 1;
                }
                if overlap_factor <= best_overlap_factor {
                    continue;
                }

                let mut possible_candidate = true;
                for j in 0..i {
                    if self.graph[unschedueld_passes[i]]
                        .depends_on(self, &self.graph[unschedueld_passes[j]])
                    {
                        possible_candidate = false;
                        break;
                    }
                }

                if possible_candidate {
                    best_candidate = i;
                    best_overlap_factor = overlap_factor;
                }
            }
            execution_order.push(unschedueld_passes[best_candidate]);
            unschedueld_passes.remove(best_candidate);
        }
        Ok(execution_order)
    }

    pub fn write_bindings(&mut self) -> Result<()> {
        for (pass_index, pass) in self.graph.iter().enumerate() {
            let descriptor_buffer_offset = &self.graph[0..pass_index]
                .iter()
                .fold(0, |acc, e| acc + e.bindings().count());

            if pass.bindings().count() != 0 {
                let bindings = pass
                    .bindings()
                    .map(|e| self.resources[e.resource.0 as usize].descriptor)
                    .collect::<Vec<_>>();

                unsafe {
                    (self.descriptor_buffer
                        .ty
                        .buffer()
                        .allocation
                        .as_ref()
                        .unwrap()
                        .mapped_ptr()
                        .unwrap()
                        .as_ptr()
                        .add(*descriptor_buffer_offset))
                    .copy_from(
                        bindings.as_ptr() as *const c_void,
                        bindings.len() * size_of::<u32>(),
                    )
                };
            }
        }

        Ok(())
    }
    pub fn begin_frame(&mut self) {
        self.constants.clear();
        self.graph.clear();
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

    //TODO Error Handeling
    pub fn draw_frame(&mut self, root_node: NodeHandle) {
        let execution_order = if self.graph.len() > 2 {
            log::error!("TODOOOOO");
            self.bake(root_node).unwrap()
        } else {
            (0..self.graph.len()).collect::<Vec<_>>()
        };
        let frame_in_flight = self.frame_number as usize % FRAMES_IN_FLIGHT;
        let frame = self.frame_data[frame_in_flight].clone();
        let ctx = Context::get();
        // println!("ImportedBuffer: {:#?}, ImportedImages: {:#?}, Buffers: {:#?}, Images: {:#?}", self.descriptor_handles[0], self.descriptor_handles[1], self.descriptor_handles[2], self.descriptor_handles[3]);

        BindlessDescriptorHeap::get().bind(&frame.cmd).unwrap();

        self.write_bindings().unwrap();

        for pass_index in execution_order.iter() {
            let pass = &self.graph[*pass_index];
            let mut image_barriers = Vec::new();
            let mut buffer_barriers = Vec::new();

            for edge in &pass.edges {
                if let Some(image) = self.image_handle(edge.resource) {
                    image_barriers.push(if let Some(origin) = edge.origin {
                        let last_edge = self.graph[origin]
                            .edges
                            .iter()
                            .find(|e| e.resource == edge.resource)
                            .unwrap();
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(self.graph[origin].execution.get_stages())
                            .dst_stage_mask(pass.execution.get_stages())
                            .src_access_mask(last_edge.edge_type.access_flags())
                            .dst_access_mask(edge.edge_type.access_flags())
                            .old_layout(last_edge.layout.unwrap())
                            .new_layout(edge.layout.unwrap())
                            .image(image.image)
                            .subresource_range(image.subresource_range())
                    } else {
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::empty())
                            .dst_stage_mask(pass.execution.get_stages())
                            .src_access_mask(vk::AccessFlags2::empty())
                            .dst_access_mask(edge.edge_type.access_flags())
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(edge.layout.unwrap())
                            .image(image.image)
                            .subresource_range(image.subresource_range())
                    });
                }
                if let Some(buffer) = self.buffer_handle(edge.resource) {
                    buffer_barriers.push(if let Some(origin) = edge.origin {
                        let last_edge = self.graph[origin]
                            .edges
                            .iter()
                            .find(|e| e.resource == edge.resource)
                            .unwrap();
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(self.graph[origin].execution.get_stages())
                            .dst_stage_mask(pass.execution.get_stages())
                            .src_access_mask(last_edge.edge_type.access_flags())
                            .dst_access_mask(edge.edge_type.access_flags())
                            .buffer(buffer.buffer)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)
                    } else {
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::empty())
                            .dst_stage_mask(pass.execution.get_stages())
                            .src_access_mask(vk::AccessFlags2::empty())
                            .dst_access_mask(edge.edge_type.access_flags())
                            .buffer(buffer.buffer)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)
                    });
                }
            }
            // println!("{:#?}", image_barriers);
            let ctx = Context::get();
            unsafe {
                ctx.device.cmd_pipeline_barrier2(
                    frame.cmd,
                    &vk::DependencyInfo::default()
                        .dependency_flags(vk::DependencyFlags::BY_REGION)
                        .buffer_memory_barriers(buffer_barriers.as_slice())
                        .image_memory_barriers(image_barriers.as_slice()),
                )
            };

            unsafe {
                let mut constants = [0u8; 16];
                constants[0..4].copy_from_slice(
                    &pass
                        .constant_offset
                        .unwrap_or(0)
                        .to_ne_bytes(),
                );

                constants[4..8].copy_from_slice(
                    &if pass.bindings().count() == 0 {
                        0
                    } else {
                        self.graph[..*pass_index]
                            .iter()
                            .fold(0, |acc, e| acc + e.bindings().count())
                            as u32
                    }
                    .to_ne_bytes(),
                );
                constants[8..12]
                    .copy_from_slice(&(self.descriptor_buffer.descriptor.index()).to_ne_bytes());
                constants[12..16]
                    .copy_from_slice(&(self.constants_buffer.descriptor.index()).to_ne_bytes());

                ctx.device.cmd_push_constants(
                    frame.cmd,
                    BindlessDescriptorHeap::get().layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &constants,
                )
            };
            pass.execution
                .execute(&frame.cmd, self, &self.graph[*pass_index].edges)
                .unwrap();
        }

        let mut last_edge = None;
        let last_node = self
            .graph
            .iter()
            .find(|n| {
                let res = n.edges.iter().find(|e| e.resource == self.get_swapchain());
                last_edge = res;
                res.is_some()
            })
            .expect("No writes to SWAPCHAIN found");
        let end_stage = last_node.execution.get_stages();
        let last_edge = last_edge.unwrap();

        unsafe {
            let swapchain_image = self.image_handle(self.get_swapchain()).unwrap();
            let barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(end_stage)
                .dst_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(last_edge.edge_type.access_flags())
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .old_layout(last_edge.layout.unwrap())
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(swapchain_image.image)
                .subresource_range(swapchain_image.subresource_range())];
            let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barrier);
            ctx.device
                .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
        }

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

        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.cmd)];

        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signal_semaphores)
            .wait_semaphore_infos(&wait_semaphores);
        unsafe {
            ctx.device
                .queue_submit2(ctx.graphics_queue, &[submit], vk::Fence::null())
                .unwrap()
        };
        self.swapchain
            .present(frame_in_flight, self.swapchain_image_index);

        self.frame_number += 1;
    }
}
