use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use ash::vk;

use crate::{
    backend::bindless::BindlessDescriptorHeap,
    vulkan::{swapchain::FRAMES_IN_FLIGHT, utils::Buffer, Context}, GConst,
};

use super::{
    Node, NodeHandle, RenderGraph, ResourceHandle, ResourceMemoryType, IMPORTED, SWPACHAIN,
};

impl RenderGraph {
    fn flatten<'a>(&'a self, root_node: NodeHandle) -> Vec<NodeHandle> {
        let mut flatt = vec![];
        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.graph.len()];

        visited[root_node] = true;
        queue.push_front(root_node);

        while !queue.is_empty() {
            let current_index = queue.pop_front().unwrap();
            let current = &self.graph[current_index];
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

    pub fn bake(&mut self) -> Result<()> {
        let root_node = self
            .graph
            .iter()
            .position(|n| n.writes.iter().find(|e| e.resource == SWPACHAIN).is_some())
            .expect("No writes to SWAPCHAIN found");

        let mut unschedueld_passes = self.flatten(root_node);
        self.execution_order.clear();
        self.execution_order.push(unschedueld_passes[0]);
        unschedueld_passes.remove(0);

        while !unschedueld_passes.is_empty() {
            let mut best_candidate = 0;
            let mut best_overlap_factor = 0;

            for i in 0..unschedueld_passes.len() {
                let mut overlap_factor = 0;
                for j in self.execution_order.iter().rev() {
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
            self.execution_order
                .push(unschedueld_passes[best_candidate]);
            unschedueld_passes.remove(best_candidate);
        }
        Ok(())
    }

    pub fn write_bindings(&mut self, swapchain_image_index: usize) -> Result<()> {
        for pass_index in &self.execution_order {
            let pass = &mut self.graph[*pass_index];
            let mut bindings = vec![];
            if let Some(buffer) = pass.constants_buffer {
                bindings.push(self.descriptor_handles[buffer.descriptor()][buffer.index()]);
            }
            pass
                .reads
                .iter()
                .for_each(|e| {
                    bindings.push(self.descriptor_handles[e.resource.descriptor()][e.resource.index()])
                });
            pass.writes.iter().for_each(|e| {
                if e.resource == SWPACHAIN {
                    bindings.push(self.swapchain_image_descriptors[swapchain_image_index]);
                }else {
                    let desc = &self.descriptor_handles[e.resource.descriptor()][e.resource.index()];
                    if !bindings.contains(&desc) {
                        bindings.push(*desc);
                    }
                }
            });

            if let Some(constants) = pass.constants {
                let buffer = &self.internal_buffers[pass.constants_buffer.unwrap().index()];
                println!("{}", unsafe { *(constants as *const i32) });
                unsafe {constants.copy_to(buffer.allocation.mapped_ptr().unwrap().as_ptr(), pass.constants_size) };
            }
            
            self.internal_buffers[pass.descriptor_buffer.index()].copy_data_to_buffer(&bindings)?;
            println!("{:?}", bindings)
        }
        Ok(())
    }

    //TODO Error Handeling
    pub fn draw(&mut self) {
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
        let swapchain_image_index = self.swapchain.next_image(frame_in_flight);
        BindlessDescriptorHeap::get().bind(&frame.cmd).unwrap();

        self.write_bindings(swapchain_image_index).unwrap();

        for (i, pass) in self.execution_order.iter().enumerate() {
            let pass = &self.graph[*pass];
            let mut image_barriers: HashMap<ResourceHandle, vk::ImageMemoryBarrier2> =
                HashMap::new();
            let mut buffer_barriers: HashMap<ResourceHandle, vk::BufferMemoryBarrier2> =
                HashMap::new();

            for read in &pass.reads {
                if read.output_of == IMPORTED {
                    continue;
                }
                if let Some(image) = self.image_handle(read.resource, swapchain_image_index) {
                    image_barriers.insert(
                        read.resource,
                        image.memory_barrier(
                            self.graph[read.output_of].writes[read.output_index]
                                .layout
                                .unwrap(),
                            read.layout.unwrap(),
                        ),
                    );
                }
                if let Some(buffer) = self.buffer_handle(read.resource) {
                    buffer_barriers.insert(
                        read.resource,
                        vk::BufferMemoryBarrier2::default()
                            .buffer(buffer.buffer)
                            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                            .src_stage_mask(read.get_parrent(&self).execution.get_stages())
                            .dst_stage_mask(pass.execution.get_stages())
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                            .offset(0)
                            .size(vk::WHOLE_SIZE),
                    );
                }
            }

            for writ in &pass.writes {
                if let Some(image) = self.image_handle(writ.resource, swapchain_image_index) {
                    image_barriers.insert(
                        writ.resource,
                        image.memory_barrier(vk::ImageLayout::UNDEFINED, writ.layout.unwrap()),
                    );
                }
                if let Some(buffer) = self.buffer_handle(writ.resource) {
                    buffer_barriers.insert(
                        writ.resource,
                        vk::BufferMemoryBarrier2::default()
                            .buffer(buffer.buffer)
                            .src_access_mask(
                                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                            )
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .dst_stage_mask(pass.execution.get_stages())
                            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                            .offset(0)
                            .size(vk::WHOLE_SIZE),
                    );
                }
            }

            let ctx = Context::get();
            unsafe {
                ctx.device.cmd_pipeline_barrier2(
                    frame.cmd,
                    &vk::DependencyInfo::default()
                        .dependency_flags(vk::DependencyFlags::BY_REGION)
                        .buffer_memory_barriers(
                            buffer_barriers.into_values().collect::<Vec<_>>().as_slice(),
                        )
                        .image_memory_barriers(
                            image_barriers.into_values().collect::<Vec<_>>().as_slice(),
                        ),
                )
            };

            unsafe {
                ctx.device.cmd_push_constants(
                    frame.cmd,
                    BindlessDescriptorHeap::get().layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &self.descriptor_handles[pass.descriptor_buffer.descriptor()]
                    [pass.descriptor_buffer.index()].to_bytes(),
                )
            };
            pass.execution.execute(&frame.cmd).unwrap();
            unsafe {
                let barrier = [self.swapchain.images[swapchain_image_index]
                    .memory_barrier(vk::ImageLayout::GENERAL, vk::ImageLayout::PRESENT_SRC_KHR)];
                let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barrier);
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
            }
        }

        unsafe { ctx.device.end_command_buffer(frame.cmd).unwrap() };

        let end_stage = self
            .graph
            .iter()
            .find(|n| n.writes.iter().find(|e| e.resource == SWPACHAIN).is_some())
            .expect("No writes to SWAPCHAIN found")
            .execution
            .get_stages();

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
            .present(frame_in_flight, swapchain_image_index);

        self.frame_number += 1;
    }
}
