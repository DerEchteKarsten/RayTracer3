use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use ash::vk;

use crate::vulkan::{swapchain::FRAMES_IN_FLIGHT, Context};

use super::{Node, NodeHandle, RenderGraph, ResourceMemoryHandle};

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

    fn bake(&mut self) -> Result<()> {
        let (root_node, _) = self.output.ok_or(Error::msg("No Output set".to_string()))?;

        let mut unschedueld_passes = self.flatten(root_node);
        let mut flattend_passes = Vec::with_capacity(unschedueld_passes.len());

        flattend_passes.push(unschedueld_passes[0]);
        unschedueld_passes.remove(0);

        while !unschedueld_passes.is_empty() {
            let mut best_candidate = 0;
            let mut best_overlap_factor = 0;

            for i in 0..unschedueld_passes.len() {
                let mut overlap_factor = 0;
                for j in flattend_passes.iter().rev() {
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
            flattend_passes.push(unschedueld_passes[best_candidate]);
            unschedueld_passes.remove(best_candidate);
        }
        for pass in flattend_passes {
            let pass = &self.graph[pass];
            
        }
        self.execution_order = flattend_passes;
        Ok(())
    }

    //TODO Error Handeling
    pub fn draw(&mut self) {
        let frame_in_flight = self.frame_number as usize % FRAMES_IN_FLIGHT;
        let swapchain_image_index = self.swapchain.next_image(frame_in_flight);

        self.frame_data[frame_in_flight]
            .clone()
            .record_command_buffer(
                self.frame_timeline_semaphore,
                &self.swapchain,
                frame_in_flight,
                |cmd| {
                    for (i, pass) in self.execution_order.iter().enumerate() {
                        let pass = &self.graph[*pass];
                        let mut image_barriers: HashMap<
                            ResourceMemoryHandle,
                            vk::ImageMemoryBarrier2,
                        > = HashMap::new();
                        let mut buffer_barriers: HashMap<
                            ResourceMemoryHandle,
                            vk::BufferMemoryBarrier2,
                        > = HashMap::new();

                        for read in &pass.reads {
                            let memory = self.get_handle(&read.description);
                            if let Some(image) = self.image_handle(memory.clone()) {
                                image_barriers.insert(
                                    memory.clone(),
                                    image.memory_barrier(
                                        self.graph[read.output_of].writes[read.output_index]
                                            .description
                                            .get_layout()
                                            .unwrap(),
                                        read.description.get_layout().unwrap(),
                                    ),
                                );
                            }
                            if let Some(buffer) = self.buffer_handle(memory.clone()) {
                                buffer_barriers.insert(
                                    memory,
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.buffer)
                                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                                        .src_stage_mask(
                                            read.get_parrent(&self).execution.get_stages(),
                                        )
                                        .dst_stage_mask(pass.execution.get_stages())
                                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                                        .offset(0)
                                        .size(vk::WHOLE_SIZE),
                                );
                            }
                        }

                        for writ in &pass.writes {
                            let memory = self.get_handle(&writ.description);
                            if let Some(image) = self.image_handle(memory.clone()) {
                                image_barriers.insert(
                                    memory.clone(),
                                    image.memory_barrier(
                                        vk::ImageLayout::UNDEFINED,
                                        writ.description.get_layout().unwrap(),
                                    ),
                                );
                            }
                            if let Some(buffer) = self.buffer_handle(memory.clone()) {
                                buffer_barriers.insert(
                                    memory,
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.buffer)
                                        .src_access_mask(
                                            vk::AccessFlags2::SHADER_READ
                                                | vk::AccessFlags2::SHADER_WRITE,
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
                                cmd,
                                &vk::DependencyInfo::default()
                                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                                    .buffer_memory_barriers(buffer_barriers.into_values().collect::<Vec<_>>().as_slice())
                                    .image_memory_barriers(image_barriers.into_values().collect::<Vec<_>>().as_slice())
                            )
                        };

                        pass.execution.execute(&cmd);
                    }
                },
            );

        self.swapchain
            .present(frame_in_flight, swapchain_image_index);
        self.frame_number += 1;
    }
}
