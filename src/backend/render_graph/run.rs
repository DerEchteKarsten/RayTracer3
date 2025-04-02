use std::ffi::CStr;

use crate::backend::{raytracing::RayTracingContext, vulkan_context::Context};

use super::*;
use ash::khr;
use build::*;
use compile::*;

impl RenderGraph {
    fn begin_frame(&self, frame_in_flight: u64) -> u32 {
        let ctx = Context::get();
        let frame = &self.frame_data[frame_in_flight as usize];
        let semaphore = [self.static_resources.frame_timeline_semaphore];
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

        unsafe {
            self.static_resources
                .swapchain
                .ash_swapchain
                .acquire_next_image(
                    self.static_resources.swapchain.vk_swapchain,
                    1000000000,
                    self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                        .image_availible_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
        }
        .0
    }

    fn execute_passes(
        &self,
        frame_in_flight: u64,
        swapchain_image_index: u32,
    ) {
        let ctx = Context::get();
        let raytracing_ctx = RayTracingContext::get();
        let frame = &self.frame_data[frame_in_flight as usize];
        unsafe {
            for pass in &self.passes {
                if !pass.active {
                    continue;
                }
                let bind_point = match pass.command {
                    RenderPassCommand::Compute { .. }
                    | RenderPassCommand::ComputeIndirect { .. } => {
                        Some(vk::PipelineBindPoint::COMPUTE)
                    }
                    RenderPassCommand::Raytracing { .. } => {
                        Some(vk::PipelineBindPoint::RAY_TRACING_KHR)
                    }
                    RenderPassCommand::Raster { .. } => Some(vk::PipelineBindPoint::GRAPHICS),
                    RenderPassCommand::Custom { .. } => None,
                };

                if !pass.sync_resources.is_empty() {
                    let mut buffers = Vec::new();
                    let mut images = Vec::new();
                    for sync_resource in &pass.sync_resources {
                        let resource = &self.resources[&sync_resource.resource_key];
                        match &resource.get_current(self) {
                            ResourceData::Buffer(buffer) => {
                                buffers.push(
                                    vk::BufferMemoryBarrier2::default()
                                        .buffer(buffer.buffer)
                                        .size(buffer.size)
                                        .src_stage_mask(sync_resource.last_write)
                                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                                        .offset(0)
                                        .dst_stage_mask(match bind_point {
                                            Some(vk::PipelineBindPoint::COMPUTE) => {
                                                vk::PipelineStageFlags2::COMPUTE_SHADER
                                            }
                                            Some(vk::PipelineBindPoint::RAY_TRACING_KHR) => {
                                                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                                            }
                                            Some(vk::PipelineBindPoint::GRAPHICS) => {
                                                vk::PipelineStageFlags2::VERTEX_SHADER
                                            } //TODO
                                            _ => vk::PipelineStageFlags2::TOP_OF_PIPE,
                                        })
                                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                                );
                            }
                            ResourceData::Image(image) => {
                                let mut info = vk::ImageMemoryBarrier2::default()
                                    .image(image.image.image)
                                    .src_stage_mask(sync_resource.last_write)
                                    .subresource_range(
                                        vk::ImageSubresourceRange::default()
                                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                                            .base_array_layer(0)
                                            .base_mip_level(0)
                                            .layer_count(1)
                                            .level_count(1),
                                    )
                                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                                    .dst_stage_mask(match bind_point {
                                        Some(vk::PipelineBindPoint::COMPUTE) => {
                                            vk::PipelineStageFlags2::COMPUTE_SHADER
                                        }
                                        Some(vk::PipelineBindPoint::RAY_TRACING_KHR) => {
                                            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                                        }
                                        Some(vk::PipelineBindPoint::GRAPHICS) => {
                                            vk::PipelineStageFlags2::VERTEX_SHADER
                                        } //TODO
                                        _ => vk::PipelineStageFlags2::TOP_OF_PIPE,
                                    })
                                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

                                if let Some(old_layout) = sync_resource.old_layout
                                    && let Some(new_layout) = sync_resource.new_layout
                                {
                                    info = info.old_layout(old_layout).new_layout(new_layout);
                                }

                                images.push(info);
                            }
                        }
                    }

                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(&buffers)
                        .image_memory_barriers(&images)
                        .dependency_flags(vk::DependencyFlags::empty());

                    let lable_message = &format!(
                        "Barrier for {}\0",
                        pass.sync_resources
                            .iter()
                            .fold("".to_owned(), |acc, e| format!(
                                "{}, {}",
                                acc,
                                e.resource_key.as_str()
                            ))
                    );
                    let label = vk::DebugUtilsLabelEXT::default()
                        .label_name(CStr::from_bytes_with_nul(lable_message.as_bytes()).unwrap())
                        .color([1.0, 1.0, 1.0, 1.0]);
                    ctx.debug_utils
                        .cmd_insert_debug_utils_label(frame.cmd, &label);

                    ctx.device
                        .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
                }

                if let Some(bind_point) = bind_point {
                    let mut descriptor_sets = Vec::new();
                    if pass.bindless_descriptor {
                        descriptor_sets.push(self.static_descriptor_set);
                    }
                    if let Some(descriptor) = pass.descriptor_set {
                        descriptor_sets.push(descriptor);
                    }
                    if let Some(temporal_descriptors) = pass.temporal_descriptor_sets {
                        descriptor_sets.push(temporal_descriptors[self.frame_number as usize % 2]);
                    }
                    if let Some(temporal_descriptors2) = pass.temporal_descriptor_sets2 {
                        descriptor_sets.push(temporal_descriptors2[self.frame_number as usize % 2]);
                    }

                    if !descriptor_sets.is_empty() {
                        ctx.device.cmd_bind_descriptor_sets(
                            frame.cmd,
                            bind_point,
                            pass.layout,
                            0,
                            &descriptor_sets,
                            &[],
                        );
                    }
                    ctx.device
                        .cmd_bind_pipeline(frame.cmd, bind_point, pass.pipeline);
                }

                let label = vk::DebugUtilsLabelEXT::default()
                    .label_name(CStr::from_ptr(pass.name.as_ptr() as _))
                    .color([1.0, 1.0, 1.0, 1.0]);
                ctx.debug_utils
                    .cmd_insert_debug_utils_label(frame.cmd, &label);

                match &pass.command {
                    RenderPassCommand::Compute { x, y, z } => {
                        ctx.device.cmd_dispatch(frame.cmd,*x, *y, *z);
                    }
                    RenderPassCommand::Raster {color_attachments, depth_attachment, stencil_attachment, render_area, x, y, z} => {
                        let color_attachments = color_attachments.iter().map(|e| {
                            vk::RenderingAttachmentInfo::default()
                                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .clear_value(vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: e.clear.unwrap_or([0.0, 0.0, 0.0, 0.0]),
                                    },
                                })
                                .image_view(self.resources[&e.resource].get_current(self).get_image().unwrap().view)
                        }).collect::<Vec<_>>();
                        let mut rendering_info = vk::RenderingInfo::default()
                            .color_attachments(color_attachments.as_slice())
                            .layer_count(1)
                            .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                                width: render_area.x,
                                height: render_area.y,
                            }))
                            .view_mask(0);
                        
                        let render_info1; let render_info2;

                        if let Some(depth_attachment) = &depth_attachment {
                            render_info1 = depth_attachment.render_info(&self);
                            rendering_info = rendering_info.depth_attachment(&render_info1);
                        }
                        if let Some(stencil_attachment) = &stencil_attachment {
                            render_info2 = stencil_attachment.render_info(&self);
                            rendering_info = rendering_info.stencil_attachment(&render_info2);
                        }
                        
                        ctx.device.cmd_begin_rendering(frame.cmd, &rendering_info);
                        ctx.mesh_fn.cmd_draw_mesh_tasks(frame.cmd, *x, *y, *z);
                        ctx.device.cmd_end_rendering(frame.cmd);
                    }
                    RenderPassCommand::Raytracing {
                        x,
                        y,
                        shader_binding_table,
                    } => {
                        let call_region = vk::StridedDeviceAddressRegionKHR::default();
                        raytracing_ctx.pipeline_fn.cmd_trace_rays(
                            frame.cmd,
                            &shader_binding_table.raygen_region,
                            &shader_binding_table.miss_region,
                            &shader_binding_table.hit_region,
                            &call_region,
                            *x,
                            *y,
                            1,
                        );
                    }
                    RenderPassCommand::ComputeIndirect { indirect_buffer } => {
                        ctx.device
                            .cmd_dispatch_indirect(frame.cmd, *indirect_buffer, 0);
                    }
                    RenderPassCommand::Custom(func) => {
                        func(&frame.cmd, &pass);
                    }
                }
            }

            let copy_image = |image: &ImageResource| {
                let image_barriers = [
                    self.static_resources.swapchain.images[swapchain_image_index as usize]
                        .image
                        .memory_barrier(
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        ),
                    image.image.memory_barrier(
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ),
                ];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info);

                image.image.blit(
                    &frame.cmd,
                    &self.static_resources.swapchain.images[swapchain_image_index as usize].image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                let image_barriers = [
                    self.static_resources.swapchain.images[swapchain_image_index as usize]
                        .image
                        .memory_barrier(
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::PRESENT_SRC_KHR,
                        ),
                    image.image.memory_barrier(
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                    ),
                ];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
                ctx.device
                    .cmd_pipeline_barrier2(frame.cmd, &dependency_info);
            };

            if let Some(entry) = self.resources.get(&self.back_buffer) {
                match &entry.handle {
                    ResourceTemporal::Single(res) => match res {
                        ResourceData::Buffer(_) => {
                            panic!("Back Buffer is Buffer")
                        }
                        ResourceData::Image(image) => copy_image(image),
                    },
                    ResourceTemporal::Temporal(res) => match &res[(self.frame_number % 2) as usize]
                    {
                        ResourceData::Buffer(_) => {
                            panic!("Back Buffer is Buffer")
                        }
                        ResourceData::Image(image) => copy_image(image),
                    },
                }
            }
        }
    }

    fn submit_frame(&mut self, frame_in_flight: u64, swapchain_image_index: u32) {
        let ctx = Context::get();
        let frame = &mut self.frame_data[frame_in_flight as usize];

        unsafe { ctx.device.end_command_buffer(frame.cmd).unwrap() };

        let wait_semaphores = [vk::SemaphoreSubmitInfo::default()
            .semaphore(
                self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                    .image_availible_semaphore,
            )
            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)];

        let signal_frame_value = frame.frame_number + FRAMES_IN_FLIGHT;
        frame.frame_number = signal_frame_value;

        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                        .render_finished_semaphore,
                )
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.static_resources.frame_timeline_semaphore)
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
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

        let binding = [
            self.static_resources.swapchain.frame_resources[frame_in_flight as usize]
                .render_finished_semaphore,
        ];
        let swapchains = [self.static_resources.swapchain.vk_swapchain];
        let image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .image_indices(&image_indices)
            .swapchains(&swapchains)
            .wait_semaphores(&binding);
        unsafe {
            self.static_resources
                .swapchain
                .ash_swapchain
                .queue_present(ctx.present_queue, &present_info)
                .unwrap()
        };
        self.frame_number += 1;
    }

    fn render_imgui(
        &mut self,
        draw_data: &imgui::DrawData,
        cmd: vk::CommandBuffer,
        swapchain_image_index: u32,
    ) {
        let ctx = Context::get();
        unsafe {
            let begin_info = vk::RenderPassBeginInfo::default()
                .framebuffer(
                    self.static_resources.imgui_resources.frame_buffers
                        [swapchain_image_index as usize],
                )
                .render_pass(self.static_resources.imgui_resources.renderpass)
                .render_area(
                    vk::Rect2D::default()
                        .extent(vk::Extent2D {
                            width: WINDOW_SIZE.x as u32,
                            height: WINDOW_SIZE.y as u32,
                        })
                        .offset(vk::Offset2D { x: 0, y: 0 }),
                );
            ctx.device
                .cmd_begin_render_pass(cmd, &begin_info, vk::SubpassContents::INLINE);
            self.static_resources
                .imgui_resources
                .renderer
                .cmd_draw(cmd, draw_data)
                .unwrap();
            ctx.device.cmd_end_render_pass(cmd);
        }
    }

    //TODO Error Handeling
    pub fn draw(
        &mut self,
        draw_data: &imgui::DrawData,
    ) {
        let frame_in_flight = self.frame_number % FRAMES_IN_FLIGHT;
        let swapchain_image_index = self.begin_frame(frame_in_flight);

        self.execute_passes(frame_in_flight, swapchain_image_index);
        self.render_imgui(
            draw_data,
            self.frame_data[frame_in_flight as usize].cmd,
            swapchain_image_index,
        );
        self.submit_frame(frame_in_flight, swapchain_image_index);
    }

    pub fn toggle_pass(&mut self, pass: &str, value: bool) {
        self.passes
            .iter_mut()
            .find(|e| e.name == pass)
            .unwrap()
            .active = value;
        self.build_sync_resources();
    }
}
