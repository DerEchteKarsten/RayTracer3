use anyhow::Result;
use ash::{khr, vk, Device, Instance};

use crate::WINDOW_SIZE;

use super::{
    render_graph::FRAMES_IN_FLIGHT,
    utils::{Image, ImageResource},
    vulkan_context::Context,
};

pub struct FrameResources {
    pub image_availible_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
}

pub struct Swapchain {
    pub ash_swapchain: khr::swapchain::Device,
    pub vk_swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<ImageResource>,
    pub frame_resources: [FrameResources; FRAMES_IN_FLIGHT as usize],
}

impl Swapchain {
    pub fn new(ctx: &Context) -> Result<Self> {
        let format = {
            let formats = unsafe {
                ctx.surface.ash.get_physical_device_surface_formats(
                    ctx.physical_device.handel,
                    ctx.surface.vulkan,
                )?
            };
            if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_UNORM,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                }
            } else {
                *formats
                    .iter()
                    .find(|format| {
                        format.format == vk::Format::B8G8R8A8_UNORM
                            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .unwrap_or(&formats[0])
            }
        };

        let present_mode = {
            let present_modes = unsafe {
                ctx.surface.ash.get_physical_device_surface_present_modes(
                    ctx.physical_device.handel,
                    ctx.surface.vulkan,
                )?
            };
            if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            }
        };

        let capabilities = unsafe {
            ctx.surface.ash.get_physical_device_surface_capabilities(
                ctx.physical_device.handel,
                ctx.surface.vulkan,
            )?
        };

        let extent = {
            if capabilities.current_extent.width != std::u32::MAX {
                capabilities.current_extent
            } else {
                let min = capabilities.min_image_extent;
                let max = capabilities.max_image_extent;
                let width = (WINDOW_SIZE.x as u32).min(max.width).max(min.width);
                let height = (WINDOW_SIZE.y as u32).min(max.height).max(min.height);
                vk::Extent2D { width, height }
            }
        };

        let image_count = capabilities.min_image_count + 1;

        let families_indices = [
            ctx.graphics_queue_family.index,
            ctx.present_queue_family.index,
        ];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(ctx.surface.vulkan)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                );

            builder = if ctx.graphics_queue_family.index != ctx.present_queue_family.index {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let ash_swapchain = khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let vk_swapchain = unsafe { ash_swapchain.create_swapchain(&create_info, None).unwrap() };

        let images = unsafe { ash_swapchain.get_swapchain_images(vk_swapchain).unwrap() };

        let images = images
            .into_iter()
            .map(|i| {
                Image {
                    image: i,
                    format: format.format,
                    extent,
                    allocation: None,
                }
                .new_resource(&ctx.device, extent)
            })
            .collect::<Vec<_>>();

        let mut frame_resources: [FrameResources; FRAMES_IN_FLIGHT as usize] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..FRAMES_IN_FLIGHT as usize {
            let create_info = vk::SemaphoreCreateInfo::default();
            let image_availible_semaphore =
                unsafe { ctx.device.create_semaphore(&create_info, None).unwrap() };
            let render_finished_semaphore =
                unsafe { ctx.device.create_semaphore(&create_info, None).unwrap() };
            frame_resources[i] = FrameResources {
                image_availible_semaphore,
                render_finished_semaphore,
            }
        }

        Ok(Self {
            ash_swapchain,
            vk_swapchain,
            format: format.format,
            color_space: format.color_space,
            present_mode,
            images,
            frame_resources,
        })
    }
}
