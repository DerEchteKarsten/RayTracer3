use std::ffi::CStr;

use anyhow::Result;
use ash::{
    ext::{extended_dynamic_state3, mesh_shader},
    khr::{
        acceleration_structure, deferred_host_operations, get_memory_requirements2,
        ray_tracing_pipeline, shader_float_controls, shader_non_semantic_info, synchronization2,
    },
    vk,
};

use super::physical_device::{PhysicalDevice, QueueFamily};

pub const DEVICE_EXTENSIONS: [&'static CStr; 13] = [
    ash::khr::swapchain::NAME,
    ray_tracing_pipeline::NAME,
    acceleration_structure::NAME,
    ash::ext::descriptor_indexing::NAME,
    ash::ext::scalar_block_layout::NAME,
    get_memory_requirements2::NAME,
    deferred_host_operations::NAME,
    vk::KHR_SPIRV_1_4_NAME,
    shader_float_controls::NAME,
    shader_non_semantic_info::NAME,
    synchronization2::NAME,
    mesh_shader::NAME,
    extended_dynamic_state3::NAME,
];

pub(super) fn create_device(
    instance: &ash::Instance,
    physical_device: &PhysicalDevice,
    queue_families: &[&QueueFamily; 3],
    required_extensions: &[&str],
) -> Result<ash::Device> {
    let queue_priorities = [1.0f32];

    let queue_create_infos = {
        let mut indices = queue_families.iter().map(|f| f.index).collect::<Vec<_>>();
        indices.dedup();

        indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>()
    };
    // let mut atomics = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
    //     .shader_buffer_float64_atomic_add(true);
    let mut ray_tracing_feature =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let mut acceleration_struct_feature =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true)
            .descriptor_binding_acceleration_structure_update_after_bind(true);
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        .runtime_descriptor_array(true)
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .shader_float16(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_variable_descriptor_count(true)
        .descriptor_binding_storage_image_update_after_bind(true)
        .descriptor_binding_sampled_image_update_after_bind(true);
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .maintenance4(true)
        .synchronization2(true);

    let features = vk::PhysicalDeviceFeatures::default()
        .shader_int64(true)
        .fragment_stores_and_atomics(true)
        .shader_int16(true)
        .vertex_pipeline_stores_and_atomics(true);

    let mut mesh_shading = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
        .task_shader(true)
        .mesh_shader(true);
    let mut dynamic_state = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
        .extended_dynamic_state3_depth_clamp_enable(true)
        .extended_dynamic_state3_polygon_mode(true)
        .extended_dynamic_state3_logic_op_enable(true)
        .extended_dynamic_state3_color_blend_equation(true)
        .extended_dynamic_state3_color_write_mask(true)
        .extended_dynamic_state3_color_blend_enable(true);
    let mut dynamic_state2 = vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default()
        .extended_dynamic_state2_logic_op(true);

    let mut features = vk::PhysicalDeviceFeatures2::default()
        .features(features)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut vulkan_13_features)
        .push_next(&mut ray_tracing_feature)
        .push_next(&mut acceleration_struct_feature)
        .push_next(&mut mesh_shading)
        .push_next(&mut dynamic_state)
        .push_next(&mut dynamic_state2);
    // .push_next(&mut atomics);

    let device_extensions_as_ptr = required_extensions
        .into_iter()
        .map(|e| e.as_ptr() as *const i8)
        .collect::<Vec<_>>();

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(device_extensions_as_ptr.as_slice())
        .push_next(&mut features);

    let device = unsafe {
        instance
            .create_device(physical_device.handel, &device_create_info, None)
            .unwrap()
    };

    Ok(device)
}
