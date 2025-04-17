use std::{mem::MaybeUninit, sync::Once};

use anyhow::Result;
use ash::vk;

use crate::vulkan::Context;

#[derive(Default, PartialEq, Clone, Copy)]
pub struct SamplerInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl SamplerInfo {
    fn to_vk<'a>(&self) -> vk::SamplerCreateInfo<'a> {
        vk::SamplerCreateInfo {
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_mode: self.mipmap_mode,
            address_mode_u: self.address_mode_u,
            address_mode_v: self.address_mode_v,
            address_mode_w: self.address_mode_w,
            mip_lod_bias: self.mip_lod_bias,
            anisotropy_enable: if self.anisotropy_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            max_anisotropy: self.max_anisotropy,
            compare_enable: if self.compare_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            compare_op: self.compare_op,
            min_lod: self.min_lod,
            max_lod: self.max_lod,
            border_color: self.border_color,
            unnormalized_coordinates: if self.unnormalized_coordinates {
                vk::TRUE
            } else {
                vk::FALSE
            },
            ..Default::default()
        }
    }
}

pub fn create_sampler(info: &SamplerInfo) -> Result<vk::Sampler> {
    static mut SAMPLER_CACHE: MaybeUninit<Vec<(SamplerInfo, vk::Sampler)>> = MaybeUninit::uninit();
    static ONCE: Once = Once::new();
    unsafe {
        ONCE.call_once(|| {
            SAMPLER_CACHE.write(Vec::new());
        });
        let cache = SAMPLER_CACHE.assume_init_mut();

        match cache.iter().find(|e| e.0 == *info) {
            Some(sampler) => Ok(sampler.1),
            None => {
                let sampler = Context::get().device.create_sampler(&info.to_vk(), None)?;
                cache.push((*info, sampler));
                Ok(sampler)
            }
        }
    }
}

pub fn alinged_size(size: u32, alignment: u32) -> u32 {
    (size + (alignment - 1)) & !(alignment - 1)
}
