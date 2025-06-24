use std::future::IntoFuture;

use anyhow::{Ok, Result};
use bevy_app::Plugin;
use bevy_asset::{processor::LoadTransformAndSave, saver::AssetSaver, transformer::{AssetTransformer, TransformedAsset}, AssetApp, AssetLoader, AsyncReadExt, AsyncWriteExt, LoadContext};
use bevy_reflect::TypePath;
use bincode::{config::Configuration, de::read::Reader, enc::write::Writer};
use futures::executor::block_on;
use glam::vec3;
use meshopt;
use serde::Serialize;

use crate::renderer::bindless::DescriptorResourceHandle;


pub struct MeshAssets;
impl Plugin for MeshAssets {
    fn build(&self, app: &mut bevy_app::App) {
        app
            .register_asset_processor::<LoadTransformAndSave<GltfMeshLoader, MeshTransformer, MeshSaver>>(
                LoadTransformAndSave::new(MeshTransformer, MeshSaver),
            )
            .set_default_asset_processor::<LoadTransformAndSave<GltfMeshLoader, MeshTransformer, MeshSaver>>("glb")
            .register_asset_loader(GltfMeshLoader)
            .register_asset_loader(MeshLoader)
            .init_asset::<Mesh>()
            .init_asset::<GltfMesh>();
    }
}

#[repr(C)]
#[derive(bincode::Encode, bincode::Decode)]
pub struct Bounds {
    pub center: [f32; 3usize],
    pub radius: f32,
    pub cone_apex: [f32; 3usize],
    pub cone_axis: [f32; 3usize],
    pub cone_cutoff: f32,
    pub cone_axis_s8: [::std::os::raw::c_schar; 3usize],
    pub cone_cutoff_s8: ::std::os::raw::c_schar,
}

#[repr(C)]
#[derive(bincode::Encode, bincode::Decode)]
struct Meshlet {
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u32,
    triangle_count: u32,
}

#[repr(C)]
#[derive(Debug)]
struct Material {
    metalic_factor: f16,
    roughness_factor: f16,
    color: [f16; 3],
    texture_offset: u16,
}

impl bincode::Encode for Material {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> std::result::Result<(), bincode::error::EncodeError> {
        encoder.writer().write(&self.metalic_factor.to_be_bytes())?;
        encoder.writer().write(&self.roughness_factor.to_be_bytes())?;
        for i in 0..3 {
            encoder.writer().write(&self.color[i].to_be_bytes())?;
        }
        bincode::Encode::encode(&self.texture_offset, encoder)?;
        std::result::Result::Ok(())
    }
}

impl<Context> bincode::Decode<Context> for Material {
    fn decode<D: bincode::de::Decoder<Context = Context>>(decoder: &mut D) -> std::result::Result<Self, bincode::error::DecodeError> {
        let mut metalic_factor_buf = [0u8; 2];
        decoder.reader().read(&mut metalic_factor_buf);
        let mut roughness_buf = [0u8; 2];
        decoder.reader().read(&mut roughness_buf);
        let mut color = [0f16; 3];
        let mut color_buf = [0u8; 2];
        for i in 0..3 {
            decoder.reader().read(&mut color_buf);
            color[i] = f16::from_be_bytes(color_buf);
        }
        
        std::result::Result::Ok(Self {
            metalic_factor: f16::from_be_bytes(metalic_factor_buf),
            roughness_factor: f16::from_be_bytes(metalic_factor_buf),
            color,
            texture_offset: bincode::Decode::decode(decoder)?,
        })
    }
}

impl<'a, Context> bincode::BorrowDecode<'a, Context> for Material {
    fn borrow_decode<D: bincode::de::Decoder<Context = Context>>(decoder: &mut D) -> std::result::Result<Self, bincode::error::DecodeError> {
        let mut metalic_factor_buf = [0u8; 2];
        decoder.reader().read(&mut metalic_factor_buf);
        let mut roughness_buf = [0u8; 2];
        decoder.reader().read(&mut roughness_buf);
        let mut color = [0f16; 3];
        let mut color_buf = [0u8; 2];
        for i in 0..3 {
            decoder.reader().read(&mut color_buf);
            color[i] = f16::from_be_bytes(color_buf);
        }
        
        std::result::Result::Ok(Self {
            metalic_factor: f16::from_be_bytes(metalic_factor_buf),
            roughness_factor: f16::from_be_bytes(metalic_factor_buf),
            color,
            texture_offset: bincode::Decode::decode(decoder)?,
        })
    }
}


#[derive(bevy_asset::Asset, TypePath, bincode::Encode, bincode::Decode)]
pub struct Mesh {
    meshlets: Vec<Meshlet>,
    bounds: Vec<Bounds>,
    materials: Vec<Material>,
    vertices: Vec<Vertex>,
    indices: Vec<u8>,
}

#[repr(C)]
#[derive(bincode::Encode, bincode::Decode)]
pub struct Vertex {
    pub p: [f32; 3],
    pub n: [f32; 3],
    pub t: [f32; 2],
}

const CONFIG: Configuration<bincode::config::BigEndian> = bincode::config::standard()
            .with_variable_int_encoding()
            .with_big_endian();


struct MReader<'a>(&'a mut dyn bevy_asset::io::Reader);

impl<'a> Reader for MReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::result::Result<(), bincode::error::DecodeError> {
        match block_on(self.0.read(buf)) {
            std::result::Result::Ok(_) => std::result::Result::Ok(()),
            std::result::Result::Err(e) => std::result::Result::Err(bincode::error::DecodeError::Other("TODO: error handeling!!"))
        }
    }
}

struct MeshLoader;
impl AssetLoader for MeshLoader {
    type Asset = Mesh;
    type Error = anyhow::Error;
    type Settings = ();
    async fn load(
            &self,
            reader: &mut dyn bevy_asset::io::Reader,
            settings: &Self::Settings,
            load_context: &mut LoadContext<'_>,
        ) -> std::result::Result<Self::Asset, Self::Error> {
        let mesh = bincode::decode_from_reader(MReader(reader), CONFIG)?;
        Ok(mesh)
    }
    fn extensions(&self) -> &[&str] {
        &["mesh"]
    }
}


#[derive(bevy_asset::Asset, TypePath)]
pub struct GltfMesh {
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
}


pub struct GltfMeshLoader;
impl AssetLoader for GltfMeshLoader {
    type Asset = GltfMesh;
    type Error = gltf::Error;
    type Settings = ();

    async fn load(
            &self,
            reader: &mut dyn bevy_asset::io::Reader,
            settings: &(),
            load_context: &mut bevy_asset::LoadContext<'_>,
        ) -> gltf::Result<Self::Asset> {
        let mut file_buf = Vec::new();
        reader.read_to_end(&mut file_buf).await?;
        let (document, buffers, images) = gltf::import_slice(file_buf)?;
        gltf::Result::Ok(GltfMesh { document, buffers, images })
    }

    fn extensions(&self) -> &[&str] {
        &["glb"]
    }
}

struct MeshTransformer;
impl AssetTransformer for MeshTransformer {
    type AssetInput = GltfMesh;
    type AssetOutput = Mesh;
    type Error = anyhow::Error;
    type Settings = ();
    async fn transform<'a>(
            &'a self,
            asset: bevy_asset::transformer::TransformedAsset<Self::AssetInput>,
            settings: &'a Self::Settings,
        ) -> std::result::Result<bevy_asset::transformer::TransformedAsset<Self::AssetOutput>, Self::Error> {
        let mut mesh = Mesh {
            indices: Vec::new(),
            materials: Vec::new(),
            meshlets: Vec::new(),
            vertices: Vec::new(),
            bounds: Vec::new(),
        };

        let mut global_verticies: Vec<u32> = Vec::new();
        let mut materials = Vec::new();

        for i in asset.document.meshes() {
            for primitive in i.primitives() {
                let reader = primitive.reader(|buffer| Some(&asset.buffers[buffer.index()]));

                let normals = reader
                    .read_normals()
                    .unwrap()
                    .collect::<Vec<_>>();
    
                let uvs = reader
                    .read_tex_coords(0)
                    .map(|reader| reader.into_f32().collect::<Vec<_>>());
    
                reader.read_positions().unwrap().enumerate().for_each(|(index, p)| {
                    let n = normals[index];    
                    let t = uvs.as_ref().map_or([0.0, 0.0], |uvs| uvs[index]);
    
                    mesh.vertices.push(Vertex {
                        n,
                        p,
                        t
                    });
                });
    
                let index_reader = reader.read_indices().unwrap().into_u32();
                global_verticies.extend(index_reader.collect::<Vec<u32>>());
                let pmaterial = primitive.material();
                let pbr = pmaterial.pbr_metallic_roughness();
                materials.push(Material {
                    color: [pbr.base_color_factor()[0] as f16, pbr.base_color_factor()[1] as f16, pbr.base_color_factor()[2] as f16],
                    metalic_factor: pbr.metallic_factor() as f16,
                    roughness_factor: pbr.roughness_factor() as f16,
                    texture_offset: pbr.base_color_texture().map(|v| { v.texture().index() as u16 }).unwrap_or(!0u16),
                });
            }
        }
        log::info!("{:?}", materials);
        let asset = asset.replace_asset(mesh);
        return Ok(asset);
    }
}

struct MWriter<'a>(&'a mut bevy_asset::io::Writer);

impl<'a> Writer for MWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::result::Result<(), bincode::error::EncodeError> {
        match block_on(self.0.write(buf)) {
            std::result::Result::Ok(_) => std::result::Result::Ok(()),
            std::result::Result::Err(e) => std::result::Result::Err(bincode::error::EncodeError::Other("TODO: error handeling!!"))
        }
    }
}

struct MeshSaver;
impl AssetSaver for MeshSaver {
    type Asset = Mesh;
    type Error = anyhow::Error;
    type Settings = ();
    type OutputLoader = MeshLoader;
    async fn save(
            &self,
            writer: &mut bevy_asset::io::Writer,
            asset: bevy_asset::saver::SavedAsset<'_, Self::Asset>,
            settings: &Self::Settings,
        ) -> std::result::Result<<Self::OutputLoader as AssetLoader>::Settings, Self::Error> {
        bincode::encode_into_writer(asset.get(), MWriter(writer), CONFIG)?;
        return Ok(());
    }
}


