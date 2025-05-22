mod error;
mod image;
mod material;
mod texture;

use bevy_asset::Asset;
use bevy_reflect::TypePath;
pub use error::*;
pub use image::*;
pub use material::*;
pub use texture::*;

use std::{collections::HashMap, path::Path};

use glam::{vec4, Vec2, Vec4};
use gltf::{import_buffers, import_images, Document, Gltf, Primitive, Semantic};

#[derive(Debug, Clone)]
pub(crate) struct GltfModel {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub nodes: Vec<Node>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub samplers: Vec<Sampler>,
}

#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub transform: [[f32; 4]; 4],
    pub mesh: Mesh,
}

#[derive(Debug, Clone, Copy)]
pub struct Mesh {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub material: Material,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec4,
    pub color: Vec4,
    pub uvs: Vec2,
}

fn import_impl(
    Gltf { document, blob }: Gltf,
    base: Option<&Path>,
) -> gltf::Result<(Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>)> {
    let buffer_data = import_buffers(&document, base, blob)?;
    let image_data = import_images(&document, base, &buffer_data)?;
    let import = (document, buffer_data, image_data);
    Ok(import)
}

pub fn from_bytes(bytes: &[u8]) -> Result<GltfModel> {
    let gltf = Gltf::from_slice(bytes).map_err(|e| Error::Load(e.to_string()))?;
    let (document, buffers, gltf_images) =
        import_impl(gltf, None).map_err(|e| Error::Load(e.to_string()))?;

    let mut vertices = vec![];
    let mut indices = vec![];

    let mut meshes = vec![];
    let mut nodes = vec![];

    let mut mesh_index_redirect = HashMap::<(usize, usize), usize>::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives().filter(is_primitive_supported) {
            let og_index = (mesh.index(), primitive.index());

            if mesh_index_redirect.get(&og_index).is_none() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // vertices
                let vertex_reader = reader.read_positions().unwrap();
                let vertex_offset = vertices.len() as _;
                let vertex_count = vertex_reader.len() as _;

                let normals = reader
                    .read_normals()
                    .unwrap()
                    .map(|n| vec4(n[0], n[1], n[2], 0.0))
                    .collect::<Vec<_>>();

                let colors = reader
                    .read_colors(0)
                    .map(|reader| reader.into_rgba_f32().map(Vec4::from).collect::<Vec<_>>());

                let uvs = reader
                    .read_tex_coords(0)
                    .map(|reader| reader.into_f32().map(Vec2::from).collect::<Vec<_>>());

                vertex_reader.enumerate().for_each(|(index, p)| {
                    let position = vec4(p[0], p[1], p[2], 0.0);
                    let normal = normals[index];

                    let color = colors.as_ref().map_or(Vec4::ONE, |colors| colors[index]);
                    let uvs = uvs.as_ref().map_or(Vec2::ZERO, |uvs| uvs[index]);

                    vertices.push(Vertex {
                        position,
                        normal,
                        color,
                        uvs,
                    });
                });

                // indices
                let index_reader = reader.read_indices().unwrap().into_u32();
                let index_offset = indices.len() as _;
                let index_count = index_reader.len() as _;

                index_reader.for_each(|i| indices.push(i));
                // material
                let material = primitive.material().into();

                let mesh_index = meshes.len();

                mesh_index_redirect.insert(og_index, mesh_index);

                meshes.push(Mesh {
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    material,
                });
            }
        }
    }

    for node in document.nodes().filter(|n| n.mesh().is_some()) {
        let transform = node.transform().matrix();
        let gltf_mesh = node.mesh().unwrap();

        for primitive in gltf_mesh.primitives().filter(is_primitive_supported) {
            let og_index = (gltf_mesh.index(), primitive.index());
            let mesh_index = *mesh_index_redirect.get(&og_index).unwrap();
            let mesh = meshes[mesh_index];

            nodes.push(Node { transform, mesh })
        }
    }

    let images = gltf_images
        .iter()
        .map(Image::try_from)
        .collect::<Result<_>>()?;

    // Init samplers with a default one.
    // Textures with no sampler will reference this one.
    let mut samplers = vec![Sampler {
        mag_filter: MagFilter::Linear,
        min_filter: MinFilter::LinearMipmapLinear,
        wrap_s: WrapMode::Repeat,
        wrap_t: WrapMode::Repeat,
    }];
    document
        .samplers()
        .map(Sampler::from)
        .for_each(|s| samplers.push(s));

    let textures = document.textures().map(Texture::from).collect::<Vec<_>>();

    Ok(GltfModel {
        vertices,
        indices,
        nodes,
        images,
        textures,
        samplers,
    })
}

fn is_primitive_supported(primitive: &Primitive) -> bool {
    primitive.indices().is_some()
        && primitive.get(&Semantic::Positions).is_some()
        && primitive.get(&Semantic::Normals).is_some()
}
