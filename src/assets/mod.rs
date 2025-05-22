use bevy_app::{Plugin, Update};
use bevy_asset::{io::Reader, AssetApp, AssetEvent, AssetLoader, Assets, Handle, LoadContext};
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use gltf::GltfModel;
use model::RenderModel;

use crate::{
    raytracing::RayTracingContext, renderer::bindless::BindlessDescriptorHeap, vulkan::Context,
};

pub(crate) mod gltf;
pub(crate) mod model;

pub struct ModelPlugin;

fn create_models(
    mut ctx: ResMut<Context>,
    mut bindless: ResMut<BindlessDescriptorHeap>,
    mut raytracing_ctx: Option<ResMut<RayTracingContext>>,
    mut models: ResMut<Assets<Model>>,
    mut events: EventReader<AssetEvent<Model>>,
) {
    for i in events.read() {
        if let AssetEvent::LoadedWithDependencies { id } = i {
            if let Some(model) = models.get_mut(*id) {
                model.render_model = Some(
                    RenderModel::from_gltf(
                        &mut ctx,
                        &mut bindless,
                        raytracing_ctx.as_deref_mut(),
                        &model.gltf,
                    )
                    .unwrap(),
                );
                log::info!("Loaded")
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct ModelLoader;

impl AssetLoader for ModelLoader {
    type Asset = Model;
    type Settings = ();
    type Error = anyhow::Error;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let model = gltf::from_bytes(&bytes)?;
        Ok(Model {
            gltf: model,
            render_model: None,
        })
    }

    fn extensions(&self) -> &[&str] {
        &[]
    }
}

#[derive(bevy_asset::Asset, TypePath)]
pub(crate) struct Model {
    pub gltf: GltfModel,
    pub render_model: Option<RenderModel>,
}

impl Plugin for ModelPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_asset::<Model>()
            .init_asset_loader::<ModelLoader>()
            .add_systems(Update, create_models);
    }
}
