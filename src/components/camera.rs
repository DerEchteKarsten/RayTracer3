use std::time::Duration;

use bevy_app::{App, PreUpdate, Update};
use bevy_ecs::resource::Resource;
use bevy_input::{
    keyboard::KeyCode,
    mouse::{MouseButton, MouseButtonInput, MouseMotion},
    ButtonInput, ButtonState,
};
use bevy_time::Time;
use bevy_window::{CursorGrabMode, PrimaryWindow, Window};
use glam::{vec3, Mat3, Mat4, Quat, Vec3};

use bevy_ecs::prelude::*;

use crate::WINDOW_SIZE;

const MOVE_SPEED: f32 = 10.0;
const ANGLE_PER_POINT: f32 = 0.5;

const UP: Vec3 = vec3(0.0, 1.0, 0.0);

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        fov: f32,
        aspect_ratio: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            fov,
            aspect_ratio,
            z_near,
            z_far,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.direction, UP)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect_ratio, self.z_near, self.z_far)
    }
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct Controls {
    pub go_forward: bool,
    pub go_backward: bool,
    pub strafe_right: bool,
    pub strafe_left: bool,
    pub go_up: bool,
    pub go_down: bool,
    pub look_around: bool,
    pub cursor_delta: [f32; 2],
    pub cursor_position: [f64; 2],
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            go_forward: false,
            go_backward: false,
            strafe_right: false,
            strafe_left: false,
            go_up: false,
            go_down: false,
            look_around: false,
            cursor_delta: [0.0; 2],
            cursor_position: [0.0; 2],
        }
    }
}

pub fn update_mouse_buttons(
    mut controls: ResMut<Controls>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut mousebtn_evr: EventReader<MouseButtonInput>,
) {
    let mut window = windows.single_mut().unwrap();
    for ev in mousebtn_evr.read() {
        if ev.button == MouseButton::Right && ev.state == ButtonState::Pressed {
            controls.look_around = true;
            window.cursor_options.grab_mode = CursorGrabMode::Confined;
            window.cursor_options.visible = false;
        }
        if ev.button == MouseButton::Right && ev.state == ButtonState::Released {
            controls.look_around = false;
            window.cursor_options.grab_mode = CursorGrabMode::None;
            window.cursor_options.visible = true;
        }
    }
}
pub fn update_mouse_move(mut controls: ResMut<Controls>, mut evr_motion: EventReader<MouseMotion>) {
    for ev in evr_motion.read() {
        controls.cursor_delta = [
            controls.cursor_delta[0] + ev.delta.x,
            controls.cursor_delta[1] + ev.delta.y,
        ];
    }
}

pub fn update_keyboard(keys: Res<ButtonInput<KeyCode>>, mut controls: ResMut<Controls>) {
    controls.go_forward = keys.pressed(KeyCode::KeyW);
    controls.go_backward = keys.pressed(KeyCode::KeyS);
    controls.strafe_right = keys.pressed(KeyCode::KeyD);
    controls.strafe_left = keys.pressed(KeyCode::KeyA);
    controls.go_up = keys.pressed(KeyCode::Space);
    controls.go_down = keys.pressed(KeyCode::ShiftLeft);
}

pub fn editor_camera(mut query: Query<&mut Camera>, controls: Res<Controls>, time: Res<Time>) {
    let delta_time = time.delta_secs();
    for mut camera in &mut query {
        let side = camera.direction.cross(UP);

        // Update direction
        let new_direction = if controls.look_around {
            let side_rot = Quat::from_axis_angle(
                side,
                controls.cursor_delta[1] * ANGLE_PER_POINT * delta_time,
            );
            let y_rot =
                Quat::from_rotation_y(-controls.cursor_delta[0] * ANGLE_PER_POINT * delta_time);
            let rot = Mat3::from_quat(side_rot * y_rot);

            (rot * camera.direction).normalize()
        } else {
            camera.direction
        };

        // Update position
        let mut direction = Vec3::ZERO;

        if controls.go_forward {
            direction += new_direction;
        }
        if controls.go_backward {
            direction -= new_direction;
        }
        if controls.strafe_right {
            direction += side;
        }
        if controls.strafe_left {
            direction -= side;
        }
        if controls.go_up {
            direction -= UP;
        }
        if controls.go_down {
            direction += UP;
        }

        let direction = if direction.length_squared() == 0.0 {
            direction
        } else {
            direction.normalize()
        };

        camera.position += direction * MOVE_SPEED * delta_time;
        camera.direction = new_direction;
    }
}

fn reset(mut controls: ResMut<Controls>) {
    controls.cursor_delta = [0.0; 2];
}

pub fn CameraPlugin(app: &mut App) {
    app.init_resource::<Controls>()
        .add_systems(
            PreUpdate,
            (update_mouse_move, update_mouse_buttons, update_keyboard),
        )
        .add_systems(Update, (editor_camera, reset).chain());
}
