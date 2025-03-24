use std::time::Duration;

use glam::{vec3, Mat3, Mat4, Quat, Vec3};
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

use crate::{PlanarViewConstants, WINDOW_SIZE};

const MOVE_SPEED: f32 = 30.0;
const ANGLE_PER_POINT: f32 = 1.0;

const UP: Vec3 = vec3(0.0, -1.0, 0.0);

#[derive(Debug, Clone, Copy, PartialEq)]
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

    pub fn update(self, controls: &Controls, delta_time: Duration) -> Self {
        let delta_time = delta_time.as_secs_f32();
        let side = self.direction.cross(UP);

        // Update direction
        let new_direction = if controls.look_around {
            let side_rot = Quat::from_axis_angle(
                side,
                -controls.cursor_delta[1] * ANGLE_PER_POINT * delta_time,
            );
            let y_rot =
                Quat::from_rotation_y(-controls.cursor_delta[0] * ANGLE_PER_POINT * delta_time);
            let rot = Mat3::from_quat(side_rot * y_rot);

            (rot * self.direction).normalize()
        } else {
            self.direction
        };

        // Update position
        let mut direction = Vec3::ZERO;

        if controls.go_forward {
            direction -= new_direction;
        }
        if controls.go_backward {
            direction += new_direction;
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

        Self {
            position: self.position + direction * MOVE_SPEED * delta_time,
            direction: new_direction,
            ..self
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.direction, UP)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        perspective(
            self.fov.to_radians(),
            self.aspect_ratio,
            self.z_near,
            self.z_far,
        )
    }
    pub fn planar_view_constants(&self) -> PlanarViewConstants {
        let window_size = glam::vec2(WINDOW_SIZE.x as f32, WINDOW_SIZE.y as f32);
        let clip_to_window_scale = glam::vec2(0.5 * window_size.x, -0.5 * window_size.y);
        let clip_to_window_bias = window_size * 0.5;
        let window_to_clip_scale = 1.0 / clip_to_window_scale;

        PlanarViewConstants {
            mat_world_to_view: self.view_matrix(),
            mat_view_to_clip: self.projection_matrix(),
            mat_world_to_clip: self.projection_matrix() * self.view_matrix(),
            mat_clip_to_view: self.projection_matrix().inverse(),
            mat_view_to_world: self.view_matrix().inverse(),
            mat_clip_to_world: self.projection_matrix().inverse() * self.view_matrix().inverse(),
            viewport_origin: glam::vec2(0.0, 0.0),
            viewport_size: window_size,
            viewport_size_inv: 1.0 / window_size,

            clip_to_window_scale,
            clip_to_window_bias,

            window_to_clip_scale,
            window_to_clip_bias: -clip_to_window_bias * window_to_clip_scale,

            camera_direction_or_position: glam::vec4(
                self.position.x,
                self.position.y,
                self.position.z,
                1.0,
            ),
            pixel_offset: glam::vec2(0.0, 0.0),
        }
    }
}

#[rustfmt::skip]
pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    
    let y_scale = 1.0 / f32::tan(0.5 * fovy);
    let x_scale = y_scale / aspect;
    let z_scale = 1.0 / (far - near);
    return Mat4::from_cols_array(&[
                x_scale, 0.0, 0.0, 0.0,
                0.0, y_scale, 0.0, 0.0,
                0.0, 0.0, -(near + far) * z_scale, 1.0,
                0.0, 0.0, -2.0 * near * far * z_scale, 0.0
            ]);
}

#[derive(Debug, Clone, Copy)]
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

impl Controls {
    pub fn reset(self) -> Self {
        Self {
            cursor_delta: [0.0; 2],
            ..self
        }
    }

    pub fn handle_event(self, event: &Event<()>, window: &winit::window::Window) -> Self {
        let mut new_state = self;
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (x, y) },
                ..
            } => {
                let x = *x as f32;
                let y = *y as f32;
                new_state.cursor_delta = [x, y];
            }
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.physical_key == PhysicalKey::Code(KeyCode::KeyW) {
                            new_state.go_forward = event.state == ElementState::Pressed;
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::KeyS) {
                            new_state.go_backward = event.state == ElementState::Pressed;
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::KeyD) {
                            new_state.strafe_right = event.state == ElementState::Pressed;
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::KeyA) {
                            new_state.strafe_left = event.state == ElementState::Pressed;
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::Space) {
                            new_state.go_up = event.state == ElementState::Pressed;
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::ShiftLeft) {
                            new_state.go_down = event.state == ElementState::Pressed;
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::ArrowLeft)
                            && event.state == ElementState::Pressed
                        {
                            new_state.look_around = true;
                            new_state.cursor_delta = [-1.0, 0.0];
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::ArrowRight)
                            && event.state == ElementState::Pressed
                        {
                            new_state.look_around = true;
                            new_state.cursor_delta = [1.0, 0.0];
                        }

                        if event.physical_key == PhysicalKey::Code(KeyCode::ArrowUp)
                            && event.state == ElementState::Pressed
                        {
                            new_state.look_around = true;
                            new_state.cursor_delta = [0.0, -1.0];
                        }
                        if event.physical_key == PhysicalKey::Code(KeyCode::ArrowDown)
                            && event.state == ElementState::Pressed
                        {
                            new_state.look_around = true;
                            new_state.cursor_delta = [0.0, 1.0];
                        }

                        if (event.physical_key == PhysicalKey::Code(KeyCode::ArrowRight)
                            || event.physical_key == PhysicalKey::Code(KeyCode::ArrowLeft)
                            || event.physical_key == PhysicalKey::Code(KeyCode::ArrowDown)
                            || event.physical_key == PhysicalKey::Code(KeyCode::ArrowUp))
                            && event.state == ElementState::Released
                        {
                            new_state.look_around = false;
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if *button == MouseButton::Right && *state == ElementState::Pressed {
                            new_state.look_around = true;
                            window
                                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                .unwrap_or(());
                        } else if *button == MouseButton::Right && *state == ElementState::Released
                        {
                            new_state.look_around = false;
                            window
                                .set_cursor_grab(winit::window::CursorGrabMode::None)
                                .unwrap_or(());
                        }
                    }
                    WindowEvent::CursorMoved {
                        device_id: _,
                        position,
                    } => {
                        new_state.cursor_position = [position.x, position.y];
                    }
                    _ => {}
                };
            }
            _ => (),
        }

        new_state
    }
}
