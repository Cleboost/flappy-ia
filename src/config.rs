use macroquad::prelude::Color;

pub const GRAVITY: f32 = 0.15;
pub const JUMP_FORCE: f32 = -5.0;
pub const BIRD_X: f32 = 120.0;
pub const BIRD_RADIUS: f32 = 16.0;
pub const PIPE_WIDTH: f32 = 80.0;
pub const PIPE_GAP: f32 = 150.0;
pub const PIPE_SPEED: f32 = 180.0; // pixels per second
pub const SPAWN_INTERVAL: f32 = 1.8; // seconds between pipe spawns
pub const GROUND_HEIGHT: f32 = 100.0;
pub const PHYSICS_DT: f32 = 1.0 / 60.0;

// --- COLOR PALETTE (Classic Modern Pastel) ---
pub fn color_sky() -> Color { Color::from_rgba(113, 197, 207, 255) }
pub fn color_ground() -> Color { Color::from_rgba(222, 216, 149, 255) }
pub fn color_grass() -> Color { Color::from_rgba(115, 191, 46, 255) }
pub fn color_pipe_green() -> Color { Color::from_rgba(115, 191, 46, 255) }
pub fn color_pipe_dark() -> Color { Color::from_rgba(83, 128, 32, 255) }
pub fn color_pipe_light() -> Color { Color::from_rgba(156, 219, 67, 255) }
pub fn color_bird_body() -> Color { Color::from_rgba(250, 188, 32, 255) }
pub fn color_bird_wing() -> Color { Color::from_rgba(255, 255, 255, 255) }
pub fn color_bird_beak() -> Color { Color::from_rgba(240, 90, 40, 255) }
