use macroquad::prelude::*;
use ::rand::RngExt;
use crate::config::*;
use crate::state::State;

// --- UTILITY: ROTATION IN 2D ---
pub fn rotate_point(x: f32, y: f32, cx: f32, cy: f32, angle: f32) -> (f32, f32) {
    let s = angle.sin();
    let c = angle.cos();
    let dx = x - cx;
    let dy = y - cy;
    let rx = dx * c - dy * s;
    let ry = dx * s + dy * c;
    (rx + cx, ry + cy)
}

// --- PARTICLE SYSTEM ---
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub size: f32,
    pub life: f32,      // from 1.0 down to 0.0
    pub color: Color,
}

impl Particle {
    pub fn new(x: f32, y: f32, color: Color) -> Self {
        let mut rng = ::rand::rng();
        Self {
            x,
            y,
            vx: rng.random_range(-3.0..-0.5),
            vy: rng.random_range(-2.0..2.0),
            size: rng.random_range(3.0..6.0),
            life: 1.0,
            color: if rng.random_bool(0.7) { color } else { Color::from_rgba(255, 255, 255, 200) },
        }
    }

    pub fn update(&mut self) {
        self.x += self.vx;
        self.y += self.vy;
        self.vy += 0.05; // tiny gravity on particles
        self.life -= 0.02;
    }

    pub fn draw(&self) {
        if self.life > 0.0 {
            let mut col = self.color;
            col.a = self.life;
            draw_circle(self.x, self.y, self.size, col);
        }
    }
}

// --- BACKGROUND CLOUD ---
pub struct Cloud {
    pub x: f32,
    pub y: f32,
    pub speed: f32,
    pub width: f32,
    pub height: f32,
}

impl Cloud {
    pub fn new(x: f32, y: f32, speed: f32, width: f32, height: f32) -> Self {
        Self { x, y, speed, width, height }
    }

    pub fn update(&mut self, dt: f32) {
        self.x -= self.speed * dt;
        if self.x + self.width < 0.0 {
            self.x = screen_width() + 50.0;
            let mut rng = ::rand::rng();
            self.y = rng.random_range(50.0..220.0);
        }
    }

    pub fn draw(&self) {
        let col = Color::from_rgba(255, 255, 255, 100);
        // Draw a fluffy cloud using 3 overlapping circles
        let r1 = self.height * 0.5;
        let r2 = self.height * 0.7;
        let r3 = self.height * 0.5;
        draw_circle(self.x + r1, self.y, r1, col);
        draw_circle(self.x + self.width * 0.5, self.y - 10.0, r2, col);
        draw_circle(self.x + self.width - r3, self.y, r3, col);
        draw_rectangle(self.x + r1, self.y - r1 + 5.0, self.width - r1 - r3, r1 * 2.0 - 5.0, col);
    }
}

// --- OBSTACLE: PIPE ---
pub struct Pipe {
    pub x: f32,
    pub gap_y: f32, // y-center of the gap
    pub passed: bool,
}

impl Pipe {
    pub fn new(x: f32, gap_y: f32) -> Self {
        Self { x, gap_y, passed: false }
    }

    pub fn update(&mut self, dt: f32) {
        self.x -= PIPE_SPEED * dt;
    }

    pub fn draw(&self) {
        let screen_h = screen_height();
        let ground_y = screen_h - GROUND_HEIGHT;

        let top_pipe_height = self.gap_y - PIPE_GAP / 2.0;
        let bottom_pipe_y = self.gap_y + PIPE_GAP / 2.0;
        let bottom_pipe_height = ground_y - bottom_pipe_y;

        // Colors
        let main_color = color_pipe_green();
        let border_color = color_pipe_dark();
        let highlight = color_pipe_light();

        // 1. TOP PIPE
        // Column
        draw_rectangle(self.x, 0.0, PIPE_WIDTH, top_pipe_height, main_color);
        // Column borders and highlights
        draw_rectangle_lines(self.x, -5.0, PIPE_WIDTH, top_pipe_height + 5.0, 3.0, border_color);
        draw_rectangle(self.x + 5.0, 0.0, 8.0, top_pipe_height, highlight); // shine effect
        // Top Cap (the lip at the bottom of the top pipe)
        let cap_h = 24.0;
        let cap_w = PIPE_WIDTH + 8.0;
        let cap_x = self.x - 4.0;
        let cap_y = top_pipe_height - cap_h;
        draw_rectangle(cap_x, cap_y, cap_w, cap_h, main_color);
        draw_rectangle_lines(cap_x, cap_y, cap_w, cap_h, 3.0, border_color);
        draw_rectangle(cap_x + 5.0, cap_y + 3.0, 8.0, cap_h - 6.0, highlight);

        // 2. BOTTOM PIPE
        // Column
        draw_rectangle(self.x, bottom_pipe_y, PIPE_WIDTH, bottom_pipe_height, main_color);
        // Column borders and highlights
        draw_rectangle_lines(self.x, bottom_pipe_y, PIPE_WIDTH, bottom_pipe_height + 5.0, 3.0, border_color);
        draw_rectangle(self.x + 5.0, bottom_pipe_y, 8.0, bottom_pipe_height, highlight);
        // Bottom Cap (the lip at the top of the bottom pipe)
        let cap_y_bottom = bottom_pipe_y;
        draw_rectangle(cap_x, cap_y_bottom, cap_w, cap_h, main_color);
        draw_rectangle_lines(cap_x, cap_y_bottom, cap_w, cap_h, 3.0, border_color);
        draw_rectangle(cap_x + 5.0, cap_y_bottom + 3.0, 8.0, cap_h - 6.0, highlight);
    }

    // Check circular collision with bird
    pub fn collides_with(&self, bird_y: f32, r: f32) -> bool {
        let bx = BIRD_X;
        let by = bird_y;

        // Gap boundaries
        let gap_top = self.gap_y - PIPE_GAP / 2.0;

        // Left and Right bounds of the pipe column
        let pipe_left = self.x;
        let pipe_right = self.x + PIPE_WIDTH;

        // Shrink the physical collision hitbox to 75% of visual radius for a more forgiving gaming experience
        let collision_radius = r * 0.75;
        let r_sq = collision_radius * collision_radius;

        // Simple AABB vs Circle collision check
        // Top pipe collision
        let cx1 = bx.max(pipe_left).min(pipe_right);
        let cy1 = by.max(0.0).min(gap_top);
        let dist_sq1 = (bx - cx1) * (bx - cx1) + (by - cy1) * (by - cy1);

        // Bottom pipe collision
        let ground_y = screen_height() - GROUND_HEIGHT;
        let cx2 = bx.max(pipe_left).min(pipe_right);
        let cy2 = by.max(bottom_pipe_y(self)).min(ground_y);
        let dist_sq2 = (bx - cx2) * (bx - cx2) + (by - cy2) * (by - cy2);

        dist_sq1 < r_sq || dist_sq2 < r_sq
    }
}

// Helper to avoid borrow-checker issue in collides_with
fn bottom_pipe_y(pipe: &Pipe) -> f32 {
    pipe.gap_y + PIPE_GAP / 2.0
}

// --- BIRD ---
pub struct Bird {
    pub y: f32,
    pub velocity: f32,
    pub radius: f32,
    pub rotation: f32,
    pub wing_flap_timer: f32,
}

impl Bird {
    pub fn new() -> Self {
        Self {
            y: 250.0,
            velocity: 0.0,
            radius: BIRD_RADIUS,
            rotation: 0.0,
            wing_flap_timer: 0.0,
        }
    }

    pub fn update(&mut self, state: &State) {
        match state {
            State::Start => {
                // Gentle floating up and down
                self.wing_flap_timer += 0.15;
                self.y = 250.0 + (self.wing_flap_timer.sin() * 15.0);
                self.rotation = self.wing_flap_timer.sin() * 0.15;
            }
            State::PlayerPlaying | State::AIPlaying => {
                self.velocity += GRAVITY;
                self.y += self.velocity;
                self.wing_flap_timer += 0.25;

                // Adjust rotation based on velocity
                // Tilts up when jumping, tilts down when falling
                let target_rot = if self.velocity < 0.0 {
                    (self.velocity * 0.08).max(-0.4) // max ~23 deg up
                } else {
                    (self.velocity * 0.05).min(0.9) // max ~50 deg down
                };
                // Smooth interpolation for rotation
                self.rotation += (target_rot - self.rotation) * 0.2;

                // Ceiling collision
                if self.y - self.radius < 0.0 {
                    self.y = self.radius;
                    self.velocity = 0.0;
                }
            }
            State::GameOver => {
                // Let the bird fall to the ground
                let ground_y = screen_height() - GROUND_HEIGHT - self.radius;
                if self.y < ground_y {
                    self.velocity += GRAVITY * 1.5;
                    self.y = (self.y + self.velocity).min(ground_y);
                    self.rotation += (1.2 - self.rotation) * 0.15; // Point straight down on death
                }
            }
        }
    }

    pub fn jump(&mut self, _particles: &mut Vec<Particle>, _color: Color) {
        self.velocity = JUMP_FORCE;
    }

    pub fn draw(&self, body_color: Color) {
        let cx = BIRD_X;
        let cy = self.y;
        let r = self.radius;

        // Draw bird parts with rotation
        // 1. body with custom color
        draw_circle(cx, cy, r, body_color);
        draw_circle_lines(cx, cy, r, 2.5, BLACK);

        // 2. White big eye
        let eye_ox = r * 0.4;
        let eye_oy = -r * 0.35;
        let (eye_x, eye_y) = rotate_point(cx + eye_ox, cy + eye_oy, cx, cy, self.rotation);
        draw_circle(eye_x, eye_y, r * 0.35, WHITE);
        draw_circle_lines(eye_x, eye_y, r * 0.35, 1.8, BLACK);

        // Black pupil
        let (pupil_x, pupil_y) = rotate_point(cx + eye_ox + 1.5, cy + eye_oy - 0.5, cx, cy, self.rotation);
        draw_circle(pupil_x, pupil_y, r * 0.15, BLACK);

        // 3. Orange beak
        let beak_tip_x = cx + r * 1.5;
        let beak_tip_y = cy + r * 0.1;
        let beak_top_x = cx + r * 0.8;
        let beak_top_y = cy - r * 0.2;
        let beak_bot_x = cx + r * 0.8;
        let beak_bot_y = cy + r * 0.4;

        let p1 = rotate_point(beak_tip_x, beak_tip_y, cx, cy, self.rotation);
        let p2 = rotate_point(beak_top_x, beak_top_y, cx, cy, self.rotation);
        let p3 = rotate_point(beak_bot_x, beak_bot_y, cx, cy, self.rotation);

        draw_triangle(
            vec2(p1.0, p1.1),
            vec2(p2.0, p2.1),
            vec2(p3.0, p3.1),
            color_bird_beak()
        );
        draw_triangle_lines(
            vec2(p1.0, p1.1),
            vec2(p2.0, p2.1),
            vec2(p3.0, p3.1),
            2.0,
            BLACK
        );

        // 4. White flapping wing
        let flap = (self.wing_flap_timer.cos() * (r * 0.4)).max(-r * 0.4);
        let wing_oy = r * 0.1;
        let wing_tip_x = cx - r * 0.9;
        let wing_tip_y = cy + wing_oy + flap;
        let wing_top_x = cx - r * 0.1;
        let wing_top_y = cy - r * 0.3;
        let wing_bot_x = cx - r * 0.1;
        let wing_bot_y = cy + r * 0.4;

        let w1 = rotate_point(wing_tip_x, wing_tip_y, cx, cy, self.rotation);
        let w2 = rotate_point(wing_top_x, wing_top_y, cx, cy, self.rotation);
        let w3 = rotate_point(wing_bot_x, wing_bot_y, cx, cy, self.rotation);

        draw_triangle(
            vec2(w1.0, w1.1),
            vec2(w2.0, w2.1),
            vec2(w3.0, w3.1),
            color_bird_wing()
        );
        draw_triangle_lines(
            vec2(w1.0, w1.1),
            vec2(w2.0, w2.1),
            vec2(w3.0, w3.1),
            2.0,
            BLACK
        );
    }
}
