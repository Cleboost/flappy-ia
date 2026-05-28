use macroquad::prelude::*;
use ::rand::Rng;

// --- CONFIGURATION CONSTANTS ---
const GRAVITY: f32 = 0.15;
const JUMP_FORCE: f32 = -5.0;
const BIRD_X: f32 = 120.0;
const BIRD_RADIUS: f32 = 16.0;
const PIPE_WIDTH: f32 = 80.0;
const PIPE_GAP: f32 = 150.0;
const PIPE_SPEED: f32 = 180.0; // pixels per second
const SPAWN_INTERVAL: f32 = 1.8; // seconds between pipe spawns
const GROUND_HEIGHT: f32 = 100.0;

// --- COLOR PALETTE (Classic Modern Pastel) ---
fn color_sky() -> Color { Color::from_rgba(113, 197, 207, 255) }
fn color_ground() -> Color { Color::from_rgba(222, 216, 149, 255) }
fn color_grass() -> Color { Color::from_rgba(115, 191, 46, 255) }
fn color_pipe_green() -> Color { Color::from_rgba(115, 191, 46, 255) }
fn color_pipe_dark() -> Color { Color::from_rgba(83, 128, 32, 255) }
fn color_pipe_light() -> Color { Color::from_rgba(156, 219, 67, 255) }
fn color_bird_body() -> Color { Color::from_rgba(250, 188, 32, 255) }
fn color_bird_wing() -> Color { Color::from_rgba(255, 255, 255, 255) }
fn color_bird_beak() -> Color { Color::from_rgba(240, 90, 40, 255) }

// --- GAME STATES ---
enum State {
    Start,
    Playing,
    GameOver,
}

// --- UTILITY: ROTATION IN 2D ---
fn rotate_point(x: f32, y: f32, cx: f32, cy: f32, angle: f32) -> (f32, f32) {
    let s = angle.sin();
    let c = angle.cos();
    let dx = x - cx;
    let dy = y - cy;
    let rx = dx * c - dy * s;
    let ry = dx * s + dy * c;
    (rx + cx, ry + cy)
}

// --- PARTICLE SYSTEM ---
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    size: f32,
    life: f32,      // from 1.0 down to 0.0
    color: Color,
}

impl Particle {
    fn new(x: f32, y: f32) -> Self {
        let mut rng = ::rand::thread_rng();
        Self {
            x,
            y,
            vx: rng.gen_range(-3.0..-0.5),
            vy: rng.gen_range(-2.0..2.0),
            size: rng.gen_range(3.0..6.0),
            life: 1.0,
            color: if rng.gen_bool(0.7) { color_bird_body() } else { Color::from_rgba(255, 255, 255, 200) },
        }
    }

    fn update(&mut self) {
        self.x += self.vx;
        self.y += self.vy;
        self.vy += 0.05; // tiny gravity on particles
        self.life -= 0.02;
    }

    fn draw(&self) {
        if self.life > 0.0 {
            let mut col = self.color;
            col.a = self.life;
            draw_circle(self.x, self.y, self.size, col);
        }
    }
}

// --- BACKGROUND CLOUD ---
struct Cloud {
    x: f32,
    y: f32,
    speed: f32,
    width: f32,
    height: f32,
}

impl Cloud {
    fn new(x: f32, y: f32, speed: f32, width: f32, height: f32) -> Self {
        Self { x, y, speed, width, height }
    }

    fn update(&mut self, dt: f32) {
        self.x -= self.speed * dt;
        if self.x + self.width < 0.0 {
            self.x = screen_width() + 50.0;
            let mut rng = ::rand::thread_rng();
            self.y = rng.gen_range(50.0..220.0);
        }
    }

    fn draw(&self) {
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
struct Pipe {
    x: f32,
    gap_y: f32, // y-center of the gap
    passed: bool,
}

impl Pipe {
    fn new(x: f32, gap_y: f32) -> Self {
        Self { x, gap_y, passed: false }
    }

    fn update(&mut self, dt: f32) {
        self.x -= PIPE_SPEED * dt;
    }

    fn draw(&self) {
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
    fn collides_with(&self, bird_y: f32, r: f32) -> bool {
        let bx = BIRD_X;
        let by = bird_y;

        // Gap boundaries
        let gap_top = self.gap_y - PIPE_GAP / 2.0;

        // Left and Right bounds of the pipe column
        let pipe_left = self.x;
        let pipe_right = self.x + PIPE_WIDTH;

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

        dist_sq1 < r * r || dist_sq2 < r * r
    }
}

// Helper to avoid borrow-checker issue in collides_with
fn bottom_pipe_y(pipe: &Pipe) -> f32 {
    pipe.gap_y + PIPE_GAP / 2.0
}

// --- BIRD ---
struct Bird {
    y: f32,
    velocity: f32,
    radius: f32,
    rotation: f32,
    wing_flap_timer: f32,
}

impl Bird {
    fn new() -> Self {
        Self {
            y: 250.0,
            velocity: 0.0,
            radius: BIRD_RADIUS,
            rotation: 0.0,
            wing_flap_timer: 0.0,
        }
    }

    fn update(&mut self, state: &State) {
        match state {
            State::Start => {
                // Gentle floating up and down
                self.wing_flap_timer += 0.15;
                self.y = 250.0 + (self.wing_flap_timer.sin() * 15.0);
                self.rotation = self.wing_flap_timer.sin() * 0.15;
            }
            State::Playing => {
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

    fn jump(&mut self, particles: &mut Vec<Particle>) {
        self.velocity = JUMP_FORCE;
        // Spawn jump particles behind/below the bird
        for _ in 0..6 {
            particles.push(Particle::new(BIRD_X - 10.0, self.y + 5.0));
        }
    }

    fn draw(&self) {
        let cx = BIRD_X;
        let cy = self.y;
        let r = self.radius;

        // Draw bird parts with rotation
        // 1. Yellow body
        draw_circle(cx, cy, r, color_bird_body());
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

// --- TEXT WITH SHADOW HELPER ---
fn draw_text_shadow(text: &str, x: f32, y: f32, size: f32, color: Color) {
    draw_text(text, x + 3.0, y + 3.0, size, Color::from_rgba(0, 0, 0, 150));
    draw_text(text, x, y, size, color);
}

// --- MAIN FUNCTION ---
#[macroquad::main("Flappy Bird Premium")]
async fn main() {
    // Game variables
    let mut state = State::Start;
    let mut bird = Bird::new();
    let mut pipes: Vec<Pipe> = Vec::new();
    let mut particles: Vec<Particle> = Vec::new();
    let mut clouds: Vec<Cloud> = vec![
        Cloud::new(100.0, 80.0, 15.0, 110.0, 45.0),
        Cloud::new(400.0, 150.0, 25.0, 140.0, 55.0),
        Cloud::new(700.0, 100.0, 10.0, 90.0, 35.0),
    ];

    let mut pipe_spawn_timer = 0.0;
    let mut score: u32 = 0;
    let mut high_score: u32 = 0;
    let mut ground_scroll_x = 0.0;

    loop {
        let dt = get_frame_time();

        // --- INPUT & UPDATE ---
        match state {
            State::Start => {
                // Scroll ground and clouds even in start screen
                ground_scroll_x = (ground_scroll_x - PIPE_SPEED * dt) % 30.0;
                for cloud in &mut clouds {
                    cloud.update(dt);
                }

                bird.update(&state);

                // Space to start
                if is_key_pressed(KeyCode::Space) || is_mouse_button_pressed(MouseButton::Left) {
                    state = State::Playing;
                    bird.jump(&mut particles);
                }
            }
            State::Playing => {
                // Update clouds
                for cloud in &mut clouds {
                    cloud.update(dt);
                }

                // Scroll ground
                ground_scroll_x = (ground_scroll_x - PIPE_SPEED * dt) % 30.0;

                // Bird physics
                bird.update(&state);

                // Jump input
                if is_key_pressed(KeyCode::Space) || is_mouse_button_pressed(MouseButton::Left) {
                    bird.jump(&mut particles);
                }

                // Spawn pipes
                pipe_spawn_timer += dt;
                if pipe_spawn_timer >= SPAWN_INTERVAL {
                    pipe_spawn_timer = 0.0;
                    let mut rng = ::rand::thread_rng();
                    // Generate gap center between 150 and screen_height - 150 - GroundHeight
                    let min_y = 150.0;
                    let max_y = (screen_height() - GROUND_HEIGHT - 150.0).max(min_y + 10.0);
                    let gap_y = rng.gen_range(min_y..max_y);
                    pipes.push(Pipe::new(screen_width() + 50.0, gap_y));
                }

                // Update pipes
                for pipe in &mut pipes {
                    pipe.update(dt);

                    // Check if passed for score
                    if !pipe.passed && pipe.x + PIPE_WIDTH / 2.0 < BIRD_X {
                        pipe.passed = true;
                        score += 1;
                    }
                }

                // Remove off-screen pipes
                pipes.retain(|pipe| pipe.x + PIPE_WIDTH + 10.0 > 0.0);

                // Collision detection
                let ground_y = screen_height() - GROUND_HEIGHT;
                
                // 1. Collides with ground
                if bird.y + bird.radius >= ground_y {
                    state = State::GameOver;
                    if score > high_score {
                        high_score = score;
                    }
                }

                // 2. Collides with any pipe
                for pipe in &pipes {
                    if pipe.collides_with(bird.y, bird.radius) {
                        state = State::GameOver;
                        if score > high_score {
                            high_score = score;
                        }
                        break;
                    }
                }
            }
            State::GameOver => {
                // Update clouds (slowly)
                for cloud in &mut clouds {
                    cloud.update(dt * 0.2);
                }

                bird.update(&state);

                // Restart input: Space or R
                if is_key_pressed(KeyCode::Space) || is_key_pressed(KeyCode::R) {
                    // Reset game
                    state = State::Playing;
                    bird = Bird::new();
                    pipes.clear();
                    particles.clear();
                    score = 0;
                    pipe_spawn_timer = 0.0;
                    bird.jump(&mut particles);
                }
            }
        }

        // Update particles
        for p in &mut particles {
            p.update();
        }
        particles.retain(|p| p.life > 0.0);

        // --- DRAWING ---
        clear_background(color_sky());

        // 1. Draw Clouds
        for cloud in &clouds {
            cloud.draw();
        }

        // 2. Draw Pipes
        for pipe in &pipes {
            pipe.draw();
        }

        // 3. Draw Particles
        for p in &particles {
            p.draw();
        }

        // 4. Draw Bird
        bird.draw();

        // 5. Draw Ground
        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = screen_h - GROUND_HEIGHT;

        // Main ground base
        draw_rectangle(0.0, ground_y, screen_w, GROUND_HEIGHT, color_ground());
        // Green grass top line
        draw_rectangle(0.0, ground_y, screen_w, 15.0, color_grass());
        // Dark separating border line
        draw_line(0.0, ground_y, screen_w, ground_y, 4.0, color_pipe_dark());
        draw_line(0.0, ground_y + 15.0, screen_w, ground_y + 15.0, 3.0, color_pipe_dark());

        // Parallax ground pattern scrolling lines
        let pattern_width = 30.0;
        let mut x = ground_scroll_x;
        while x < screen_w + pattern_width {
            // Draw nice orange-brown diagonal stripe/grass shadows
            draw_line(x, ground_y + 15.0, x - 15.0, ground_y + GROUND_HEIGHT, 4.0, Color::from_rgba(0, 0, 0, 30));
            x += pattern_width;
        }

        // 6. Draw HUD / UI
        match state {
            State::Start => {
                // Title
                let title_text = "FLAPPY BIRD";
                let font_size = 60.0;
                let text_w = measure_text(title_text, None, font_size as u16, 1.0).width;
                draw_text_shadow(title_text, (screen_w - text_w) / 2.0, 150.0, font_size, WHITE);

                // Subtitle (Pulsing instruction)
                let sub_text = "APPUYEZ SUR ESPACE POUR JOUER";
                let pulse = (get_time() * 4.0).sin() as f32;
                let col = if pulse > 0.0 { WHITE } else { Color::from_rgba(230, 230, 230, 200) };
                let font_size_sub = 24.0;
                let text_sub_w = measure_text(sub_text, None, font_size_sub as u16, 1.0).width;
                draw_text_shadow(sub_text, (screen_w - text_sub_w) / 2.0, screen_h * 0.65, font_size_sub, col);
            }
            State::Playing => {
                // Large score in the center
                let score_str = score.to_string();
                let font_size = 72.0;
                let text_w = measure_text(&score_str, None, font_size as u16, 1.0).width;
                draw_text_shadow(&score_str, (screen_w - text_w) / 2.0, 100.0, font_size, WHITE);
            }
            State::GameOver => {
                // Game Over Card
                let card_w = 340.0;
                let card_h = 240.0;
                let card_x = (screen_w - card_w) / 2.0;
                let card_y = (screen_h - card_h) / 2.0 - 20.0;

                // Shadow and Card background
                draw_rectangle(card_x + 5.0, card_y + 5.0, card_w, card_h, Color::from_rgba(0, 0, 0, 80));
                draw_rectangle(card_x, card_y, card_w, card_h, Color::from_rgba(245, 240, 220, 255));
                draw_rectangle_lines(card_x, card_y, card_w, card_h, 5.0, color_pipe_dark());

                // Title "GAME OVER"
                let go_text = "GAME OVER";
                let go_size = 40.0;
                let go_w = measure_text(go_text, None, go_size as u16, 1.0).width;
                draw_text_shadow(go_text, (screen_w - go_w) / 2.0, card_y + 50.0, go_size, Color::from_rgba(220, 50, 40, 255));

                // Scores
                let score_text = format!("SCORE: {}", score);
                let score_size = 28.0;
                let sc_w = measure_text(&score_text, None, score_size as u16, 1.0).width;
                draw_text(&score_text, (screen_w - sc_w) / 2.0, card_y + 110.0, score_size, color_pipe_dark());

                let best_text = format!("MEILLEUR: {}", high_score);
                let best_size = 28.0;
                let bt_w = measure_text(&best_text, None, best_size as u16, 1.0).width;
                draw_text(&best_text, (screen_w - bt_w) / 2.0, card_y + 150.0, best_size, color_pipe_dark());

                // Instruction to restart
                let restart_text = "ESPACE ou CLIC pour rejouer";
                let restart_size = 20.0;
                let rs_w = measure_text(restart_text, None, restart_size as u16, 1.0).width;
                let pulse = (get_time() * 5.0).sin() as f32;
                let rs_color = if pulse > 0.0 { color_bird_beak() } else { color_pipe_dark() };
                draw_text(restart_text, (screen_w - rs_w) / 2.0, card_y + 200.0, restart_size, rs_color);
            }
        }

        next_frame().await
    }
}