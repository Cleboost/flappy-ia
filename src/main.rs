use macroquad::prelude::*;
use ::rand::RngExt;
use rayon::prelude::*;

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
const PHYSICS_DT: f32 = 1.0 / 60.0;

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
#[derive(PartialEq, Clone, Copy)]
enum State {
    Start,
    PlayerPlaying,
    AIPlaying,
    GameOver,
}

// --- NEURAL NETWORK FOR AI ---
#[derive(Clone)]
struct Brain {
    // 5 inputs (dy_cur, dx_cur, velocity, dy_next, dx_next) -> 5 hidden neurons
    weights_ih: [[f32; 5]; 5],
    bias_h: [f32; 5],
    // 5 hidden -> 1 output neuron
    weights_ho: [f32; 5],
    bias_o: f32,
}

impl Brain {
    fn random() -> Self {
        let mut rng = ::rand::rng();
        let mut weights_ih = [[0.0; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                weights_ih[i][j] = rng.random_range(-1.0..1.0);
            }
        }
        let mut bias_h = [0.0; 5];
        for i in 0..5 {
            bias_h[i] = rng.random_range(-1.0..1.0);
        }
        let mut weights_ho = [0.0; 5];
        for i in 0..5 {
            weights_ho[i] = rng.random_range(-1.0..1.0);
        }
        let bias_o = rng.random_range(-1.0..1.0);

        Self {
            weights_ih,
            bias_h,
            weights_ho,
            bias_o,
        }
    }

    fn crossover(&self, other: &Self) -> Self {
        let mut rng = ::rand::rng();
        let mut child = self.clone();

        // Crossover for weights_ih (5x5 = 25 weights)
        for i in 0..5 {
            for j in 0..5 {
                if rng.random_bool(0.5) {
                    child.weights_ih[i][j] = other.weights_ih[i][j];
                }
            }
        }

        // Crossover for bias_h (5 values)
        for i in 0..5 {
            if rng.random_bool(0.5) {
                child.bias_h[i] = other.bias_h[i];
            }
        }

        // Crossover for weights_ho (5 values)
        for i in 0..5 {
            if rng.random_bool(0.5) {
                child.weights_ho[i] = other.weights_ho[i];
            }
        }

        // Crossover for bias_o
        if rng.random_bool(0.5) {
            child.bias_o = other.bias_o;
        }

        child
    }

    fn mutate(&self, rate: f32) -> Self {
        let mut mutated = self.clone();
        let mut rng = ::rand::rng();

        let mut mutate_val = |val: &mut f32| {
            if rng.random_bool(rate as f64) {
                if rng.random_bool(0.15) {
                    *val = rng.random_range(-1.0..1.0);
                } else {
                    *val += rng.random_range(-0.35..0.35);
                    *val = val.clamp(-2.0, 2.0);
                }
            }
        };

        for i in 0..5 {
            for j in 0..5 {
                mutate_val(&mut mutated.weights_ih[i][j]);
            }
        }
        for i in 0..5 {
            mutate_val(&mut mutated.bias_h[i]);
            mutate_val(&mut mutated.weights_ho[i]);
        }
        mutate_val(&mut mutated.bias_o);

        mutated
    }

    fn predict(&self, dy: f32, dx: f32, vel: f32, dy_next: f32, dx_next: f32) -> bool {
        let mut hidden = [0.0; 5];
        for i in 0..5 {
            let sum = dy * self.weights_ih[i][0]
                + dx * self.weights_ih[i][1]
                + vel * self.weights_ih[i][2]
                + dy_next * self.weights_ih[i][3]
                + dx_next * self.weights_ih[i][4]
                + self.bias_h[i];
            hidden[i] = sum.tanh();
        }

        let mut output_sum = self.bias_o;
        for i in 0..5 {
            output_sum += hidden[i] * self.weights_ho[i];
        }

        output_sum > 0.0
    }
}

// --- EVOLUTIONARY AI AGENT ---
struct Agent {
    bird: Bird,
    brain: Brain,
    alive: bool,
    score: u32,
    fitness_score: f32,
    color: Color,
}

impl Agent {
    fn new(brain: Brain, color: Color) -> Self {
        Self {
            bird: Bird::new(),
            brain,
            alive: true,
            score: 0,
            fitness_score: 0.0,
            color,
        }
    }

    fn fitness(&self) -> f32 {
        self.fitness_score + (self.score as f32) * 1000.0
    }
}

fn random_pastel_color() -> Color {
    let mut rng = ::rand::rng();
    let r = rng.random_range(0.4..1.0);
    let g = rng.random_range(0.4..1.0);
    let b = rng.random_range(0.4..1.0);
    Color::new(r, g, b, 1.0)
}

// --- SAVE & LOAD SYSTEM ---
struct SaveData {
    generation: u32,
    high_score: u32,
    last_best_score: u32,
    last_best_fitness: f32,
    brain: Brain,
}

fn save_game(
    filename: &str,
    generation: u32,
    high_score: u32,
    last_best_score: u32,
    last_best_fitness: f32,
    brain: &Brain,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;
    writeln!(file, "2")?; // Save format version
    writeln!(file, "{}", generation)?;
    writeln!(file, "{}", high_score)?;
    writeln!(file, "{}", last_best_score)?;
    writeln!(file, "{}", last_best_fitness)?;
    writeln!(file, "{}", brain.bias_o)?;
    
    for i in 0..5 {
        writeln!(file, "{}", brain.bias_h[i])?;
    }
    for i in 0..5 {
        writeln!(file, "{}", brain.weights_ho[i])?;
    }
    for i in 0..5 {
        for j in 0..5 {
            writeln!(file, "{}", brain.weights_ih[i][j])?;
        }
    }
    Ok(())
}

fn load_game(filename: &str) -> std::io::Result<SaveData> {
    use std::fs::File;
    use std::io::{BufReader, BufRead};

    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Check save format version — v2 required (5-input brain)
    let version: u32 = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing version"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid version"))?;
    if version != 2 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "incompatible save format (not v2)"));
    }

    let generation = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing generation"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid generation"))?;
    let high_score = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing high_score"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid high_score"))?;
    let last_best_score = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing last_best_score"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid last_best_score"))?;
    let last_best_fitness = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing last_best_fitness"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid last_best_fitness"))?;
    let bias_o = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing bias_o"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid bias_o"))?;
    
    let mut bias_h = [0.0; 5];
    for i in 0..5 {
        bias_h[i] = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing bias_h"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid bias_h"))?;
    }
    
    let mut weights_ho = [0.0; 5];
    for i in 0..5 {
        weights_ho[i] = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing weights_ho"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid weights_ho"))?;
    }
    
    let mut weights_ih = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            weights_ih[i][j] = lines.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing weights_ih"))??.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid weights_ih"))?;
        }
    }

    Ok(SaveData {
        generation,
        high_score,
        last_best_score,
        last_best_fitness,
        brain: Brain {
            weights_ih,
            bias_h,
            weights_ho,
            bias_o,
        }
    })
}

// --- REAL-TIME NEURAL NETWORK VISUALIZER ---
fn draw_neuron_map(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    brain: &Brain,
    dy: f32,
    dx: f32,
    vel: f32,
    dy_next: f32,
    dx_next: f32,
) {
    // 1. Draw Panel Background (Glassmorphism)
    draw_rectangle(x, y, w, h, Color::from_rgba(255, 255, 255, 30));
    draw_rectangle(x, y, w, h, Color::from_rgba(0, 0, 0, 100));
    draw_rectangle_lines(x, y, w, h, 3.0, Color::from_rgba(255, 255, 255, 80));

    // Title
    draw_text("VISION NEURONALE", x + 15.0, y + 35.0, 20.0, Color::from_rgba(100, 230, 255, 255));
    draw_line(x + 15.0, y + 45.0, x + w - 15.0, y + 45.0, 2.0, Color::from_rgba(255, 255, 255, 40));

    // 2. Node Coordinates (5 inputs, uniformly spaced)
    let x_in = x + 35.0;
    let x_hid = x + w / 2.0;
    let x_out = x + w - 35.0;

    let y_in = [y + 65.0, y + 100.0, y + 135.0, y + 170.0, y + 205.0];
    let y_hid = [y + 65.0, y + 100.0, y + 135.0, y + 170.0, y + 205.0];
    let y_out = [y + 135.0];

    // Compute node activations dynamically
    let mut act_hid = [0.0; 5];
    for i in 0..5 {
        let sum = dy * brain.weights_ih[i][0]
            + dx * brain.weights_ih[i][1]
            + vel * brain.weights_ih[i][2]
            + dy_next * brain.weights_ih[i][3]
            + dx_next * brain.weights_ih[i][4]
            + brain.bias_h[i];
        act_hid[i] = sum.tanh();
    }
    
    let mut act_out = brain.bias_o;
    for i in 0..5 {
        act_out += act_hid[i] * brain.weights_ho[i];
    }
    let jumped = act_out > 0.0;
    let act_out_val = act_out.tanh();

    // 3. Draw Connections (Weights)
    // Input -> Hidden
    for i in 0..5 {
        for j in 0..5 {
            let weight = brain.weights_ih[i][j];
            let thickness = (weight.abs() * 2.0).clamp(0.5, 4.0);
            let color = if weight > 0.0 {
                Color::from_rgba(100, 255, 100, (weight.abs() * 180.0) as u8)
            } else {
                Color::from_rgba(255, 100, 100, (weight.abs() * 180.0) as u8)
            };
            draw_line(x_in, y_in[j], x_hid, y_hid[i], thickness, color);
        }
    }

    // Input -> Hidden sparks
    let time = get_time();
    let spark_prog = ((time * 1.5) % 1.0) as f32;
    for i in 0..5 {
        for j in 0..5 {
            let weight = brain.weights_ih[i][j];
            if weight.abs() > 0.2 {
                let px = x_in + (x_hid - x_in) * spark_prog;
                let py = y_in[j] + (y_hid[i] - y_in[j]) * spark_prog;
                draw_circle(px, py, 2.0, WHITE);
            }
        }
    }

    // Hidden -> Output
    for i in 0..5 {
        let weight = brain.weights_ho[i];
        let thickness = (weight.abs() * 2.0).clamp(0.5, 4.0);
        let color = if weight > 0.0 {
            Color::from_rgba(100, 255, 100, (weight.abs() * 180.0) as u8)
        } else {
            Color::from_rgba(255, 100, 100, (weight.abs() * 180.0) as u8)
        };
        draw_line(x_hid, y_hid[i], x_out, y_out[0], thickness, color);
    }

    // Hidden -> Output sparks
    let spark_prog_ho = ((time * 1.5 + 0.5) % 1.0) as f32;
    for i in 0..5 {
        let weight = brain.weights_ho[i];
        if weight.abs() > 0.2 {
            let px = x_hid + (x_out - x_hid) * spark_prog_ho;
            let py = y_hid[i] + (y_out[0] - y_hid[i]) * spark_prog_ho;
            draw_circle(px, py, 2.0, WHITE);
        }
    }

    // 4. Draw Nodes
    // Input Nodes (5): dY_cur, dX_cur, velocity, dY_next, dX_next
    let in_vals = [dy, dx, vel, dy_next, dx_next];
    let in_labels = ["dY", "dX", "V", "dY2", "dX2"];
    for i in 0..5 {
        let val = in_vals[i];
        // dY2/dX2 (next pipe) displayed with a slightly different hue (orange instead of green)
        let color = if i >= 3 {
            if val > 0.0 {
                Color::from_rgba(255, 200, 50, (80.0 + val.abs() * 175.0).min(255.0) as u8)
            } else {
                Color::from_rgba(255, 120, 50, (80.0 + val.abs() * 175.0).min(255.0) as u8)
            }
        } else if val > 0.0 {
            Color::from_rgba(100, 255, 100, (80.0 + val.abs() * 175.0).min(255.0) as u8)
        } else {
            Color::from_rgba(255, 100, 100, (80.0 + val.abs() * 175.0).min(255.0) as u8)
        };
        draw_circle(x_in, y_in[i], 12.0, color);
        draw_circle_lines(x_in, y_in[i], 12.0, 2.0, BLACK);
        draw_text(in_labels[i], x_in - 9.0, y_in[i] + 4.0, 11.0, BLACK);
    }

    // Hidden Nodes
    for i in 0..5 {
        let val = act_hid[i];
        let color = if val > 0.0 {
            Color::from_rgba(100, 230, 255, (100.0 + val.abs() * 155.0).min(255.0) as u8)
        } else {
            Color::from_rgba(255, 200, 100, (100.0 + val.abs() * 155.0).min(255.0) as u8)
        };
        draw_circle(x_hid, y_hid[i], 10.0, color);
        draw_circle_lines(x_hid, y_hid[i], 10.0, 2.0, BLACK);
    }

    // Output Node
    let out_color = if jumped {
        Color::from_rgba(100, 255, 100, 255)
    } else {
        let val = act_out_val.abs();
        Color::from_rgba(150, 150, 150, (100.0 + val * 155.0).min(255.0) as u8)
    };
    draw_circle(x_out, y_out[0], 14.0, out_color);
    draw_circle_lines(x_out, y_out[0], 14.0, 2.5, BLACK);
    draw_text("J", x_out - 4.0, y_out[0] + 5.0, 15.0, BLACK);
}

// --- EVOLUTION PROGRESS GRAPH ---

fn draw_evolution_chart(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    max_history: &[u32],
    avg_history: &[f32],
) {
    // 1. Draw Panel Background (Glassmorphism)
    draw_rectangle(x, y, w, h, Color::from_rgba(255, 255, 255, 30));
    draw_rectangle(x, y, w, h, Color::from_rgba(0, 0, 0, 100));
    draw_rectangle_lines(x, y, w, h, 3.0, Color::from_rgba(255, 255, 255, 80));

    // Title
    draw_text("COURBE DE PROGRESSION", x + 15.0, y + 30.0, 18.0, Color::from_rgba(100, 230, 255, 255));
    draw_line(x + 15.0, y + 38.0, x + w - 15.0, y + 38.0, 2.0, Color::from_rgba(255, 255, 255, 40));

    if max_history.is_empty() {
        draw_text("En attente de données...", x + 25.0, y + h / 2.0, 16.0, Color::from_rgba(200, 200, 200, 150));
        return;
    }

    // Chart dimensions
    let chart_x = x + 30.0;
    let chart_y = y + 55.0;
    let chart_w = w - 45.0;
    let chart_h = h - 90.0;

    // Draw Grid Lines
    draw_rectangle_lines(chart_x, chart_y, chart_w, chart_h, 1.0, Color::from_rgba(255, 255, 255, 30));
    
    // Find absolute max value in history to scale the Y-axis
    let mut highest_score = 1.0f32;
    for &val in max_history {
        if val as f32 > highest_score {
            highest_score = val as f32;
        }
    }
    let y_max = (highest_score * 1.1).max(5.0);

    // Draw chart axes labels
    draw_text(&format!("{:.0}", y_max), chart_x - 22.0, chart_y + 10.0, 11.0, Color::from_rgba(220, 220, 220, 200));
    draw_text("0", chart_x - 12.0, chart_y + chart_h + 3.0, 11.0, Color::from_rgba(220, 220, 220, 200));

    // Draw Lines
    let count = max_history.len();
    if count > 1 {
        let x_step = chart_w / (count - 1) as f32;

        // Draw Average Line (Yellow/Orange)
        for i in 0..(count - 1) {
            let x1 = chart_x + i as f32 * x_step;
            let y1 = chart_y + chart_h - (avg_history[i] / y_max) * chart_h;
            let x2 = chart_x + (i + 1) as f32 * x_step;
            let y2 = chart_y + chart_h - (avg_history[i + 1] / y_max) * chart_h;
            draw_line(x1, y1, x2, y2, 2.0, Color::from_rgba(255, 200, 50, 200));
        }

        // Draw Maximum Line (Neon Cyan/Green)
        for i in 0..(count - 1) {
            let x1 = chart_x + i as f32 * x_step;
            let y1 = chart_y + chart_h - (max_history[i] as f32 / y_max) * chart_h;
            let x2 = chart_x + (i + 1) as f32 * x_step;
            let y2 = chart_y + chart_h - (max_history[i + 1] as f32 / y_max) * chart_h;
            draw_line(x1, y1, x2, y2, 3.0, Color::from_rgba(100, 255, 100, 255));
        }
    }

    // Legend
    let legend_y = y + h - 18.0;
    draw_rectangle(chart_x, legend_y - 6.0, 12.0, 4.0, Color::from_rgba(100, 255, 100, 255));
    draw_text("Record Gen", chart_x + 18.0, legend_y - 2.0, 12.0, Color::from_rgba(200, 200, 200, 255));
    
    draw_rectangle(chart_x + 100.0, legend_y - 6.0, 12.0, 4.0, Color::from_rgba(255, 200, 50, 200));
    draw_text("Moyenne", chart_x + 118.0, legend_y - 2.0, 12.0, Color::from_rgba(200, 200, 200, 255));
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
#[allow(dead_code)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    size: f32,
    life: f32,      // from 1.0 down to 0.0
    color: Color,
}

#[allow(dead_code)]
impl Particle {
    fn new(x: f32, y: f32, color: Color) -> Self {
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
            let mut rng = ::rand::rng();
            self.y = rng.random_range(50.0..220.0);
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

    fn jump(&mut self, _particles: &mut Vec<Particle>, _color: Color) {
        self.velocity = JUMP_FORCE;
    }

    fn draw(&self, body_color: Color) {
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

    let mut pipe_spawn_timer = SPAWN_INTERVAL;
    let mut score: u32 = 0;
    let mut high_score: u32 = 0;
    let mut ground_scroll_x = 0.0;

    // AI variables
    let mut generation: u32 = 1;
    let mut speed_multiplier: u32 = 1;
    let mut last_best_score: u32 = 0;
    let mut last_best_fitness: f32 = 0.0;
    let mut agents: Vec<Agent> = Vec::new();
    let mut history_max_scores: Vec<u32> = Vec::new();
    let mut history_avg_scores: Vec<f32> = Vec::new();
    let mut total_avg_accum: f32 = 0.0;
    let mut total_generations_completed: u32 = 0;

    // Initialize agents (100 agents)
    for _ in 0..100 {
        agents.push(Agent::new(Brain::random(), random_pastel_color()));
    }

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

                // Handle inputs
                let has_save = std::path::Path::new("../save_brain.txt").exists();
                let (mx, my) = mouse_position();
                let screen_w = screen_width();
                let screen_h = screen_height();
                let btn_w = 280.0;
                let btn_h = 50.0;

                // Buttons layout: 3 buttons if has_save, 2 buttons otherwise
                let (btn1_x, btn1_y, btn2_x, btn2_y, btn3_x, btn3_y) = if has_save {
                    (
                        (screen_w - btn_w) / 2.0, screen_h * 0.50,
                        (screen_w - btn_w) / 2.0, screen_h * 0.60,
                        (screen_w - btn_w) / 2.0, screen_h * 0.70,
                    )
                } else {
                    (
                        (screen_w - btn_w) / 2.0, screen_h * 0.55,
                        (screen_w - btn_w) / 2.0, screen_h * 0.66,
                        0.0, 0.0,
                    )
                };

                let hover1 = mx >= btn1_x && mx <= btn1_x + btn_w && my >= btn1_y && my <= btn1_y + btn_h;
                let hover2 = mx >= btn2_x && mx <= btn2_x + btn_w && my >= btn2_y && my <= btn2_y + btn_h;
                let hover3 = has_save && mx >= btn3_x && mx <= btn3_x + btn_w && my >= btn3_y && my <= btn3_y + btn_h;

                if has_save {
                    // Option 1: Resume training (Key R or Button 1 click)
                    if is_key_pressed(KeyCode::R) || (hover1 && is_mouse_button_pressed(MouseButton::Left)) {
                        if let Ok(data) = load_game("save_brainv1.txt") {
                            state = State::AIPlaying;
                            history_max_scores.clear();
                            history_avg_scores.clear();
                            total_avg_accum = 0.0;
                            total_generations_completed = 0;
                            generation = data.generation;
                            high_score = data.high_score;
                            last_best_score = data.last_best_score;
                            last_best_fitness = data.last_best_fitness;
                            
                            // Initialize population with loaded brain (100 agents, top 3 exact clones)
                            agents.clear();
                            agents.push(Agent::new(data.brain.clone(), color_bird_body()));
                            agents.push(Agent::new(data.brain.clone(), random_pastel_color()));
                            agents.push(Agent::new(data.brain.clone(), random_pastel_color()));
                            for _ in 0..97 {
                                agents.push(Agent::new(data.brain.mutate(0.15), random_pastel_color()));
                            }
                            
                            pipes.clear();
                            particles.clear();
                            pipe_spawn_timer = SPAWN_INTERVAL;
                            score = 0;
                            speed_multiplier = 1;
                        }
                    }
                    // Option 2: Start new training (Key Enter or Button 2 click)
                    else if is_key_pressed(KeyCode::Enter) || (hover2 && is_mouse_button_pressed(MouseButton::Left)) {
                        state = State::AIPlaying;
                        history_max_scores.clear();
                        history_avg_scores.clear();
                        total_avg_accum = 0.0;
                        total_generations_completed = 0;
                        agents.clear();
                        for _ in 0..100 {
                            agents.push(Agent::new(Brain::random(), random_pastel_color()));
                        }
                        generation = 1;
                        pipes.clear();
                        particles.clear();
                        pipe_spawn_timer = SPAWN_INTERVAL;
                        score = 0;
                        speed_multiplier = 1;
                    }
                    // Option 3: Play manually (Key Space or Button 3 click)
                    else if is_key_pressed(KeyCode::Space) || (hover3 && is_mouse_button_pressed(MouseButton::Left)) {
                        state = State::PlayerPlaying;
                        bird = Bird::new();
                        pipes.clear();
                        particles.clear();
                        pipe_spawn_timer = SPAWN_INTERVAL;
                        score = 0;
                        bird.jump(&mut particles, color_bird_body());
                    }
                } else {
                    // Classic 2-button layout (Option 1: New IA, Option 2: Player)
                    if is_key_pressed(KeyCode::Enter) || (hover1 && is_mouse_button_pressed(MouseButton::Left)) {
                        state = State::AIPlaying;
                        history_max_scores.clear();
                        history_avg_scores.clear();
                        total_avg_accum = 0.0;
                        total_generations_completed = 0;
                        agents.clear();
                        for _ in 0..100 {
                            agents.push(Agent::new(Brain::random(), random_pastel_color()));
                        }
                        generation = 1;
                        pipes.clear();
                        particles.clear();
                        pipe_spawn_timer = SPAWN_INTERVAL;
                        score = 0;
                        speed_multiplier = 1;
                    } else if is_key_pressed(KeyCode::Space) || (hover2 && is_mouse_button_pressed(MouseButton::Left)) {
                        state = State::PlayerPlaying;
                        bird = Bird::new();
                        pipes.clear();
                        particles.clear();
                        pipe_spawn_timer = SPAWN_INTERVAL;
                        score = 0;
                        bird.jump(&mut particles, color_bird_body());
                    }
                }
            }
            State::PlayerPlaying => {
                // Update clouds
                for cloud in &mut clouds {
                    cloud.update(PHYSICS_DT);
                }

                // Scroll ground
                ground_scroll_x = (ground_scroll_x - PIPE_SPEED * PHYSICS_DT) % 30.0;

                // Bird physics
                bird.update(&state);

                // Jump input
                if is_key_pressed(KeyCode::Space) || is_mouse_button_pressed(MouseButton::Left) {
                    bird.jump(&mut particles, color_bird_body());
                }

                // Spawn pipes
                pipe_spawn_timer += PHYSICS_DT;
                if pipe_spawn_timer >= SPAWN_INTERVAL {
                    pipe_spawn_timer = 0.0;
                    let mut rng = ::rand::rng();
                    let min_y = 150.0;
                    let max_y = (screen_height() - GROUND_HEIGHT - 150.0).max(min_y + 10.0);
                    let gap_y = rng.random_range(min_y..max_y);
                    pipes.push(Pipe::new(screen_width() + 50.0, gap_y));
                }

                // Update pipes
                for pipe in &mut pipes {
                    pipe.update(PHYSICS_DT);

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
                
                // 1. Collides with ground or ceiling
                if bird.y + bird.radius >= ground_y || bird.y - bird.radius <= 0.0 {
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
            State::AIPlaying => {
                // Read inputs for speed control (handles both numeric row and numpad from 0 to 9)
                if is_key_pressed(KeyCode::Key0) || is_key_pressed(KeyCode::Kp0) { speed_multiplier = 1; }
                if is_key_pressed(KeyCode::Key1) || is_key_pressed(KeyCode::Kp1) { speed_multiplier = 2; }
                if is_key_pressed(KeyCode::Key2) || is_key_pressed(KeyCode::Kp2) { speed_multiplier = 5; }
                if is_key_pressed(KeyCode::Key3) || is_key_pressed(KeyCode::Kp3) { speed_multiplier = 10; }
                if is_key_pressed(KeyCode::Key4) || is_key_pressed(KeyCode::Kp4) { speed_multiplier = 25; }
                if is_key_pressed(KeyCode::Key5) || is_key_pressed(KeyCode::Kp5) { speed_multiplier = 50; }
                if is_key_pressed(KeyCode::Key6) || is_key_pressed(KeyCode::Kp6) { speed_multiplier = 100; }
                if is_key_pressed(KeyCode::Key7) || is_key_pressed(KeyCode::Kp7) { speed_multiplier = 250; }
                if is_key_pressed(KeyCode::Key8) || is_key_pressed(KeyCode::Kp8) { speed_multiplier = 500; }
                if is_key_pressed(KeyCode::Key9) || is_key_pressed(KeyCode::Kp9) { speed_multiplier = 1000; }

                for _ in 0..speed_multiplier {
                    // Skip cosmetic updates at high speed — no impact on simulation
                    if speed_multiplier == 1 {
                        for cloud in &mut clouds {
                            cloud.update(PHYSICS_DT);
                        }
                        ground_scroll_x = (ground_scroll_x - PIPE_SPEED * PHYSICS_DT) % 30.0;
                    }

                    // Spawn pipes
                    pipe_spawn_timer += PHYSICS_DT;
                    if pipe_spawn_timer >= SPAWN_INTERVAL {
                        pipe_spawn_timer = 0.0;
                        let mut rng = ::rand::rng();
                        let min_y = 150.0;
                        let max_y = (screen_height() - GROUND_HEIGHT - 150.0).max(min_y + 10.0);
                        let gap_y = rng.random_range(min_y..max_y);
                        pipes.push(Pipe::new(screen_width() + 50.0, gap_y));
                    }

                    // Get next pipe (snapshot values to allow sharing across threads)
                    let pipe_snapshot: Option<(f32, f32)> = pipes
                        .iter()
                        .find(|p| p.x + PIPE_WIDTH - 30.0 > BIRD_X)
                        .map(|p| (p.x, p.gap_y));

                    // Get the pipe AFTER next (for anticipation)
                    let pipe_after_snapshot: Option<(f32, f32)> = {
                        let mut it = pipes.iter().filter(|p| p.x + PIPE_WIDTH - 30.0 > BIRD_X);
                        it.next(); // skip current
                        it.next().map(|p| (p.x, p.gap_y))
                    };

                    // Pre-compute shared read-only values
                    let ground_y = screen_height() - GROUND_HEIGHT;
                    let sw = screen_width();
                    let sh = screen_height();

                    // --- PARALLEL UPDATE: all 100 agents processed simultaneously ---
                    agents.par_iter_mut().for_each(|agent| {
                        if !agent.alive {
                            return;
                        }

                        // Normalize inputs for current pipe
                        let (dx, dy) = if let Some((px, py)) = pipe_snapshot {
                            (((px - BIRD_X) / sw).max(0.0), (py - agent.bird.y) / 200.0)
                        } else {
                            (1.0, (sh / 2.0 - agent.bird.y) / 200.0)
                        };

                        // Normalize inputs for next pipe (anticipation context)
                        let (dx_next, dy_next) = if let Some((px2, py2)) = pipe_after_snapshot {
                            (((px2 - BIRD_X) / sw).max(0.0), (py2 - agent.bird.y) / 200.0)
                        } else if let Some((px, py)) = pipe_snapshot {
                            // No second pipe yet — replicate current pipe data as neutral signal
                            (((px - BIRD_X) / sw).max(0.0) + 0.5, (py - agent.bird.y) / 200.0)
                        } else {
                            (1.0, 0.0)
                        };

                        // Pure survival reward — no center penalty, the bird passes where it wants
                        agent.fitness_score += PHYSICS_DT;

                        // Neural network query (5 inputs)
                        if agent.brain.predict(dy, dx, agent.bird.velocity / 10.0, dy_next, dx_next) {
                            agent.bird.velocity = JUMP_FORCE;
                        }

                        agent.bird.update(&State::AIPlaying);

                        // Collisions: ground/ceiling
                        if agent.bird.y + agent.bird.radius >= ground_y
                            || agent.bird.y - agent.bird.radius <= 0.0
                        {
                            agent.alive = false;
                        }

                        // Collisions: pipe (using snapshot to avoid borrow checker issues)
                        if let Some((px, py)) = pipe_snapshot {
                            let pipe_left = px;
                            let pipe_right = px + PIPE_WIDTH;
                            let gap_top = py - PIPE_GAP / 2.0;
                            let bottom_py = py + PIPE_GAP / 2.0;

                            let collision_radius = agent.bird.radius * 0.75;
                            let r_sq = collision_radius * collision_radius;

                            let cx1 = BIRD_X.max(pipe_left).min(pipe_right);
                            let cy1 = agent.bird.y.max(0.0).min(gap_top);
                            let dist_sq1 = (BIRD_X - cx1) * (BIRD_X - cx1) + (agent.bird.y - cy1) * (agent.bird.y - cy1);

                            let cx2 = BIRD_X.max(pipe_left).min(pipe_right);
                            let cy2 = agent.bird.y.max(bottom_py).min(ground_y);
                            let dist_sq2 = (BIRD_X - cx2) * (BIRD_X - cx2) + (agent.bird.y - cy2) * (agent.bird.y - cy2);

                            if dist_sq1 < r_sq || dist_sq2 < r_sq {
                                agent.alive = false;
                            }
                        }
                    });
                    // --- END PARALLEL UPDATE ---


                    // Update pipes and scores
                    let mut score_increased = false;
                    for pipe in &mut pipes {
                        pipe.update(PHYSICS_DT);

                        if !pipe.passed && pipe.x + PIPE_WIDTH / 2.0 < BIRD_X {
                            pipe.passed = true;
                            score_increased = true;
                        }
                    }

                    if score_increased {
                        for agent in &mut agents {
                            if agent.alive {
                                agent.score += 1;
                            }
                        }
                    }

                    // Remove off-screen pipes
                    pipes.retain(|pipe| pipe.x + PIPE_WIDTH + 10.0 > 0.0);

                    // If all agents are dead, select the top 3 and reproduce!
                    let all_dead = agents.iter().all(|a| !a.alive);
                    if all_dead {
                        // Sort agents by fitness descending to select top 3
                        agents.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

                        let best_agent = &agents[0];
                        let best_brain = best_agent.brain.clone();
                        let best_score = best_agent.score;
                        let best_fitness = best_agent.fitness();

                        // Calculate average score and push to history
                        let avg_score = agents.iter().map(|a| a.score).sum::<u32>() as f32 / agents.len() as f32;
                        history_max_scores.push(best_score);
                        history_avg_scores.push(avg_score);
                        if history_max_scores.len() > 50 {
                            history_max_scores.remove(0);
                            history_avg_scores.remove(0);
                        }

                        // Accumulate for global historical average of generation max scores
                        total_avg_accum += best_score as f32;
                        total_generations_completed += 1;

                        last_best_score = best_score;
                        last_best_fitness = best_fitness;
                        if best_score > high_score {
                            high_score = best_score;
                        }

                        generation += 1;

                        // Save best agent's brain to file automatically!
                        if let Err(e) = save_game(
                            "save_brainv1.txt",
                            generation,
                            high_score,
                            last_best_score,
                            last_best_fitness,
                            &best_brain,
                        ) {
                            println!("Error saving: {:?}", e);
                        }

                        // Extract top 10 brains (10% of population)
                        let top_brains: Vec<Brain> = agents[0..10]
                            .iter()
                            .map(|a| a.brain.clone())
                            .collect();

                        agents.clear();

                        // Elitism: clone the top 10 brains exactly (best keeps its color)
                        agents.push(Agent::new(top_brains[0].clone(), color_bird_body()));
                        for i in 1..10 {
                            agents.push(Agent::new(top_brains[i].clone(), random_pastel_color()));
                        }

                        // Generate the remaining 90 mutated brains from the top 10
                        // 30 Refiners (5% rate)  — fine micro-tuning
                        // 45 Adapters (15% rate) — standard evolution
                        // 15 Explorers (35% rate) — radical jumps to escape local minima
                        let mut rng = ::rand::rng();
                        for i in 0..90 {
                            let parent_idx = rng.random_range(0..10);

                            let mutation_rate = if i < 30 {
                                0.05
                            } else if i < 75 {
                                0.15
                            } else {
                                0.35
                            };

                            let mutated_brain = top_brains[parent_idx].mutate(mutation_rate);
                            agents.push(Agent::new(mutated_brain, random_pastel_color()));
                        }

                        pipes.clear();
                        particles.clear();
                        pipe_spawn_timer = SPAWN_INTERVAL;
                        break; // exit fast forward loop early to reset on a fresh frame
                    }
                }
            }
            State::GameOver => {
                // Update clouds (slowly)
                for cloud in &mut clouds {
                    cloud.update(dt * 0.2);
                }

                bird.update(&state);

                // Restart input: Space or R or Enter goes back to start menu
                if is_key_pressed(KeyCode::Space) || is_key_pressed(KeyCode::R) || is_key_pressed(KeyCode::Enter) {
                    state = State::Start;
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

        // 4. Draw Birds
        if state == State::AIPlaying {
            let best_alive = agents.iter()
                .filter(|a| a.alive)
                .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

            // Draw Sensory Laser Guides for the Leading Bird
            if let Some(agent) = best_alive {
                let next_pipe = pipes.iter().find(|p| p.x + PIPE_WIDTH - 30.0 > BIRD_X);
                
                // Target position (gap center or horizontal target)
                let (target_x, target_y) = if let Some(pipe) = next_pipe {
                    (pipe.x + PIPE_WIDTH / 2.0, pipe.gap_y)
                } else {
                    (BIRD_X + 180.0, screen_height() / 2.0)
                };
                
                // Differentiate laser color dynamically based on danger level
                let dist_x = target_x - BIRD_X;
                let laser_color = if next_pipe.is_some() && dist_x < 140.0 {
                    let a = (120.0 + (get_time() * 12.0).sin() as f32 * 80.0) as u8;
                    Color::from_rgba(255, 60, 60, a) // Warning Red
                } else {
                    let a = (80.0 + (get_time() * 6.0).sin() as f32 * 40.0) as u8;
                    Color::from_rgba(0, 230, 255, a) // Safe Cyan
                };
                
                // Draw dotted laser line
                let dot_count = 20;
                for i in 0..dot_count {
                    let t1 = i as f32 / dot_count as f32;
                    let t2 = (i as f32 + 0.6) / dot_count as f32;
                    let lx1 = BIRD_X + (target_x - BIRD_X) * t1;
                    let ly1 = agent.bird.y + (target_y - agent.bird.y) * t1;
                    let lx2 = BIRD_X + (target_x - BIRD_X) * t2;
                    let ly2 = agent.bird.y + (target_y - agent.bird.y) * t2;
                    draw_line(lx1, ly1, lx2, ly2, 2.5, laser_color);
                }
                
                // Locking Target Crosshair
                if next_pipe.is_some() {
                    let pulse = 4.0 + (get_time() * 8.0).sin() as f32 * 2.0;
                    let c_color = Color::from_rgba(0, 255, 255, 180);
                    draw_circle_lines(target_x, target_y, pulse, 2.0, c_color);
                    draw_line(target_x - pulse - 3.0, target_y, target_x + pulse + 3.0, target_y, 1.5, c_color);
                    draw_line(target_x, target_y - pulse - 3.0, target_x, target_y + pulse + 3.0, 1.5, c_color);
                }
            }

            // Draw all alive birds
            for agent in &agents {
                if agent.alive {
                    agent.bird.draw(agent.color);
                }
            }

            // Draw Champion highlights (neon aura and golden crown) on top of the champion
            if let Some(agent) = best_alive {
                let time = get_time();
                let pulse = (time * 5.0).sin() as f32;
                
                // Neon rings
                let ring1 = BIRD_RADIUS + 5.0 + pulse * 2.0;
                let ring2 = BIRD_RADIUS + 9.0 - pulse * 2.0;
                draw_circle_lines(BIRD_X, agent.bird.y, ring1, 2.0, Color::from_rgba(0, 255, 255, 150));
                draw_circle_lines(BIRD_X, agent.bird.y, ring2, 1.0, Color::from_rgba(0, 255, 255, 80));
                
                // Hovering gold crown
                let crown_y = agent.bird.y - BIRD_RADIUS - 10.0 + (time * 4.0).cos() as f32 * 2.0;
                draw_triangle(
                    vec2(BIRD_X - 6.0, crown_y),
                    vec2(BIRD_X - 9.0, crown_y - 8.0),
                    vec2(BIRD_X - 3.0, crown_y - 4.0),
                    Color::from_rgba(255, 215, 0, 255)
                );
                draw_triangle(
                    vec2(BIRD_X - 3.0, crown_y - 4.0),
                    vec2(BIRD_X, crown_y - 12.0),
                    vec2(BIRD_X + 3.0, crown_y - 4.0),
                    Color::from_rgba(255, 215, 0, 255)
                );
                draw_triangle(
                    vec2(BIRD_X + 3.0, crown_y - 4.0),
                    vec2(BIRD_X + 9.0, crown_y - 8.0),
                    vec2(BIRD_X + 6.0, crown_y),
                    Color::from_rgba(255, 215, 0, 255)
                );
                draw_line(BIRD_X - 6.0, crown_y, BIRD_X + 6.0, crown_y, 2.0, Color::from_rgba(255, 215, 0, 255));
                
                // "CHAMPION" tag bouncing
                draw_text("CHAMPION", BIRD_X - 25.0, crown_y - 5.0, 11.0, Color::from_rgba(255, 255, 120, 255));
            }
        } else {
            bird.draw(color_bird_body());
        }

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
                let title_text = "FLAPPY BIRD AI";
                let font_size = 60.0;
                let text_w = measure_text(title_text, None, font_size as u16, 1.0).width;
                draw_text_shadow(title_text, (screen_w - text_w) / 2.0, 120.0, font_size, WHITE);

                // Subtitle (AI and Player selection menu)
                let has_save = std::path::Path::new("../save_brainv1.txt").exists();
                let (mx, my) = mouse_position();
                let btn_w = 280.0;
                let btn_h = 50.0;

                let (btn1_x, btn1_y, btn2_x, btn2_y, btn3_x, btn3_y) = if has_save {
                    (
                        (screen_w - btn_w) / 2.0, screen_h * 0.50,
                        (screen_w - btn_w) / 2.0, screen_h * 0.60,
                        (screen_w - btn_w) / 2.0, screen_h * 0.70,
                    )
                } else {
                    (
                        (screen_w - btn_w) / 2.0, screen_h * 0.55,
                        (screen_w - btn_w) / 2.0, screen_h * 0.66,
                        0.0, 0.0,
                    )
                };

                let hover1 = mx >= btn1_x && mx <= btn1_x + btn_w && my >= btn1_y && my <= btn1_y + btn_h;
                let hover2 = mx >= btn2_x && mx <= btn2_x + btn_w && my >= btn2_y && my <= btn2_y + btn_h;
                let hover3 = has_save && mx >= btn3_x && mx <= btn3_x + btn_w && my >= btn3_y && my <= btn3_y + btn_h;

                if has_save {
                    // Draw Button 1: Resume IA
                    let btn1_color = if hover1 { Color::from_rgba(100, 220, 255, 255) } else { Color::from_rgba(50, 150, 250, 255) };
                    draw_rectangle(btn1_x, btn1_y, btn_w, btn_h, btn1_color);
                    draw_rectangle_lines(btn1_x, btn1_y, btn_w, btn_h, 3.0, color_pipe_dark());
                    let t1 = "REPRENDRE L'IA [TOUCHE R]";
                    let tw1 = measure_text(t1, None, 20, 1.0).width;
                    draw_text(t1, btn1_x + (btn_w - tw1) / 2.0, btn1_y + 32.0, 20.0, WHITE);

                    // Draw Button 2: New IA Mode
                    let btn2_color = if hover2 { Color::from_rgba(100, 220, 100, 255) } else { Color::from_rgba(115, 191, 46, 255) };
                    draw_rectangle(btn2_x, btn2_y, btn_w, btn_h, btn2_color);
                    draw_rectangle_lines(btn2_x, btn2_y, btn_w, btn_h, 3.0, color_pipe_dark());
                    let t2 = "NOUVELLE IA [ENTRÉE]";
                    let tw2 = measure_text(t2, None, 20, 1.0).width;
                    draw_text(t2, btn2_x + (btn_w - tw2) / 2.0, btn2_y + 32.0, 20.0, WHITE);

                    // Draw Button 3: Player Mode
                    let btn3_color = if hover3 { Color::from_rgba(250, 200, 50, 255) } else { color_bird_body() };
                    draw_rectangle(btn3_x, btn3_y, btn_w, btn_h, btn3_color);
                    draw_rectangle_lines(btn3_x, btn3_y, btn_w, btn_h, 3.0, color_pipe_dark());
                    let t3 = "JOUER SOI-MÊME [ESPACE]";
                    let tw3 = measure_text(t3, None, 20, 1.0).width;
                    draw_text(t3, btn3_x + (btn_w - tw3) / 2.0, btn3_y + 32.0, 20.0, WHITE);
                } else {
                    // Draw Button 1: IA Mode
                    let btn1_color = if hover1 { Color::from_rgba(100, 220, 100, 255) } else { Color::from_rgba(115, 191, 46, 255) };
                    draw_rectangle(btn1_x, btn1_y, btn_w, btn_h, btn1_color);
                    draw_rectangle_lines(btn1_x, btn1_y, btn_w, btn_h, 3.0, color_pipe_dark());
                    let t1 = "LANCER L'IA [ENTRÉE]";
                    let tw1 = measure_text(t1, None, 20, 1.0).width;
                    draw_text(t1, btn1_x + (btn_w - tw1) / 2.0, btn1_y + 32.0, 20.0, WHITE);

                    // Draw Button 2: Player Mode
                    let btn2_color = if hover2 { Color::from_rgba(250, 200, 50, 255) } else { color_bird_body() };
                    draw_rectangle(btn2_x, btn2_y, btn_w, btn_h, btn2_color);
                    draw_rectangle_lines(btn2_x, btn2_y, btn_w, btn_h, 3.0, color_pipe_dark());
                    let t2 = "JOUER SOI-MÊME [ESPACE]";
                    let tw2 = measure_text(t2, None, 20, 1.0).width;
                    draw_text(t2, btn2_x + (btn_w - tw2) / 2.0, btn2_y + 32.0, 20.0, WHITE);
                }

                // Footer
                let footer_text = "Développé avec Macroquad & Algorithme Génétique";
                let font_size_foot = 18.0;
                let text_foot_w = measure_text(footer_text, None, font_size_foot as u16, 1.0).width;
                draw_text_shadow(footer_text, (screen_w - text_foot_w) / 2.0, screen_h - 40.0, font_size_foot, Color::from_rgba(240, 240, 240, 220));
            }
            State::PlayerPlaying => {
                // Large score in the center
                let score_str = score.to_string();
                let font_size = 72.0;
                let text_w = measure_text(&score_str, None, font_size as u16, 1.0).width;
                draw_text_shadow(&score_str, (screen_w - text_w) / 2.0, 100.0, font_size, WHITE);
            }
            State::AIPlaying => {
                // Top-Center HUD Dashboard (Glassmorphic & Cyberpunk style)
                let hud_w = 400.0;
                let hud_h = 55.0;
                let hud_x = (screen_w - hud_w) / 2.0;
                let hud_y = 20.0;

                // Glassmorphism HUD background
                draw_rectangle(hud_x, hud_y, hud_w, hud_h, Color::from_rgba(255, 255, 255, 30));
                draw_rectangle(hud_x, hud_y, hud_w, hud_h, Color::from_rgba(0, 0, 0, 110)); // Translucent dark
                draw_rectangle_lines(hud_x, hud_y, hud_w, hud_h, 2.5, Color::from_rgba(255, 255, 255, 80));

                // Stats inside HUD
                let hud_center_y = hud_y + 33.0;

                // 1. RECORD ABSOLU (Gold)
                draw_text("🏆 RECORD:", hud_x + 15.0, hud_center_y, 14.0, Color::from_rgba(255, 215, 0, 255));
                draw_text(&high_score.to_string(), hud_x + 95.0, hud_center_y, 16.0, Color::from_rgba(255, 215, 0, 255));

                // Separator
                draw_line(hud_x + 130.0, hud_y + 12.0, hud_x + 130.0, hud_y + hud_h - 12.0, 1.5, Color::from_rgba(255, 255, 255, 40));

                // 2. MAX ACTUEL (Cyan)
                let current_max_score = agents.iter().filter(|a| a.alive).map(|a| a.score).max().unwrap_or(0);
                draw_text("⚡ MAX:", hud_x + 145.0, hud_center_y, 14.0, Color::from_rgba(100, 230, 255, 255));
                draw_text(&current_max_score.to_string(), hud_x + 195.0, hud_center_y, 16.0, Color::from_rgba(100, 230, 255, 255));

                // Separator
                draw_line(hud_x + 235.0, hud_y + 12.0, hud_x + 235.0, hud_y + hud_h - 12.0, 1.5, Color::from_rgba(255, 255, 255, 40));

                // 3. SCORE MOYEN HISTORIQUE (Orange)
                let global_avg_score = if total_generations_completed > 0 {
                    total_avg_accum / total_generations_completed as f32
                } else {
                    0.0
                };
                draw_text("📈 MOY. RECORDS:", hud_x + 245.0, hud_center_y, 13.0, Color::from_rgba(255, 165, 0, 255));
                draw_text(&format!("{:.1}", global_avg_score), hud_x + 360.0, hud_center_y, 15.0, Color::from_rgba(255, 165, 0, 255));

                // Glassmorphic side panel on the right for AI Telemetry
                let panel_w = 260.0;
                let panel_h = 450.0;
                let panel_x = screen_w - panel_w - 20.0;
                let panel_y = 20.0;

                // Draw Panel Background (Glassmorphism effect)
                draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::from_rgba(255, 255, 255, 30));
                draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::from_rgba(0, 0, 0, 100)); // Dark overlay
                draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 3.0, Color::from_rgba(255, 255, 255, 80));

                // Title
                draw_text("TÉLÉMÉTRIE DE L'IA", panel_x + 15.0, panel_y + 35.0, 22.0, Color::from_rgba(100, 230, 255, 255));
                draw_line(panel_x + 15.0, panel_y + 45.0, panel_x + panel_w - 15.0, panel_y + 45.0, 2.0, Color::from_rgba(255, 255, 255, 40));

                // Stats list helper
                let draw_stat = |label: &str, val: &str, y_offset: f32, color: Color| {
                    draw_text(label, panel_x + 15.0, panel_y + y_offset, 18.0, Color::from_rgba(220, 220, 220, 255));
                    let val_w = measure_text(val, None, 18, 1.0).width;
                    draw_text(val, panel_x + panel_w - 15.0 - val_w, panel_y + y_offset, 18.0, color);
                };

                draw_stat("Génération :", &generation.to_string(), 75.0, WHITE);
                
                let speed_str = format!("{}x", speed_multiplier);
                let speed_color = match speed_multiplier {
                    1 => WHITE,
                    2 => Color::from_rgba(100, 255, 100, 255),
                    5 => Color::from_rgba(255, 200, 50, 255),
                    _ => Color::from_rgba(255, 100, 100, 255),
                };
                draw_stat("Vitesse :", &speed_str, 101.0, speed_color);
                
                let alive_count = agents.iter().filter(|a| a.alive).count();
                draw_stat("Oiseaux en vie :", &format!("{}/100", alive_count), 127.0, Color::from_rgba(100, 255, 100, 255));
                
                let current_max_score = agents.iter().filter(|a| a.alive).map(|a| a.score).max().unwrap_or(0);
                draw_stat("Score actuel :", &current_max_score.to_string(), 153.0, WHITE);
                
                draw_stat("Score moyen (max) :", &format!("{:.1}", global_avg_score), 179.0, Color::from_rgba(255, 200, 50, 255));
                
                draw_stat("Dernier score :", &last_best_score.to_string(), 205.0, Color::from_rgba(200, 200, 200, 255));
                draw_stat("Dernière fitness :", &format!("{:.0}", last_best_fitness), 231.0, Color::from_rgba(180, 220, 255, 255));
                draw_stat("Record absolu :", &high_score.to_string(), 257.0, Color::from_rgba(255, 215, 0, 255));

                // Speed controls help text
                draw_line(panel_x + 15.0, panel_y + 268.0, panel_x + panel_w - 15.0, panel_y + 268.0, 1.0, Color::from_rgba(255, 255, 255, 25));
                draw_text("Vitesse : Touches 0-9 (Pavé incl.)", panel_x + 15.0, panel_y + 290.0, 14.0, Color::from_rgba(200, 200, 200, 200));

                // List of agents
                draw_text("AGENTS :", panel_x + 15.0, panel_y + 320.0, 16.0, Color::from_rgba(220, 220, 220, 255));
                let agent_dot_start_y = panel_y + 343.0;
                
                for i in 0..100 {
                    let col_idx = (i % 10) as f32;
                    let row_idx = (i / 10) as f32;
                    let dot_x = panel_x + 25.0 + col_idx * 23.0;
                    let dot_y = agent_dot_start_y + row_idx * 8.0;
                    
                    let agent = &agents[i];
                    
                    if !agent.alive {
                        // Draw tiny red dot for dead agents
                        draw_circle(dot_x, dot_y, 2.0, RED);
                    } else {
                        // Draw colored dot for alive agents
                        draw_circle(dot_x, dot_y, 4.0, agent.color);
                    }
                }

                // Neural Network Visualizer on the right side underneath AI Telemetry
                let best_alive = agents.iter()
                    .filter(|a| a.alive)
                    .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
                if let Some(agent) = best_alive {
                    // Current pipe inputs
                    let next_pipe = pipes.iter().find(|p| p.x + PIPE_WIDTH - 30.0 > BIRD_X);
                    let (dx, dy) = if let Some(pipe) = next_pipe {
                        (((pipe.x - BIRD_X) / screen_width()).max(0.0), (pipe.gap_y - agent.bird.y) / 200.0)
                    } else {
                        (1.0, (screen_height() / 2.0 - agent.bird.y) / 200.0)
                    };

                    // Next pipe inputs (anticipation)
                    let pipe_after = {
                        let mut it = pipes.iter().filter(|p| p.x + PIPE_WIDTH - 30.0 > BIRD_X);
                        it.next();
                        it.next()
                    };
                    let (dx_next, dy_next) = if let Some(pipe2) = pipe_after {
                        (((pipe2.x - BIRD_X) / screen_width()).max(0.0), (pipe2.gap_y - agent.bird.y) / 200.0)
                    } else if let Some(pipe) = next_pipe {
                        (((pipe.x - BIRD_X) / screen_width()).max(0.0) + 0.5, (pipe.gap_y - agent.bird.y) / 200.0)
                    } else {
                        (1.0, 0.0)
                    };
                    
                    draw_neuron_map(
                        panel_x,
                        panel_y + panel_h + 15.0,
                        panel_w,
                        240.0,
                        &agent.brain,
                        dy,
                        dx,
                        agent.bird.velocity / 10.0,
                        dy_next,
                        dx_next,
                    );
                }


                // Render Evolution Progress Graph on the right underneath Vision Neuronale
                draw_evolution_chart(
                    panel_x,
                    panel_y + panel_h + 15.0 + 240.0 + 15.0,
                    panel_w,
                    210.0,
                    &history_max_scores,
                    &history_avg_scores,
                );
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
                let restart_text = "ESPACE ou ENTRÉE pour le menu";
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