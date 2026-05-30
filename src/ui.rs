use macroquad::prelude::*;
use crate::brain::Brain;

// --- REAL-TIME NEURAL NETWORK VISUALIZER ---
pub fn draw_neuron_map(
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
pub fn draw_evolution_chart(
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

// --- TEXT WITH SHADOW HELPER ---
pub fn draw_text_shadow(text: &str, x: f32, y: f32, size: f32, color: Color) {
    draw_text(text, x + 3.0, y + 3.0, size, Color::from_rgba(0, 0, 0, 150));
    draw_text(text, x, y, size, color);
}
