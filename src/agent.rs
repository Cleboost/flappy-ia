use macroquad::prelude::Color;
use ::rand::RngExt;
use crate::brain::Brain;
use crate::game_objects::Bird;

pub struct Agent {
    pub bird: Bird,
    pub brain: Brain,
    pub alive: bool,
    pub score: u32,
    pub fitness_score: f32,
    pub color: Color,
}

impl Agent {
    pub fn new(brain: Brain, color: Color) -> Self {
        Self {
            bird: Bird::new(),
            brain,
            alive: true,
            score: 0,
            fitness_score: 0.0,
            color,
        }
    }

    pub fn fitness(&self) -> f32 {
        self.fitness_score + (self.score as f32) * 1000.0
    }
}

pub fn random_pastel_color() -> Color {
    let mut rng = ::rand::rng();
    let r = rng.random_range(0.4..1.0);
    let g = rng.random_range(0.4..1.0);
    let b = rng.random_range(0.4..1.0);
    Color::new(r, g, b, 1.0)
}
