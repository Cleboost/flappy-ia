use ::rand::RngExt;

#[derive(Clone)]
pub struct Brain {
    // 5 inputs (dy_cur, dx_cur, velocity, dy_next, dx_next) -> 5 hidden neurons
    pub weights_ih: [[f32; 5]; 5],
    pub bias_h: [f32; 5],
    // 5 hidden -> 1 output neuron
    pub weights_ho: [f32; 5],
    pub bias_o: f32,
}

impl Brain {
    pub fn random() -> Self {
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

    pub fn crossover(&self, other: &Self) -> Self {
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

    pub fn mutate(&self, rate: f32) -> Self {
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

    pub fn predict(&self, dy: f32, dx: f32, vel: f32, dy_next: f32, dx_next: f32) -> bool {
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
