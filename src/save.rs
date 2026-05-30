use crate::brain::Brain;

pub struct SaveData {
    pub generation: u32,
    pub high_score: u32,
    pub last_best_score: u32,
    pub last_best_fitness: f32,
    pub brain: Brain,
}

pub fn save_game(
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

pub fn load_game(filename: &str) -> std::io::Result<SaveData> {
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
