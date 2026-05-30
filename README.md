# Flappy Bird AI Premium

An interactive and highly visual AI simulation learning to play Flappy Bird in real-time using a genetic algorithm and an artificial neural network, developed in Rust with the Macroquad graphics library.

---

## Key Features

*   **Artificial Neural Network**: Each bird features a 5-input brain (dY to current pipe, dX to current pipe, vertical velocity, dY to next pipe, dX to next pipe) feeding into a hidden layer of 5 neurons and 1 output neuron controlling the jump.
*   **Genetic Evolution**: Elite selection mechanism carrying top-performing brains over generations combined with adaptive mutations (Fine-tuning, Standard evolution, and Radical exploration to break out of local minima).
*   **Rayon Parallel Processing**: Multithreaded simulation processing the entire population of 100 birds simultaneously using all available CPU cores.
*   **Simulation Time Acceleration**: Real-time adjustable speed multiplier from 1x up to 1000x to fast-forward the training phases.
*   **Premium Visual Telemetry**:
    *   **Neural Vision**: Live display of the leading champion's neural network, mapping out neuron activation and synaptic weight intensity.
    *   **Evolution Chart**: Real-time plotting of the record and average scores across the latest 50 generations.
    *   **Telemetry Panel**: Detailed dashboard displaying current generation, simulation speed, active population count, and historical score statistics.
*   **Autosave System**: Automatically saves the champion brain to save_brainv1.txt at the end of each generation, allowing you to resume training later.
*   **Playable Mode**: Take control of the bird yourself to challenge and test your skills against the trained AI.

---

## Project Architecture

The codebase is highly modularized for maximum readability and clean maintainability inside the src/ directory:

*   [`src/main.rs`](file:///home/cleboost/Code/flappy-ia/src/main.rs): Entry point orchestrating the main game loop, rendering, and simulation cycles.
*   [`src/config.rs`](file:///home/cleboost/Code/flappy-ia/src/config.rs): Physics configurations (gravity, jump force, speeds) and graphical color schemes.
*   [`src/state.rs`](file:///home/cleboost/Code/flappy-ia/src/state.rs): Game state machine (Start, PlayerPlaying, AIPlaying, GameOver).
*   [`src/brain.rs`](file:///home/cleboost/Code/flappy-ia/src/brain.rs): Neural network structures, activation functions (hyperbolic tangent), crossover, and mutation operations.
*   [`src/agent.rs`](file:///home/cleboost/Code/flappy-ia/src/agent.rs): Holds the evolutionary Agent struct combining physical bird simulation with its neural controller.
*   [`src/game_objects.rs`](file:///home/cleboost/Code/flappy-ia/src/game_objects.rs): Consolidates all interactive entities (Bird, Pipe, Cloud, Particle).
*   [`src/save.rs`](file:///home/cleboost/Code/flappy-ia/src/save.rs): Encapsulates serialization and deserialization routines to save or load top brains.
*   [`src/ui.rs`](file:///home/cleboost/Code/flappy-ia/src/ui.rs): Renders HUD components, the dynamic live neural visualizer, and the progress graph.

---

## Keyboard Controls

### Main Menu
*   **ENTER** (or Button click): Start a fresh AI training session.
*   **R Key** (or Button click): Resume training from the saved champion brain (if save_brainv1.txt exists).
*   **SPACE** (or Button click): Take manual control and play the game yourself.

### During AI Simulation (Speed Adjustment)
You can instantly change the speed of the training simulation by pressing number keys (Numpad included):
*   `0`: Standard Speed (1x)
*   `1`: 2x speedup
*   `2`: 5x speedup
*   `3`: 10x speedup
*   `4`: 25x speedup
*   `5`: 50x speedup
*   `6`: 100x speedup
*   `7`: 250x speedup
*   `8`: 500x speedup
*   `9`: Extreme speedup (1000x)

---

## How to Run

Make sure you have Rust installed.

### Compile and run the project in release mode (strongly recommended for AI performance):
```bash
cargo run --release
```
