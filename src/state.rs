#[derive(PartialEq, Clone, Copy)]
pub enum State {
    Start,
    PlayerPlaying,
    AIPlaying,
    GameOver,
}
