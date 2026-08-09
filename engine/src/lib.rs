#[cfg(test)]
pub mod test;
pub mod types;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub mod engine {
    use std::fmt::Debug;
    use pyo3::prelude::*;
    use rand::{SeedableRng, Rng, RngExt, seq::SliceRandom};
    use serde_json::Value;
    use serde::{Deserialize, Serialize};
    use super::types;

    pub trait SeedRng: SeedableRng + Rng + RngExt {}

    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PieceType {
        I = 0,
        O = 1,
        T = 2,
        S = 3,
        Z = 4,
        J = 5,
        L = 6,
    }

    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PieceRotation {
        Zero = 0,
        One = 1,
        Two = 2,
        Three = 3,
    }

    impl From<u32> for PieceRotation {
        fn from(value: u32) -> Self {
            match value % 4 {
                0 => PieceRotation::Zero,
                1 => PieceRotation::One,
                2 => PieceRotation::Two,
                3 => PieceRotation::Three,
                _ => unreachable!("Invalid rotation value."),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PieceShape {
        columns: [u32; 4],
        largest_width: u32,
        largest_height: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PieceRotations {
        shapes: [PieceShape; 4],
        size: usize,
    }

    pub static PIECE_SHAPES: [PieceRotations; 7] = [
        // long
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b10, 0b10, 0b10, 0b10],
                    largest_width: 4,
                    largest_height: 1,
                },
                PieceShape {
                    columns: [0b0, 0b0, 0b1111, 0b0],
                    largest_width: 1,
                    largest_height: 4,
                },
                PieceShape {
                    columns: [0b100, 0b100, 0b100, 0b100],
                    largest_width: 4,
                    largest_height: 1,
                },
                PieceShape {
                    columns: [0b0, 0b1111, 0b0, 0b0],
                    largest_width: 1,
                    largest_height: 4,
                },
            ],
            size: 4
        },

        // square
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b11, 0b11, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b11, 0b11, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b11, 0b11, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b11, 0b11, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
            ],
            size: 2
        },

        // hat
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b10, 0b11, 0b10, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b0, 0b111, 0b10, 0b0],
                    largest_width: 2,
                    largest_height: 3,
                },
                PieceShape {
                    columns: [0b10, 0b110, 0b10, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b10, 0b111, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 3,
                },
            ],
            size: 3
        },

        // right_snake
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b10, 0b11, 0b1, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b0, 0b11, 0b110, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b100, 0b110, 0b10, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b11, 0b110, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
            ],
            size: 3
        },

        // left_snake
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b1, 0b11, 0b10, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b0, 0b110, 0b11, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b10, 0b110, 0b100, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b110, 0b11, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 2,
                },
            ],
            size: 3
        },

        // left_gun
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b11, 0b10, 0b10, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b0, 0b111, 0b1, 0b0],
                    largest_width: 2,
                    largest_height: 3,
                },
                PieceShape {
                    columns: [0b10, 0b10, 0b110, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b100, 0b111, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 3,
                },
            ],
            size: 3
        },

        // right_gun
        PieceRotations {
            shapes: [
                PieceShape {
                    columns: [0b10, 0b10, 0b11, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b0, 0b111, 0b100, 0b0],
                    largest_width: 2,
                    largest_height: 3,
                },
                PieceShape {
                    columns: [0b110, 0b10, 0b10, 0b0],
                    largest_width: 3,
                    largest_height: 2,
                },
                PieceShape {
                    columns: [0b1, 0b111, 0b0, 0b0],
                    largest_width: 2,
                    largest_height: 3,
                },
            ],
            size: 3
        },
    ];

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Piece {
        shape: PieceType,
        rotation: PieceRotation,
        x: u32,
        y: u32,
        drop: u32,
    }

    impl Piece {
        pub fn new(shape: PieceType) -> Self {
            Self {
                shape,
                rotation: PieceRotation::Zero,
                x: match PIECE_SHAPES[shape as usize].size {
                    2 => 4,
                    _ => 3,
                },
                y: 0,
                drop: 0,
            }
        }

        pub fn random_piece<RNG: SeedRng>(rng: &mut RNG) -> Self {
            Self::new(match rng.random_range(0..7) {
                0 => PieceType::I,
                1 => PieceType::O,
                2 => PieceType::T,
                3 => PieceType::S,
                4 => PieceType::Z,
                5 => PieceType::J,
                _ => PieceType::L,
            })
        }

        pub fn mv(self, dx: i32, dy: i32) -> Self {
            Self {
                shape: self.shape,
                rotation: self.rotation,
                x: (self.x as i32 + dx) as u32,
                y: (self.y as i32 + dy) as u32,
                drop: self.drop,
            }
        }

        pub fn set_rotation(self, new_rotation: PieceRotation) -> Self {
            Self {
                shape: self.shape,
                rotation: new_rotation,
                x: self.x,
                y: self.y,
                drop: self.drop,
            }
        }

        pub fn rotate(self, dr: i32) -> Self {
            Self {
                shape: self.shape,
                rotation: PieceRotation::from((self.rotation as i32 + dr) as u32),
                x: self.x,
                y: self.y,
                drop: self.drop,
            }
        }

        pub fn get(self) -> &'static PieceShape {
            &PIECE_SHAPES[self.shape as usize].shapes[self.rotation as usize]
        }

        pub fn size(self) -> usize {
            PIECE_SHAPES[self.shape as usize].size
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum MovementFailureReason {
        OutOfBounds,
        Collision,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct Grid<const W: usize, const H: usize> {
        columns: [u32; W],
    }

    impl Default for Grid<10, 22> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<const W: usize, const H: usize> Grid<W, H> {
        pub fn new() -> Self {
            Self { columns: [0; W] }
        }

        pub fn print(&self) {
            let mut data = Vec::with_capacity(H);
            for _ in 0..H{
                data.push(Vec::<char>::with_capacity(W));
            }
            for column in self.columns {
                for (h, item) in data.iter_mut().enumerate() {
                    let value = (column >> h) & 0x1;
                    item.push(if value != 0 { 'X' } else { '.' });
                }
            }
            for row in data {
                println!("|{}|", row.iter().collect::<String>());
            }
        }

        pub fn in_bounds(&self, piece: &Piece) -> bool {
            let shape = piece.get();
            if piece.x as usize + (shape.largest_width - 1) as usize >= W || piece.y as usize >= H {
                return false;
            }
            if ((shape.largest_height - 1) as i32 + piece.y as i32) as usize >= H {
                return false;
            }
            true
        }

        pub fn fits(&self, piece: &Piece) -> Result<(), MovementFailureReason> {
            if !self.in_bounds(piece) {
                return Err(MovementFailureReason::OutOfBounds);
            }
            let shape = piece.get();
            let mut pos_x = piece.x as usize;
            let pos_y = piece.y as usize;

            for p_col in shape.columns {
                if p_col == 0 {
                    continue;
                }
                let p_col = p_col << pos_y;
                let g_col = self.columns[pos_x];
                if g_col & p_col != 0 {
                    return Err(MovementFailureReason::Collision);
                }
                pos_x += 1;
            }
            Ok(())
        }

        pub fn blend(&self, piece: &Piece) -> Result<Self, MovementFailureReason> {
            if !self.in_bounds(piece) {
                return Err(MovementFailureReason::OutOfBounds);
            }
            let mut grid = *self;
            let mut pos_x = piece.x as usize;
            for p_col in piece.get().columns {
                if p_col == 0 {
                    continue;
                }
                let p_col = p_col << piece.y;
                let g_col = self.columns[pos_x];
                if g_col & p_col != 0 {
                    return Err(MovementFailureReason::Collision);
                }
                grid.columns[pos_x] |= p_col;
                pos_x += 1
            }
            Ok(grid)
        }

        pub fn place_piece(&mut self, piece: &Piece) -> Result<(), MovementFailureReason> {
            *self = self.blend(piece)?;
            Ok(())
        }

        pub fn mv(&self, piece: &Piece, dx: i32, dy: i32) -> Result<Piece, MovementFailureReason> {
            let new_piece = piece.mv(dx, dy);
            self.fits(&new_piece)?;
            Ok(new_piece)
        }

        pub fn try_action_down(&self, piece: &Piece) -> Result<Piece, MovementFailureReason> {
            self.mv(piece, 0, 1)
        }

        pub fn try_action_left(&self, piece: &Piece) -> Result<Piece, MovementFailureReason> {
            self.mv(piece, -1, 0)
        }

        pub fn try_action_right(&self, piece: &Piece) -> Result<Piece, MovementFailureReason> {
            self.mv(piece, 1, 0)
        }

        pub fn try_action_rotate(&mut self, piece: &Piece) -> Result<Piece, MovementFailureReason> {
            let new_piece = piece.rotate(1);
            self.fits(&new_piece)?;
            Ok(new_piece)
        }

        pub fn try_action_hard_drop(&self, piece: &Piece) -> Result<Piece, MovementFailureReason> {
            let mut new_piece = self.try_action_down(piece)?;
            new_piece.drop = 1;
            while self.fits(&new_piece.mv(0, 1)).is_ok() {
                new_piece = new_piece.mv(0, 1);
                new_piece.drop += 1;
            }
            Ok(new_piece)
        }

        pub fn clear_lines(&mut self) -> u32 {
            let full_rows = self.columns.iter()
                .fold(u32::MAX, |acc, col| acc & *col);

            if full_rows == 0 {
                return 0;
            }

            let cleared = full_rows.count_ones();

            for col in &mut self.columns {
                let mut new_col = 0u32;
                let mut dst_y = H - 1;

                for src_y in (0..H).rev() {
                    let src_bit = 1u32 << src_y;

                    if full_rows & src_bit != 0 {
                        continue;
                    }

                    if *col & src_bit != 0 {
                        new_col |= 1u32 << dst_y;
                    }

                    dst_y -= 1;
                }

                *col = new_col;
            }

            cleared
        }

        pub fn compute_heights_holes(&self) -> ([u32; W], u32) {
            let mut heights = [0; W];
            let mut holes = 0;
            for (i, col) in self.columns.iter().enumerate() {
                let trailing_zeros = col.trailing_zeros();
                heights[i] = trailing_zeros;
                holes += col.count_zeros() - trailing_zeros;
            }
            (heights, holes)
        }

        pub fn compute_height_holes_bumps(&self) -> ([u32; W], f64, f64) {
            let (heights, holes) = self.compute_heights_holes();
            let mut bumps = 0;
            for i in 0..heights.len() - 1 {
                bumps += (heights[i] as i32 - heights[i + 1] as i32).unsigned_abs();
            }
            (heights, holes as f64, bumps as f64)
        }

        pub fn paper_reward_metric(&self, lines: u32) -> f64 {
            let (heights, holes, bumps) = self.compute_height_holes_bumps();
            let aggregate_height = heights.iter().sum::<u32>() as f64;
            -0.510066 * aggregate_height
                + 0.760666 * (lines * lines) as f64
                + -0.35663 * holes
                + -0.184483 * bumps
        }

        pub fn brett_reward_metric(&self, lines: u32) -> f64 {
            let (heights, holes, bumps) = self.compute_height_holes_bumps();
            let aggregate_height = heights.iter().sum::<u32>() as f64;
            let max_height = unsafe {
                *heights.iter().max().unwrap_unchecked()
            } as f64;
            -0.510066 * aggregate_height
                + (0.760666 * 16.) * (lines * lines) as f64
                + -(0.35663 * 4.) * holes
                + -0.184483 * bumps
                + -1.2 * max_height
        }

        pub fn reward_metric(&self, lines: u32) -> f64 {
            self.brett_reward_metric(lines)
        }
    }

    pub static PIECE_TYPES: [PieceType; 7] = [PieceType::I, PieceType::O, PieceType::T,
        PieceType::S, PieceType::Z, PieceType::J, PieceType::L];

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct BagOfPieces {
        pieces: [PieceType; 7],
        index: usize
    }

    impl Default for BagOfPieces {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BagOfPieces {
        pub fn new() -> Self {
            Self {
                pieces: PIECE_TYPES,
                index: 0
            }
        }

        pub fn next_piece(&mut self, rng: &mut impl SeedRng) -> Piece {
            if self.index >= self.pieces.len() {
                self.index = 0;
                self.pieces.shuffle(rng);
            }
            let piece = self.pieces[self.index];
            self.index += 1;
            Piece::new(piece)
        }
    }

    pub struct TetrisEngine<const W: usize, const H: usize, RNG: SeedRng> {
        grid: Grid<W, H>,
        config: &'static types::NNConfig,
        rng: RNG,
        bag: BagOfPieces,
        current_piece: Piece,
        next_piece: Piece,
        lines: u64,
        score: u64,
        actions_left: i64,
        placement_horizon_counter: i64,
        last_actions: Vec<Action>,
        is_game_over: bool,
        placed_last: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ActionResult {
        None,
        Placed,
        HitEdge,
        OverRotated,
        GameOver,
    }

    impl ActionResult {
        pub fn is_none(&self) -> bool {
            matches!(self, ActionResult::None)
        }
        pub fn is_game_over(&self) -> bool {
            matches!(self, ActionResult::GameOver)
        }
    }

    impl <const W: usize, const H: usize, RNG: SeedRng> TetrisEngine<W, H, RNG> {
        pub fn new(seed: RNG::Seed, config: &'static types::NNConfig) -> Self {
            let mut rng = RNG::from_seed(seed);
            let mut bag = BagOfPieces::new();

            Self {
                grid: Grid::new(),
                current_piece: bag.next_piece(&mut rng),
                next_piece: bag.next_piece(&mut rng),
                rng,
                bag,
                lines: 0,
                score: 0,
                actions_left: config.tetris.decay.actions_until_drop,
                placement_horizon_counter: config.tetris.truncate.placement_timer.value,
                config,
                last_actions: Vec::new(),
                is_game_over: false,
                placed_last: false,
            }
        }

        pub fn reset(&mut self) {
            self.grid = Grid::new();
            self.current_piece = self.bag.next_piece(&mut self.rng);
            self.next_piece = self.bag.next_piece(&mut self.rng);
            self.last_actions.clear();
            self.placed_last = false;
            self.is_game_over = false;
            self.actions_left = self.config.tetris.decay.actions_until_drop;
            self.placement_horizon_counter = self.config.tetris.truncate.placement_timer.value;
            self.lines = 0;
            self.score = 0;
        }

        pub fn next_piece(&mut self) {
            self.current_piece = self.next_piece;
            self.next_piece = self.bag.next_piece(&mut self.rng);
        }

        pub fn scored(&mut self, lines_cleared: u64) {
            self.lines += lines_cleared;
            self.score += lines_cleared * 100;
            match lines_cleared {
                2 => self.score += 100,
                3 => self.score += 200,
                4 => self.score += 400,
                _ => {}
            }
        }

        pub fn place_current(&mut self) -> Result<ActionResult, MovementFailureReason> {
            // do piece placement
            self.grid.place_piece(&self.current_piece)?;

            self.placed_last = true;

            // reset engine state effected by placing pieces
            self.actions_left = self.config.tetris.decay.actions_until_drop;
            self.placement_horizon_counter = self.config.tetris.truncate.placement_timer.value;
            self.last_actions.clear();

            let lines_cleared = self.grid.clear_lines() as u64;
            self.scored(lines_cleared);

            // next piece + game over check.
            self.next_piece();
            if self.grid.fits(&self.current_piece).is_err() {
                return Ok(ActionResult::GameOver);
            }

            Ok(ActionResult::None)
        }

        pub fn force_place(&mut self) {
            self.place_current().expect("Piece placement failed on down move. This \
                            should not be possible.");
        }

        pub fn handle_decay(&mut self, action: Action) -> Result<ActionResult, MovementFailureReason> {
            if action != Action::Down && self.config.tetris.decay.enabled {
                self.actions_left -= 1;
                if self.actions_left <= 0 {
                    self.actions_left = self.config.tetris.decay.actions_until_drop;
                    if let Ok(new_piece) = self.grid.try_action_down(&self.current_piece) {
                        self.current_piece = new_piece;
                    } else {
                        return self.place_current();
                    }
                }
            }
            Ok(ActionResult::None)
        }

        pub fn step(&mut self, action: Action) {
            self.placed_last = false;

            let lines = self.lines;
            let pre_reward = self.grid.reward_metric(lines as u32);
            let mut game_over = false;
            let mut truncated = false;

            let action_result = match action {
                Action::Right => self.grid.try_action_right(&self.current_piece),
                Action::Left => self.grid.try_action_left(&self.current_piece),
                Action::Down => self.grid.try_action_down(&self.current_piece),
                Action::Rotate => self.grid.try_action_rotate(&self.current_piece),
                Action::HardDrop => self.grid.try_action_hard_drop(&self.current_piece)
            };

            match action_result {
                Ok(new_piece) => {
                    self.current_piece = new_piece;
                    if action == Action::HardDrop {
                        self.score += self.current_piece.drop as u64;
                        self.force_place();
                    }
                },
                Err(reason) => {
                    if reason == MovementFailureReason::OutOfBounds {
                        match action {
                            Action::Right => {

                            },
                            Action::Left => {

                            },
                            _ => {}
                        }
                    }
                    match action {
                        Action::Right => {}
                        Action::Left => {}
                        Action::Rotate => {
                            if self.grid.try_action_down(&self.current_piece).is_err() {
                                self.force_place();
                            }
                        }
                        Action::Down | Action::HardDrop => {
                            self.force_place();
                        }
                    }
                }
            }

            let decay_result = self.handle_decay(action);
            if let Ok(decay_result) = decay_result {
                game_over = decay_result.is_game_over();
            } else if let Err(decay_result) = decay_result {
                panic!{"Decay Action failed: {:?}. This should not be possible.", decay_result}
            }
        }

    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    #[repr(u32)]
    pub enum Action {
        Right = 0,
        Left = 1,
        Down = 2,
        Rotate = 3,
        HardDrop = 4,
    }

    impl From<u32> for Action {
        fn from(value: u32) -> Self {
            match value % 5 {
                0 => Action::Right,
                1 => Action::Left,
                2 => Action::Down,
                3 => Action::Rotate,
                4 => Action::HardDrop,
                _ => unreachable!("Invalid action value"),
            }
        }
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}