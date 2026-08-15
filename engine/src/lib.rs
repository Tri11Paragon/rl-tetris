pub mod macros;
pub mod pieces;
#[cfg(test)]
pub mod test;
pub mod types;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub mod tetris {
    use super::pieces::*;
    use super::{down_decay_placement, pos_offset, types, x_type};
    use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
    use pyo3::prelude::*;
    use rand::{RngExt, SeedableRng, seq::SliceRandom};
    use std::collections::{HashMap, HashSet};
    use std::fmt::Debug;

    #[pymodule_export]
    pub const MATRIX_WIDTH: usize = 10;
    #[pymodule_export]
    pub const MATRIX_HEIGHT: usize = 22;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PlacementFailureReason {
        OutOfBounds,
        Collision,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct Grid<const W: usize, const H: usize> {
        columns: [u32; W],
    }

    impl<const W: usize, const H: usize> Grid<W, H> {
        pub fn new() -> Self {
            Self { columns: [0; W] }
        }

        pub fn print(&self) {
            let mut data = Vec::with_capacity(H);
            for _ in 0..H {
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

        pub fn fits(&self, piece: &Piece) -> Result<(), PlacementFailureReason> {
            let _ = self.blend(piece)?;
            Ok(())
        }

        pub fn blend(&self, piece: &Piece) -> Result<Self, PlacementFailureReason> {
            // if piece.size() > 4 {
            //     unreachable!();
            // }
            // let mut grid = *self;
            // let mut pos_x = piece.x as i32 - piece.center() as i32;
            // for p_col in piece.get().columns[0..piece.size()].iter().cloned() {
            //     if pos_x < 0 || pos_x >= W as i32 {
            //         if p_col == 0 {
            //             pos_x += 1;
            //             continue;
            //         } else {
            //             return Err(PlacementFailureReason::OutOfBounds);
            //         }
            //     }
            //     let p_col = p_col << piece.y;
            //     if p_col.bit_width() > H as u32 {
            //         return Err(PlacementFailureReason::OutOfBounds);
            //     }
            //     let g_col = self.columns[pos_x as usize];
            //     if g_col & p_col != 0 {
            //         return Err(PlacementFailureReason::Collision);
            //     }
            //     grid.columns[pos_x as usize] |= p_col;
            //     pos_x += 1;
            // }
            // Ok(grid)
            let mut grid = *self;
            if piece.x >= W as x_type!() {
                return Err(PlacementFailureReason::OutOfBounds);
            }
            let pos_x = piece.x as i32;
            let pos_x = pos_x .. pos_x + piece.size() as i32;
            for (p_col, pos_x) in piece.get().columns[0..piece.size()].iter().cloned().zip(pos_x) {
                if p_col == 0 {
                    continue;
                }
                let p_col = p_col << piece.y;
                if pos_x < 0 || pos_x >= W as i32 || p_col.bit_width() > H as u32 {
                    return Err(PlacementFailureReason::OutOfBounds);
                }
                let g_col = self.columns[pos_x as usize];
                if g_col & p_col != 0 {
                    return Err(PlacementFailureReason::Collision);
                }
                grid.columns[pos_x as usize] |= p_col;

            }
            Ok(grid)
        }

        pub fn place_piece(&mut self, piece: &Piece) -> Result<(), PlacementFailureReason> {
            *self = self.blend(piece)?;
            Ok(())
        }

        pub fn mv(&self, piece: &Piece, dx: i32, dy: i32) -> Result<Piece, PlacementFailureReason> {
            let new_piece = piece.mv(dx, dy);
            self.fits(&new_piece)?;
            Ok(new_piece)
        }

        pub fn try_action_down(&self, piece: &Piece) -> Result<Piece, PlacementFailureReason> {
            let mut new_piece = self.mv(piece, 0, 1)?;
            new_piece.drop = 1;
            Ok(new_piece)
        }

        pub fn try_action_left(&self, piece: &Piece) -> Result<Piece, PlacementFailureReason> {
            self.mv(piece, -1, 0)
        }

        pub fn try_action_right(&self, piece: &Piece) -> Result<Piece, PlacementFailureReason> {
            self.mv(piece, 1, 0)
        }

        pub fn try_action_rotate(&self, piece: &Piece) -> Result<Piece, PlacementFailureReason> {
            let new_piece = piece.rotate(1);
            let first_result = self.fits(&new_piece);
            if first_result.is_ok() {
                return Ok(new_piece);
            }
            let new_piece = piece.mv(1, 0).rotate(1);
            if self.fits(&new_piece).is_ok() {
                return Ok(new_piece);
            }
            let new_piece = piece.mv(-1, 0).rotate(1);
            if self.fits(&new_piece).is_ok() {
                return Ok(new_piece);
            }
            Err(first_result.err().unwrap())
        }

        pub fn try_action_hard_drop(&self, piece: &Piece) -> Result<Piece, PlacementFailureReason> {
            let mut new_piece = self.try_action_down(piece).unwrap_or(*piece);
            new_piece.drop = 1;
            while self.fits(&new_piece.mv(0, 1)).is_ok() {
                new_piece = new_piece.mv(0, 1);
                new_piece.drop += 1;
            }
            Ok(new_piece)
        }

        pub fn preview_fall_location(&self, piece: Piece) -> Grid<W, H> {
            let piece = self.try_action_hard_drop(&piece).unwrap_or(piece);
            Grid::new().blend(&piece).unwrap_or(*self)
        }

        pub fn clear_lines(&mut self) -> u32 {
            let full_rows = self.columns.iter().fold(u32::MAX, |acc, col| acc & *col);

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

                    // dst_y -= 1;
                    dst_y = dst_y.saturating_sub(1);
                }

                *col = new_col;
            }

            cleared
        }

        pub fn compute_heights_holes(&self) -> ([u32; W], u32) {
            let unnatural_zeros: u32 = 32 - H as u32;
            let mut heights = [0; W];
            let mut holes = 0;
            for (i, col) in self.columns.iter().enumerate() {
                let trailing_zeros = std::cmp::min(col.trailing_zeros(), H as u32);
                heights[i] = H as u32 - trailing_zeros;
                holes += col.count_zeros() - unnatural_zeros - trailing_zeros;
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
            -0.510066 * aggregate_height + 0.760666 * (lines * lines) as f64 + -0.35663 * holes + -0.184483 * bumps
        }

        pub fn brett_reward_metric(&self, lines: u32) -> f64 {
            let (heights, holes, bumps) = self.compute_height_holes_bumps();
            let aggregate_height = heights.iter().sum::<u32>() as f64;
            let max_height = *heights.iter().max().unwrap();
            // println!("max_height: {}, heights: {:?}, aggregate_height: {}, holes: {}, bumps: {}", max_height,
            //          heights, aggregate_height, holes, bumps);
            -0.510066 * aggregate_height
                + (0.760666 * 4.) * (lines * lines) as f64
                + -(0.35663 * 3.) * holes
                + -0.184483 * bumps
                + -0.1 * max_height as f64
        }

        pub fn reward_metric(&self, lines: u32) -> f64 {
            self.brett_reward_metric(lines)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct BagOfPieces {
        pieces: [PieceType; 7],
        index: usize,
    }

    impl BagOfPieces {
        pub fn new(rng: &mut impl RngExt) -> Self {
            let mut beep = Self {
                pieces: PIECE_TYPES,
                index: 0,
            };
            // beep.pieces.shuffle(rng);
            beep
        }

        pub fn next_piece(&mut self, rng: &mut impl RngExt) -> Piece {
            if self.index >= self.pieces.len() {
                self.index = 0;
                // self.pieces.shuffle(rng);
            }
            let piece = self.pieces[self.index];
            self.index += 1;
            Piece::new(piece)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct State<const W: usize, const H: usize> {
        board: Grid<W, H>,
        piece: Grid<W, H>,
    }

    impl<const W: usize, const H: usize> State<W, H> {
        pub fn new(board: Grid<W, H>, piece: Grid<W, H>) -> Self {
            Self { board, piece }
        }
    }

    pub struct TetrisEngine<const W: usize, const H: usize, RNG: SeedableRng + RngExt> {
        grid: Grid<W, H>,
        config: types::NNConfig,
        discouraged_actions: HashSet<Action>,
        encouraged_actions: HashSet<Action>,
        early_move_actions: HashMap<Action, f64>,
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
        is_truncated: bool,
        placed_last: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ActionResult {
        None,
        Placed,
        HitEdge,
        GameOver,
    }

    impl ActionResult {
        pub fn is_none(&self) -> bool {
            matches!(self, ActionResult::None)
        }
        pub fn is_game_over(&self) -> bool {
            matches!(self, ActionResult::GameOver)
        }

        pub fn is_placed(&self) -> bool {
            matches!(self, ActionResult::Placed)
        }
    }

    impl<const W: usize, const H: usize, RNG: SeedableRng + RngExt> TetrisEngine<W, H, RNG> {
        pub fn new(mut rng: RNG, config: types::NNConfig) -> Self {
            let mut bag = BagOfPieces::new(&mut rng);

            let mut discouraged_actions = HashSet::new();
            let mut encouraged_actions = HashSet::new();
            let mut early_move_actions = HashMap::new();

            for action_str in config.tetris.discouraged_actions.actions.iter() {
                discouraged_actions.insert(Action::from(action_str.as_str().unwrap()));
            }

            for action_str in config.tetris.encouraged_actions.actions.iter() {
                encouraged_actions.insert(Action::from(action_str.as_str().unwrap()));
            }

            for action_rwd in &config.tetris.states.early_move.actions_reward {
                early_move_actions.insert(
                    Action::from(action_rwd.name.as_str()),
                    action_rwd.reward.as_f64().unwrap_or(0.),
                );
            }

            Self {
                grid: Grid::new(),
                current_piece: bag.next_piece(&mut rng),
                next_piece: bag.next_piece(&mut rng),
                rng,
                bag,
                lines: 0,
                score: 0,
                actions_left: config.tetris.decay.actions_until_drop.as_i64().unwrap_or(0),
                placement_horizon_counter: config.tetris.truncate.placement_timer.value.as_i64().unwrap_or(0),
                config,
                discouraged_actions,
                encouraged_actions,
                early_move_actions,
                last_actions: Vec::new(),
                is_game_over: false,
                is_truncated: false,
                placed_last: false,
            }
        }

        pub fn reset(&mut self) {
            self.grid = Grid::new();
            self.bag = BagOfPieces::new(&mut self.rng);
            self.current_piece = self.bag.next_piece(&mut self.rng);
            self.next_piece = self.bag.next_piece(&mut self.rng);
            self.last_actions.clear();
            self.placed_last = false;
            self.is_game_over = false;
            self.actions_left = self.config.tetris.decay.actions_until_drop.as_i64().unwrap_or(0);
            self.placement_horizon_counter = self.config.tetris.truncate.placement_timer.value.as_i64().unwrap_or(0);
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

        pub fn place_current(&mut self) -> Result<ActionResult, PlacementFailureReason> {
            // do piece placement
            self.grid.place_piece(&self.current_piece)?;

            self.placed_last = true;

            // reset engine state effected by placing pieces
            self.actions_left = self.config.tetris.decay.actions_until_drop.as_i64().unwrap_or(0);
            self.placement_horizon_counter = self.config.tetris.truncate.placement_timer.value.as_i64().unwrap_or(0);
            self.last_actions.clear();

            let lines_cleared = self.grid.clear_lines() as u64;
            self.scored(lines_cleared);

            // next piece + game over check.
            self.next_piece();
            if self.grid.fits(&self.current_piece).is_err() {
                return Ok(ActionResult::GameOver);
            }

            Ok(ActionResult::Placed)
        }

        pub fn force_place(&mut self) -> ActionResult {
            // the panic won't happen as pieces are always checked before move
            // if it does happen this should crash as it is a hard bug.
            self.place_current()
                .expect("Piece placement failed on down move. This should not be possible.")
        }

        pub fn force_drop_place(&mut self) -> ActionResult {
            while let Ok(new_piece) = self.grid.try_action_down(&self.current_piece) {
                self.current_piece = new_piece;
            }
            self.force_place()
        }

        pub fn handle_decay(&mut self, action: Action) -> Result<ActionResult, PlacementFailureReason> {
            // no decay on action down or if the action resulted in a placement.
            if action != Action::Down
                && action != Action::HardDrop
                && self.config.tetris.decay.enabled
                && !self.placed_last
            {
                self.actions_left -= 1;
                if self.actions_left <= 0 {
                    self.actions_left = self.config.tetris.decay.actions_until_drop.as_i64().unwrap_or(0);
                    if let Ok(new_piece) = self.grid.try_action_down(&self.current_piece) {
                        self.current_piece = new_piece;
                        down_decay_placement!(self, Ok);
                    } else {
                        return self.place_current();
                    }
                }
            }
            Ok(ActionResult::None)
        }

        fn handle_action_ok(&mut self, action: &Action) -> ActionResult {
            match action {
                Action::HardDrop => {
                    self.score += self.current_piece.drop as u64 * 2;
                    self.force_place()
                }
                Action::Down => {
                    self.score += 1;
                    down_decay_placement!(self);
                    ActionResult::None
                }
                Action::Left | Action::Right => {
                    #[allow(clippy::collapsible_match)]
                    if self.grid.try_action_down(&self.current_piece).is_err() {
                        return self.force_place();
                    }
                    ActionResult::None
                }
                _ => {
                    down_decay_placement!(self);
                    ActionResult::None
                },
            }
        }

        fn handle_action_err(&mut self, reason: PlacementFailureReason, action: &Action) -> ActionResult {
            let mut result = ActionResult::None;
            if reason == PlacementFailureReason::OutOfBounds {
                result = match action {
                    Action::Right => ActionResult::HitEdge,
                    Action::Left => ActionResult::HitEdge,
                    _ => ActionResult::None,
                };
            };
            match action {
                Action::Rotate => {
                    if self.grid.try_action_down(&self.current_piece).is_err() {
                        return self.force_place();
                    }
                }
                Action::Down | Action::HardDrop => {
                    return self.force_place();
                }
                _ => {}
            }
            result
        }

        pub fn step(&mut self, action: Action) -> (State<W, H>, f64, u64, bool, bool) {
            self.placed_last = false;
            self.is_truncated = false;

            let lines = self.lines;
            let pre_reward = self.grid.reward_metric(0);
            let mut reward = 0f64;

            let action_result = match action {
                Action::Right => self.grid.try_action_right(&self.current_piece),
                Action::Left => self.grid.try_action_left(&self.current_piece),
                Action::Down => self.grid.try_action_down(&self.current_piece),
                Action::Rotate => self.grid.try_action_rotate(&self.current_piece),
                Action::HardDrop => self.grid.try_action_hard_drop(&self.current_piece),
            };

            let result = match action_result {
                Ok(new_piece) => {
                    self.current_piece = new_piece;
                    self.handle_action_ok(&action)
                }
                Err(reason) => self.handle_action_err(reason, &action),
            };
            self.last_actions.push(action);

            self.is_game_over |= result.is_game_over();
            self.is_truncated |= result.is_placed() && self.config.tetris.truncate.piece_placement_truncates;

            let decay_result = self.handle_decay(action);
            if let Ok(decay_result) = decay_result {
                self.is_game_over |= decay_result.is_game_over();
            } else if let Err(decay_result) = decay_result {
                // intentional, effectively unreachable.
                panic! {"Decay Action failed: {:?}. This should not be possible.", decay_result}
            }

            let lines_cleared = self.lines - lines;
            let post_reward = self.grid.reward_metric(lines_cleared as u32);
            reward += post_reward - pre_reward;

            if self.discouraged_actions.contains(&action) {
                reward += self.config.tetris.discouraged_actions.reward.as_f64().unwrap_or(0.);
            }

            if self.encouraged_actions.contains(&action) {
                reward += self.config.tetris.encouraged_actions.reward.as_f64().unwrap_or(0.);
            }

            if self.is_game_over {
                reward += self.config.tetris.states.game_over.as_f64().unwrap_or(0.);
            }

            self.placement_horizon_counter -= 1;
            if self.config.tetris.truncate.placement_timer.enabled && self.placement_horizon_counter <= 0 {
                self.is_truncated = true;
                reward += self
                    .config
                    .tetris
                    .truncate
                    .placement_timer
                    .reward
                    .as_f64()
                    .unwrap_or(0.);
                self.placement_horizon_counter =
                    self.config.tetris.truncate.placement_timer.value.as_i64().unwrap_or(0);
            }

            if self.config.tetris.states.cyclic.enabled {
                if self.last_actions.len() >= 2
                    && (self.last_actions.ends_with(&[Action::Left, Action::Right])
                        || self.last_actions.ends_with(&[Action::Right, Action::Left]))
                {
                    reward += self.config.tetris.states.cyclic.reward.as_f64().unwrap_or(0.);
                }
                let max_rotates = self.config.tetris.states.cyclic.max_rotates.as_i64().unwrap_or(0);
                if max_rotates > 0 && self.last_actions.len() >= max_rotates as usize {
                    let all_rotates = self
                        .last_actions
                        .iter()
                        .rev()
                        .take(max_rotates as usize)
                        .all(|&action| action == Action::Rotate);
                    if all_rotates {
                        let rotate_iter = self.last_actions.iter().rev().skip(max_rotates as
                            usize).take(self.config.tetris.states.cyclic.rotate_horizon.as_u64()
                            .unwrap_or(4) as usize - max_rotates as usize);
                        for action in rotate_iter {
                            if *action != Action::Rotate {
                                break;
                            }
                            reward += self.config.tetris.states.cyclic.reward.as_f64().unwrap_or(0.);
                        }
                    }
                }
            }

            if self.config.tetris.states.edges.enabled && result == ActionResult::HitEdge {
                reward += self.config.tetris.states.edges.reward.as_f64().unwrap_or(0.);
            }

            if self.config.tetris.states.early_move.enabled {
                if self.current_piece.y <= self.config.tetris.states.early_move.cutoff.as_u64().unwrap_or(0) as u32 {
                    if self.early_move_actions.contains_key(&action) {
                        let factor = self.current_piece.x as f64;
                        let diminish_factor = self
                            .config
                            .tetris
                            .states
                            .early_move
                            .diminish_factor
                            .as_f64()
                            .unwrap_or(1.)
                            .powf((factor - pos_offset!()).abs());
                        reward += self.early_move_actions[&action] * diminish_factor;
                    }
                } else if self.config.tetris.states.early_move.punishment.punish_late_moves
                    && self.early_move_actions.contains_key(&action)
                {
                    reward -= self.early_move_actions[&action]
                        * self
                            .config
                            .tetris
                            .states
                            .early_move
                            .punishment
                            .factor
                            .as_f64()
                            .unwrap_or(1.);
                }
            }

            (
                self.state(),
                reward,
                lines_cleared,
                self.is_game_over,
                self.is_truncated,
            )
        }

        pub fn preview_fall_location(&self, piece: Piece) -> Grid<W, H> {
            self.grid.preview_fall_location(piece)
        }

        pub fn state(&self) -> State<W, H> {
            let mut piece_grid = Grid::<W, H>::new();
            // this will literally never fail as piece_grid is empty.
            piece_grid
                .place_piece(&self.current_piece)
                .expect("Shouldn't be possible");
            State::new(self.grid, piece_grid)
        }
    }

    macro_rules! count {
        ($($item:tt),* $(,)?) => {
            <[()]>::len(&[$(count!(@replace $item ())),*])
        };

        (@replace $_item:tt $replacement:expr) => {
            $replacement
        };
    }

    macro_rules! define_action_enums {
        (
            $(
                $variant:ident = $value:expr; $str:literal
            ),+ $(,)?
        ) => {
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
            #[repr(u32)]
            pub enum Action {
                $(
                    $variant = $value,
                )+
            }

            #[pyclass(from_py_object)]
            #[repr(u32)]
            #[derive(Clone)]
            pub enum PyAction {
                $(
                    $variant = $value,
                )+
            }

            impl From<PyAction> for Action {
                fn from(value: PyAction) -> Self {
                    match value {
                        $(
                            PyAction::$variant => Action::$variant,
                        )+
                    }
                }
            }

            impl From<u32> for Action {
                fn from(value: u32) -> Self {
                    match value % count!($($variant),+) as u32 {
                        $(
                            $value => Action::$variant,
                        )+
                        _ => unreachable!("Invalid action value"),
                    }
                }
            }

            impl From<&str> for Action {
                fn from(value: &str) -> Self {
                    match value.to_ascii_uppercase().as_str() {
                        $(
                            $str => Action::$variant,
                        )+
                        // we need to panic if the user provides an invalid type.
                        _ => panic!("Invalid action value"),
                    }
                }
            }

        };
    }

    define_action_enums! {
        Right = 0; "RIGHT",
        Left = 1; "LEFT",
        Down = 2; "DOWN",
        Rotate = 3; "ROTATE",
        HardDrop = 4; "HARD_DROP",
    }

    #[pyclass]
    pub struct PyTetrisEngine {
        engine: TetrisEngine<MATRIX_WIDTH, MATRIX_HEIGHT, rand::rngs::SmallRng>,
    }

    fn grids_to_numpy<'py, const W: usize, const H: usize>(
        py: Python<'py>,
        grids: &[Grid<W, H>],
    ) -> Bound<'py, PyArray3<f32>> {
        let mut data = Vec::with_capacity(grids.len() * H * W);

        for grid in grids {
            for y in 0..H {
                for x in 0..W {
                    let occupied = (grid.columns[x] >> y) & 1;
                    data.push(occupied as f32);
                }
            }
        }

        data.into_pyarray(py).reshape([grids.len(), H, W]).unwrap()
    }

    fn grids_to_bitwise_numpy<'py, const W: usize, const H: usize>(
        py: Python<'py>,
        grids: &[Grid<W, H>],
    ) -> Bound<'py, PyArray2<u32>> {
        let mut data = Vec::with_capacity(grids.len() * W);

        for grid in grids {
            for x in 0..W {
                data.push(grid.columns[x]);
            }
        }

        data.into_pyarray(py).reshape([grids.len(), W]).unwrap()
    }

    fn state_to_numpy<'py, const W: usize, const H: usize>(
        py: Python<'py>,
        state: State<W, H>,
    ) -> Bound<'py, PyArray3<f32>> {
        grids_to_numpy(py, &[state.board, state.piece])
    }

    fn state_to_bitwise_numpy<'py, const W: usize, const H: usize>(
        py: Python<'py>,
        state: State<W, H>,
    ) -> Bound<'py, PyArray2<u32>> {
        grids_to_bitwise_numpy(py, &[state.board, state.piece])
    }

    #[pymethods]
    impl PyTetrisEngine {
        #[new]
        pub fn new(seed: u64, config_json: &str) -> PyResult<Self> {
            let config: types::NNConfig = serde_json::from_str(config_json)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;

            let rng = rand::rngs::SmallRng::seed_from_u64(seed);

            let engine = TetrisEngine::new(rng, config);
            Ok(Self { engine })
        }

        pub fn step<'py>(
            &mut self,
            py: Python<'py>,
            action: u32,
        ) -> PyResult<(Bound<'py, PyArray3<f32>>, f64, u64, bool, bool)> {
            let (state, reward, lines_cleared, game_over, truncated) = self.engine.step(Action::from(action));
            Ok((state_to_numpy(py, state), reward, lines_cleared, game_over, truncated))
        }

        pub fn step_bitwise<'py>(
            &mut self,
            py: Python<'py>,
            action: u32,
        ) -> PyResult<(Bound<'py, PyArray2<u32>>, f64, u64, bool, bool)> {
            let (state, reward, lines_cleared, game_over, truncated) = self.engine.step(Action::from(action));
            Ok((
                state_to_bitwise_numpy(py, state),
                reward,
                lines_cleared,
                game_over,
                truncated,
            ))
        }

        pub fn current_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
            Ok(state_to_numpy(py, self.engine.state()))
        }

        pub fn current_state_bitwise<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
            Ok(state_to_bitwise_numpy(py, self.engine.state()))
        }

        // Only used for rendering debug info.
        pub fn preview_fall_location<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
            Ok(grids_to_numpy(
                py,
                &[self.engine.preview_fall_location(self.engine.current_piece)],
            ))
        }

        pub fn reset(&mut self) -> PyResult<()> {
            self.engine.reset();
            Ok(())
        }

        pub fn lines(&self) -> PyResult<u64> {
            Ok(self.engine.lines)
        }

        pub fn score(&self) -> PyResult<u64> {
            Ok(self.engine.score)
        }

        pub fn print(&self) -> PyResult<()> {
            self.engine.grid.print();
            Ok(())
        }

        pub fn current_piece(&self) -> PyResult<char> {
            Ok(self.engine.current_piece.as_char())
        }

        pub fn next_piece(&self) -> PyResult<char> {
            Ok(self.engine.next_piece.as_char())
        }
    }
}
