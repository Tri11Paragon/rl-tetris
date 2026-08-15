#[macro_export]
macro_rules! x_type {
    () => {
        i32
    };
    // () => {
    //     u32
    // }'
}

#[macro_export]
macro_rules! pos_movement {
    ($shape:ident) => {
        match PIECE_SHAPES[$shape as usize].size {
            2 => 4,
            _ => 3,
        }
    };
    // ($shape:ident) => {
    //     match PIECE_SHAPES[$shape as usize].size {
    //         2 => 4,
    //         _ => 3,
    //     }
    // };
}

#[macro_export]
macro_rules! pos_offset {
    () => {
        3.5
    };
    // () => {
    //     4.5
    // };
}

#[macro_export]
macro_rules! down_decay_placement {
    ($self:ident) => {
        if $self.grid.try_action_down(&$self.current_piece).is_err() {
            return $self.force_place();
        }
    };
    ($self:ident, $ok:expr) => {
        if $self.grid.try_action_down(&$self.current_piece).is_err() {
            return $ok($self.force_place());
        }
    };
}