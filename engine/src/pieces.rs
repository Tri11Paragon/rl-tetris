use rand::RngExt;
use crate::{pos_movement, x_type};

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

pub static PIECE_TYPES: [PieceType; 7] = [
    PieceType::I,
    PieceType::O,
    PieceType::T,
    PieceType::S,
    PieceType::Z,
    PieceType::J,
    PieceType::L,
];

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
    pub columns: [u32; 4],
    pub largest_width: u32,
    pub largest_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceRotations {
    pub shapes: [PieceShape; 4],
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub shape: PieceType,
    pub rotation: PieceRotation,
    pub x: x_type!(),
    pub y: u32,
    pub drop: u32,
}

impl Piece {
    pub fn new(shape: PieceType) -> Self {
        Self {
            shape,
            rotation: PieceRotation::Zero,
            x: pos_movement!(shape),
            y: 0,
            drop: 0,
        }
    }

    pub fn random_piece<RNG: RngExt>(rng: &mut RNG) -> Self {
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

    pub fn as_char(&self) -> char {
        match self.shape {
            PieceType::I => 'I',
            PieceType::O => 'O',
            PieceType::T => 'T',
            PieceType::S => 'S',
            PieceType::Z => 'Z',
            PieceType::J => 'J',
            PieceType::L => 'L',
        }
    }

    pub fn mv(self, dx: i32, dy: i32) -> Self {
        Self {
            shape: self.shape,
            rotation: self.rotation,
            x: (self.x as i32 + dx) as x_type!(),
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

    pub fn center(self) -> usize {
        self.size() / 2
    }
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
        size: 4,
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
        size: 2,
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
        size: 3,
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
        size: 3,
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
        size: 3,
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
        size: 3,
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
        size: 3,
    },
];

// pub static PIECE_SHAPES: [PieceRotations; 7] = [
//     // long
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b10, 0b10, 0b10, 0b10],
//                 largest_width: 4,
//                 largest_height: 1,
//             },
//             PieceShape {
//                 columns: [0b0, 0b0, 0b1111, 0b0],
//                 largest_width: 1,
//                 largest_height: 4,
//             },
//             PieceShape {
//                 columns: [0b100, 0b100, 0b100, 0b100],
//                 largest_width: 4,
//                 largest_height: 1,
//             },
//             PieceShape {
//                 columns: [0b0, 0b1111, 0b0, 0b0],
//                 largest_width: 1,
//                 largest_height: 4,
//             },
//         ],
//         size: 4,
//     },
//     // square
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b11, 0b11, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b11, 0b11, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b11, 0b11, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b11, 0b11, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//         ],
//         size: 2,
//     },
//     // hat
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b10, 0b11, 0b10, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b0, 0b111, 0b10, 0b0],
//                 largest_width: 2,
//                 largest_height: 3,
//             },
//             PieceShape {
//                 columns: [0b10, 0b110, 0b10, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b10, 0b111, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 3,
//             },
//         ],
//         size: 3,
//     },
//     // right_snake
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b10, 0b11, 0b1, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b0, 0b11, 0b110, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b100, 0b110, 0b10, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b11, 0b110, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//         ],
//         size: 3,
//     },
//     // left_snake
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b1, 0b11, 0b10, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b0, 0b110, 0b11, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b10, 0b110, 0b100, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b110, 0b11, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 2,
//             },
//         ],
//         size: 3,
//     },
//     // left_gun
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b11, 0b10, 0b10, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b0, 0b111, 0b1, 0b0],
//                 largest_width: 2,
//                 largest_height: 3,
//             },
//             PieceShape {
//                 columns: [0b10, 0b10, 0b110, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b100, 0b111, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 3,
//             },
//         ],
//         size: 3,
//     },
//     // right_gun
//     PieceRotations {
//         shapes: [
//             PieceShape {
//                 columns: [0b10, 0b10, 0b11, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b0, 0b111, 0b100, 0b0],
//                 largest_width: 2,
//                 largest_height: 3,
//             },
//             PieceShape {
//                 columns: [0b110, 0b10, 0b10, 0b0],
//                 largest_width: 3,
//                 largest_height: 2,
//             },
//             PieceShape {
//                 columns: [0b1, 0b111, 0b0, 0b0],
//                 largest_width: 2,
//                 largest_height: 3,
//             },
//         ],
//         size: 3,
//     },
// ];
