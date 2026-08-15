use crate::tetris::*;
use crate::pieces::*;
#[test]
pub fn test_something() {
    println!("test");
    let mut grid = Grid::<10, 22>::new();
    let piece = Piece::new(PieceType::Z);
    let piece = piece.rotate(0);
    let piece = piece.mv(4, 20);
    println!("{:?}", piece);
    grid.place_piece(&piece).unwrap();
    let piece = Piece::new(PieceType::Z);
    let piece = piece.rotate(1);
    let piece = piece.mv(2, 19);
    grid.place_piece(&piece).unwrap();
    let piece = Piece::new(PieceType::S);
    let piece = piece.rotate(1);
    let piece = piece.mv(5, 18);
    grid.place_piece(&piece).unwrap();
    let piece = Piece::new(PieceType::O);
    let piece = piece.mv(-1, 20);
    grid.place_piece(&piece).unwrap();
    let piece = Piece::new(PieceType::O);
    let piece = piece.mv(-1 - 2, 20);
    grid.place_piece(&piece).unwrap();
    let piece = Piece::new(PieceType::J);
    let piece = piece.rotate(1);
    let piece = piece.mv(-3, 19);
    grid.place_piece(&piece).unwrap();
    grid.print();
    println!("Lines Cleared: {}", grid.clear_lines());
    grid.print();
}