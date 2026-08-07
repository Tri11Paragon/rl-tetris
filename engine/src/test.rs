use crate::engine::*;
#[test]
pub fn test_something() {
    println!("test");
    let mut grid = Grid::<10, 22>::new();
    let piece = Piece::new(PieceType::Z);
    let piece = piece.rotate(0);
    let piece = piece.mv(4, 20);
    println!("{:?}", piece);
    grid.place_piece(&piece);
    let piece = Piece::new(PieceType::Z);
    let piece = piece.rotate(1);
    let piece = piece.mv(2, 19);
    grid.place_piece(&piece);
    grid.print();
}