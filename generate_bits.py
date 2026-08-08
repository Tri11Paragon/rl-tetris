import MaTris.tetrominoes as tet

def trailing_zeros(n: int) -> int:
    if n == 0:
        return 0  # or choose a fixed width, depending on your use case
    return (n & -n).bit_length() - 1

def print_cols(bitfield_columns):
    print("PieceShape {")
    print("columns: [", end='')
    for i, col in enumerate(bitfield_columns):
        print(bin(col), end=', ' if (i < len(bitfield_columns) - 1 or len(bitfield_columns) < 4) else '')
    for i in range(len(bitfield_columns), 4):
        print("0b0", end=', ' if i < 3 else '')
    print("],")
    largest_width = [item for item in filter(lambda x: x != 0, iter(bitfield_columns))]
    print(f"largest_width: {len(largest_width)},")
    largest_height = [item for item in map(lambda x: x.bit_length() - trailing_zeros(x), iter(bitfield_columns))]
    print(f"largest_height: {max(largest_height)},")
    print("},")

def rotate_columns_clockwise(bitfield_columns):
    width = len(bitfield_columns)
    rotated = [0] * width

    for x, col in enumerate(bitfield_columns):
        for y in range(width):
            if col & (1 << y):
                new_x = width - 1 - y
                new_y = x
                rotated[new_x] |= 1 << new_y

    return rotated

def main():
    for k, v in tet.tetrominoes.items():
        bitfield_columns = [0] * len(v.shape)
        idx = 0
        for row in v.shape:
            for column, data in enumerate(row):
                val = 0 if data is None else 1
                bitfield_columns[column] |= val << idx
            idx += 1
        print()
        print(f"// {k}")
        print("PieceRotations {")
        print("shapes: [")
        for i in range(4):
            print_cols(bitfield_columns)
            bitfield_columns = rotate_columns_clockwise(bitfield_columns)
        print("],")
        print(f"size: {len(bitfield_columns)}")
        print("},")

if __name__ == "__main__":
    main()

