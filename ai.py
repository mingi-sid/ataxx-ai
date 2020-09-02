import sys
import random

if __name__ == "__main__":

    input_str = sys.stdin.read()

    # 입력 예시
    # READY 1234567890.1234567 (입력시간)
    # "OK" 를 출력하세요.
    if input_str.startswith("READY"):
        # 출력
        sys.stdout.write("OK")

    # 입력 예시
    # PLAY
    # 2 0 0 0 0 0 1
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 1 0 0 0 0 0 2
    # 1234567890.1234567 (입력시간)

    # AI의 액션을 출력하세요.
    # 출력 예시 : "0 0 2 2"
    elif input_str.startswith("PLAY"):
        player = __file__[2]
        board = []
        actions = {} # { key: piece(start position), value: list of position(destination position) }

        # make board
        input_lines = input_str.split("\n")
        for i in range(7):
            board.append(input_lines[i+1].split(" "))

        for row in range(7):
            for col in range(7):
                if board[row][col] == player:
                    moveable_positions = []
                    for i in range(max(row-2, 0), min(row+3, 7)):
                        for j in range(max(col-2, 0), min(col+3, 7)):
                            if board[i][j] == "0":
                                moveable_positions.append((i, j))
                    if moveable_positions:
                        actions[(row, col)] = moveable_positions

        piece, positions = random.choice(list(actions.items()))
        position = random.choice(positions)

        # 출력
        sys.stdout.write(f"{piece[0]} {piece[1]} {position[0]} {position[1]}")
