import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from environment.board import Board
from evaluation.opponents import minimax_opponent


def print_test(title, board):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    board.display()

    move = minimax_opponent(
        board,
        depth=3,
        player=-1
    )

    print("Minimax selected move:", move)


# ============================================================
# TEST 1: O HAS AN IMMEDIATE WIN
# ============================================================

board = Board()

# O O O _
board.state[0] = -1
board.state[1] = -1
board.state[2] = -1

# X X _ _
board.state[4] = 1
board.state[5] = 1

print_test(
    "TEST 1: Minimax should win at cell 3",
    board
)


# ============================================================
# TEST 2: O MUST BLOCK X
# ============================================================

board = Board()

# X X X _
board.state[0] = 1
board.state[1] = 1
board.state[2] = 1

# O O _ _
board.state[4] = -1
board.state[5] = -1

print_test(
    "TEST 2: Minimax should block X at cell 3",
    board
)


# ============================================================
# TEST 3: SIMPLE EMPTY BOARD
# ============================================================

board = Board()

print_test(
    "TEST 3: Minimax move on empty board",
    board
)