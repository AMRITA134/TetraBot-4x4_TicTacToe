import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from environment.tictactoe_env import TicTacToe4x4Env


def print_result(title, reward, done, info):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)


# ============================================================
# TEST 1
# X BLOCKS O'S IMMEDIATE WIN
# ============================================================

env = TicTacToe4x4Env(
    opponent_type="random"
)

env.reset()

# O O O _
# X needs to play cell 3
env.board.state[0] = -1
env.board.state[1] = -1
env.board.state[2] = -1

# X plays cell 3 to block O
_, reward, done, info = env.step(3)

print_result(
    "TEST 1: X blocks O's immediate winning move",
    reward,
    done,
    info
)


# ============================================================
# TEST 2
# X CREATES AN IMMEDIATE WINNING THREAT
# ============================================================

env = TicTacToe4x4Env(
    opponent_type="random"
)

env.reset()

# X X _ _
# X needs to play cell 2
# This creates X X X _
env.board.state[0] = 1
env.board.state[1] = 1

# O somewhere harmless
env.board.state[8] = -1

_, reward, done, info = env.step(2)

print_result(
    "TEST 2: X creates an immediate winning threat",
    reward,
    done,
    info
)


# ============================================================
# TEST 3
# X LEAVES AN IMMEDIATE O WINNING THREAT
# ============================================================

env = TicTacToe4x4Env(
    opponent_type="random"
)

env.reset()

# O O O _
# X must block cell 3
env.board.state[0] = -1
env.board.state[1] = -1
env.board.state[2] = -1

# X deliberately plays somewhere else
_, reward, done, info = env.step(8)

print_result(
    "TEST 3: X leaves an immediate O winning opportunity",
    reward,
    done,
    info
)