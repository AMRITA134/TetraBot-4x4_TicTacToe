import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from environment.tictactoe_env import TicTacToe4x4Env
from agent.ddqn_agent import DDQNAgent


# ============================================================
# LOAD TRAINED DDQN
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x.pth"
)

agent = DDQNAgent(
    state_size=16,
    action_size=16,
    epsilon=0.0
)

agent.load(MODEL_PATH)

# Make absolutely sure we are evaluating greedily
agent.epsilon = 0.0


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = TicTacToe4x4Env(
    opponent_type="minimax"
)


# ============================================================
# PLAY ONE GAME
# ============================================================

state = env.reset()

print("\n")
print("=" * 60)
print("        DDQN X  vs  MINIMAX O")
print("=" * 60)

print("\nInitial board:")
env.render()


move_number = 1

while not env.is_done():

    valid_actions = env.get_valid_actions()

    # --------------------------------------------------------
    # DDQN selects X move
    # --------------------------------------------------------

    action = agent.select_action(
        state,
        valid_actions
    )

    print("\n" + "-" * 60)
    print(f"Move {move_number}")
    print("-" * 60)

    print("DDQN X selects cell:", action)

    next_state, reward, done, info = env.step(action)

    print("\nBoard after X move and Minimax response:")
    env.render()

    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    state = next_state

    move_number += 1


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("                    GAME OVER")
print("=" * 60)

winner = info.get("winner")

if winner == "X":
    print("Result: DDQN X WON")
elif winner == "O":
    print("Result: MINIMAX O WON")
elif winner == "draw":
    print("Result: DRAW")
else:
    print("Result:", winner)