import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(PROJECT_ROOT)

from environment.tictactoe_env import TicTacToe4x4Env
from agent.ddqn_agent import DDQNAgent

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x_minimax_shaped.pth"
)

NUM_GAMES = 3

# ============================================================
# LOAD TRAINED DDQN
# ============================================================

agent = DDQNAgent(
    state_size=16,
    action_size=16,
    epsilon=0.0
)

agent.load(MODEL_PATH)
agent.epsilon = 0.0

print("Loaded model:", MODEL_PATH)
print("Device:", agent.device if hasattr(agent, "device") else "default")

# ============================================================
# PLAY GAMES
# ============================================================

results = {"DDQN wins": 0, "Rule wins": 0, "Draws": 0}

for game_number in range(1, NUM_GAMES + 1):

    env = TicTacToe4x4Env(opponent_type="rule")
    state = env.reset()

    print("\n" + "=" * 60)
    print(f"GAME {game_number}: DDQN X vs RULE-BASED O")
    print("=" * 60)

    env.render()
    move_number = 1
    info = {}

    while not env.is_done():

        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)

        print(f"\nMove {move_number}: DDQN X selects cell {action}")

        next_state, reward, done, info = env.step(action)

        env.render()
        print("Reward:", reward)
        print("Info:", info)

        state = next_state
        move_number += 1

    winner = info.get("winner")

    if winner == "X":
        result = "DDQN wins"
    elif winner == "O":
        result = "Rule wins"
    else:
        result = "Draws"

    results[result] += 1
    print("\nGAME RESULT:", result.upper())

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for result, count in results.items():
    print(f"{result}: {count}/{NUM_GAMES}")
