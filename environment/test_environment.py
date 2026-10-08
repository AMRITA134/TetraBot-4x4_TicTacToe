import os
import sys

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from environment.tictactoe_env import TicTacToe4x4Env


# --------------------------------------------------
# Create environment
# --------------------------------------------------

env = TicTacToe4x4Env(
    opponent_type="random"
)


# --------------------------------------------------
# Reset
# --------------------------------------------------

state = env.reset()

print("Initial state:")
print(state)


# --------------------------------------------------
# Display board
# --------------------------------------------------

print("\nInitial board:")

env.render()


# --------------------------------------------------
# Valid actions
# --------------------------------------------------

print("Valid actions:")

print(
    env.get_valid_actions()
)


# --------------------------------------------------
# Test X move
# --------------------------------------------------

action = 0

print(
    f"\nX selects cell: {action}"
)


next_state, reward, done, info = env.step(
    action
)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\nAfter X move and O response:")

env.render()


print("Next state:")

print(next_state)


print("Reward:", reward)

print("Done:", done)

print("Info:", info)