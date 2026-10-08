import os
import sys
import numpy as np
import torch

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
from agent.ddqn_agent import DDQNAgent


# ==================================================
# TRAINING SETTINGS
# ==================================================

NUM_EPISODES = 20_000

PRINT_EVERY = 100

SAVE_EVERY = 1_000

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x.pth"
)


# ==================================================
# CREATE DIRECTORIES
# ==================================================

os.makedirs(
    os.path.join(PROJECT_ROOT, "models"),
    exist_ok=True
)


# ==================================================
# CREATE ENVIRONMENT
# ==================================================

env = TicTacToe4x4Env()


# ==================================================
# CREATE DDQN AGENT
# ==================================================

agent = DDQNAgent(
    state_size=16,
    action_size=16,

    learning_rate=0.001,

    gamma=0.99,

    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,

    buffer_size=100_000,

    batch_size=64,

    target_update_frequency=100
)


# ==================================================
# STATISTICS
# ==================================================

wins = 0
losses = 0
draws = 0

episode_rewards = []

loss_history = []


# ==================================================
# TRAINING LOOP
# ==================================================

for episode in range(1, NUM_EPISODES + 1):

    # ----------------------------------------------
    # Reset environment
    # ----------------------------------------------

    state = env.reset()

    done = False

    total_reward = 0

    episode_losses = []


    # ----------------------------------------------
    # Play one game
    # ----------------------------------------------

    while not done:

        # Get valid cells
        valid_actions = env.get_valid_actions()

        # Select X's action using DDQN
        action = agent.select_action(
            state,
            valid_actions
        )

        # Perform action
        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)

        # Store experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Train DDQN
        loss = agent.train()

        if loss is not None:
            episode_losses.append(loss)

        # Update state
        state = next_state

        total_reward += reward


    # ----------------------------------------------
    # Update statistics
    # ----------------------------------------------

    winner = info.get(
        "winner",
        None
    )

    if winner == "X":
        wins += 1

    elif winner == "O":
        losses += 1

    elif winner == "draw":
        draws += 1


    # ----------------------------------------------
    # Epsilon decay
    # ----------------------------------------------

    agent.decay_epsilon()


    # ----------------------------------------------
    # Store statistics
    # ----------------------------------------------

    episode_rewards.append(
        total_reward
    )

    if episode_losses:

        loss_history.append(
            np.mean(episode_losses)
        )


    # ==================================================
    # PRINT PROGRESS
    # ==================================================

    if episode % PRINT_EVERY == 0:

        total_games = (
            wins +
            losses +
            draws
        )

        win_rate = (
            wins / total_games
        ) * 100

        loss_rate = (
            losses / total_games
        ) * 100

        draw_rate = (
            draws / total_games
        ) * 100

        average_reward = np.mean(
            episode_rewards[-PRINT_EVERY:]
        )

        if loss_history:

            average_loss = np.mean(
                loss_history[-PRINT_EVERY:]
            )

        else:

            average_loss = 0


        print(
            f"\nEpisode: {episode}/{NUM_EPISODES}"
        )

        print(
            f"Wins: {wins} | "
            f"Losses: {losses} | "
            f"Draws: {draws}"
        )

        print(
            f"Win Rate: {win_rate:.2f}%"
        )

        print(
            f"Loss Rate: {loss_rate:.2f}%"
        )

        print(
            f"Draw Rate: {draw_rate:.2f}%"
        )

        print(
            f"Average Reward: "
            f"{average_reward:.3f}"
        )

        print(
            f"Average Loss: "
            f"{average_loss:.6f}"
        )

        print(
            f"Epsilon: "
            f"{agent.epsilon:.4f}"
        )

        print(
            f"Replay Buffer: "
            f"{len(agent.memory)}"
        )


    # ==================================================
    # SAVE MODEL
    # ==================================================

    if episode % SAVE_EVERY == 0:

        agent.save(
            MODEL_PATH
        )


# ==================================================
# FINAL SAVE
# ==================================================

agent.save(
    MODEL_PATH
)


# ==================================================
# FINAL RESULTS
# ==================================================

total_games = (
    wins +
    losses +
    draws
)

print("\n======================================")
print("Training completed")
print("======================================")

print(
    f"Total games: {total_games}"
)

print(
    f"Wins: {wins}"
)

print(
    f"Losses: {losses}"
)

print(
    f"Draws: {draws}"
)

print(
    f"Final win rate: "
    f"{(wins / total_games) * 100:.2f}%"
)

print(
    f"Final epsilon: "
    f"{agent.epsilon:.4f}"
)

print(
    f"Model saved at:"
)

print(
    MODEL_PATH
)