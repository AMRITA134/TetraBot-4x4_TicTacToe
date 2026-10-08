import os
import sys
import random
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

# Percentage of games against Minimax
MINIMAX_PROBABILITY = 0.80

# Percentage of games against Random
RANDOM_PROBABILITY = 0.20


# ==================================================
# MODEL PATHS
# ==================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x.pth"
)

UPGRADED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x_minimax_shaped.pth"
)


# ==================================================
# CREATE DIRECTORIES
# ==================================================

os.makedirs(
    os.path.join(
        PROJECT_ROOT,
        "models"
    ),
    exist_ok=True
)


# ==================================================
# CREATE DDQN AGENT
# ==================================================

print("\n")
print("=" * 60)
print("      DDQN 4x4 MINIMAX UPGRADED TRAINING")
print("=" * 60)

print("\nLoading existing DDQN model:")
print(MODEL_PATH)


agent = DDQNAgent(
    state_size=16,
    action_size=16,

    learning_rate=0.001,

    gamma=0.99,

    # Start with controlled exploration.
    # We are continuing from an already trained model.
    epsilon=0.10,
    epsilon_min=0.05,
    epsilon_decay=0.9995,

    buffer_size=100_000,

    batch_size=64,

    target_update_frequency=100
)


# ==================================================
# LOAD EXISTING MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nExisting model was not found:\n"
        f"{MODEL_PATH}\n\n"
        "Make sure your original DDQN model "
        "ddqn_4x4_x.pth exists before running "
        "this training."
    )


agent.load(MODEL_PATH)

print("\nExisting model loaded successfully.")

# Make sure epsilon is appropriate for continued training
agent.epsilon = 0.10


# ==================================================
# STATISTICS
# ==================================================

total_wins = 0
total_losses = 0
total_draws = 0

minimax_games = 0
minimax_wins = 0
minimax_losses = 0
minimax_draws = 0

random_games = 0
random_wins = 0
random_losses = 0
random_draws = 0

episode_rewards = []

loss_history = []


# ==================================================
# TRAINING LOOP
# ==================================================

for episode in range(1, NUM_EPISODES + 1):

    # ==================================================
    # SELECT OPPONENT
    # ==================================================

    if random.random() < MINIMAX_PROBABILITY:

        opponent_type = "minimax"

        minimax_games += 1

    else:

        opponent_type = "random"

        random_games += 1


    # ==================================================
    # CREATE ENVIRONMENT
    # ==================================================

    env = TicTacToe4x4Env(
        opponent_type=opponent_type
    )


    # ==================================================
    # RESET
    # ==================================================

    state = env.reset()

    done = False

    total_reward = 0

    episode_losses = []


    # ==================================================
    # PLAY ONE GAME
    # ==================================================

    while not done:

        # ----------------------------------------------
        # Get valid actions
        # ----------------------------------------------

        valid_actions = env.get_valid_actions()


        # ----------------------------------------------
        # DDQN selects X move
        # ----------------------------------------------

        action = agent.select_action(
            state,
            valid_actions
        )


        # ----------------------------------------------
        # Environment
        #
        # X = DDQN
        # O = selected opponent
        # ----------------------------------------------

        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)


        # ----------------------------------------------
        # Store experience
        # ----------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )


        # ----------------------------------------------
        # Train DDQN
        # ----------------------------------------------

        loss = agent.train()

        if loss is not None:

            episode_losses.append(loss)


        # ----------------------------------------------
        # Update state
        # ----------------------------------------------

        state = next_state

        total_reward += reward


    # ==================================================
    # GAME RESULT
    # ==================================================

    winner = info.get(
        "winner",
        None
    )


    # ==================================================
    # UPDATE TOTAL STATISTICS
    # ==================================================

    if winner == "X":

        total_wins += 1

    elif winner == "O":

        total_losses += 1

    elif winner == "draw":

        total_draws += 1


    # ==================================================
    # UPDATE OPPONENT-SPECIFIC STATISTICS
    # ==================================================

    if opponent_type == "minimax":

        if winner == "X":

            minimax_wins += 1

        elif winner == "O":

            minimax_losses += 1

        elif winner == "draw":

            minimax_draws += 1


    elif opponent_type == "random":

        if winner == "X":

            random_wins += 1

        elif winner == "O":

            random_losses += 1

        elif winner == "draw":

            random_draws += 1


    # ==================================================
    # EPSILON DECAY
    # ==================================================

    agent.decay_epsilon()


    # ==================================================
    # STORE STATISTICS
    # ==================================================

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
            total_wins
            + total_losses
            + total_draws
        )

        win_rate = (
            total_wins
            / total_games
        ) * 100

        loss_rate = (
            total_losses
            / total_games
        ) * 100

        draw_rate = (
            total_draws
            / total_games
        ) * 100


        # ------------------------------------------------
        # Recent reward
        # ------------------------------------------------

        recent_rewards = episode_rewards[
            -PRINT_EVERY:
        ]

        average_reward = np.mean(
            recent_rewards
        )


        # ------------------------------------------------
        # Recent loss
        # ------------------------------------------------

        if loss_history:

            average_loss = np.mean(
                loss_history[
                    -PRINT_EVERY:
                ]
            )

        else:

            average_loss = 0


        # ==================================================
        # PRINT
        # ==================================================

        print("\n" + "=" * 60)

        print(
            f"Episode: "
            f"{episode}/{NUM_EPISODES}"
        )

        print(
            f"Current opponent mix: "
            f"{MINIMAX_PROBABILITY * 100:.0f}% Minimax / "
            f"{RANDOM_PROBABILITY * 100:.0f}% Random"
        )


        print("\nOverall:")

        print(
            f"Wins: {total_wins} | "
            f"Losses: {total_losses} | "
            f"Draws: {total_draws}"
        )

        print(
            f"Win Rate: "
            f"{win_rate:.2f}%"
        )

        print(
            f"Loss Rate: "
            f"{loss_rate:.2f}%"
        )

        print(
            f"Draw Rate: "
            f"{draw_rate:.2f}%"
        )


        # ==================================================
        # MINIMAX RESULTS
        # ==================================================

        print("\nAgainst Minimax:")

        print(
            f"Games: "
            f"{minimax_games}"
        )

        if minimax_games > 0:

            print(
                f"DDQN Wins: "
                f"{minimax_wins}"
            )

            print(
                f"Minimax Wins: "
                f"{minimax_losses}"
            )

            print(
                f"Draws: "
                f"{minimax_draws}"
            )

            print(
                f"DDQN Win Rate: "
                f"{(
                    minimax_wins
                    / minimax_games
                ) * 100:.2f}%"
            )


        # ==================================================
        # RANDOM RESULTS
        # ==================================================

        print("\nAgainst Random:")

        print(
            f"Games: "
            f"{random_games}"
        )

        if random_games > 0:

            print(
                f"DDQN Wins: "
                f"{random_wins}"
            )

            print(
                f"Random Wins: "
                f"{random_losses}"
            )

            print(
                f"Draws: "
                f"{random_draws}"
            )

            print(
                f"DDQN Win Rate: "
                f"{(
                    random_wins
                    / random_games
                ) * 100:.2f}%"
            )


        # ==================================================
        # TRAINING INFO
        # ==================================================

        print("\nTraining:")

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
    # SAVE CHECKPOINT
    # ==================================================

    if episode % SAVE_EVERY == 0:

        checkpoint_path = os.path.join(
            PROJECT_ROOT,
            "models",
            f"ddqn_4x4_minimax_ep{episode}.pth"
        )

        agent.save(
            checkpoint_path
        )

        print(
            f"\nCheckpoint saved:"
        )

        print(
            checkpoint_path
        )


# ==================================================
# FINAL SAVE
# ==================================================

agent.save(
    UPGRADED_MODEL_PATH
)


# ==================================================
# FINAL RESULTS
# ==================================================

total_games = (
    total_wins
    + total_losses
    + total_draws
)


print("\n")
print("=" * 60)
print("             TRAINING COMPLETED")
print("=" * 60)


print(
    f"\nTotal games: "
    f"{total_games}"
)

print(
    f"DDQN X wins: "
    f"{total_wins}"
)

print(
    f"Opponent wins: "
    f"{total_losses}"
)

print(
    f"Draws: "
    f"{total_draws}"
)


print(
    f"\nOverall DDQN win rate: "
    f"{(
        total_wins
        / total_games
    ) * 100:.2f}%"
)


# ==================================================
# MINIMAX FINAL RESULTS
# ==================================================

if minimax_games > 0:

    print("\nAgainst Minimax:")

    print(
        f"Games: "
        f"{minimax_games}"
    )

    print(
        f"DDQN wins: "
        f"{minimax_wins}"
    )

    print(
        f"Minimax wins: "
        f"{minimax_losses}"
    )

    print(
        f"Draws: "
        f"{minimax_draws}"
    )

    print(
        f"DDQN win rate: "
        f"{(
            minimax_wins
            / minimax_games
        ) * 100:.2f}%"
    )


# ==================================================
# RANDOM FINAL RESULTS
# ==================================================

if random_games > 0:

    print("\nAgainst Random:")

    print(
        f"Games: "
        f"{random_games}"
    )

    print(
        f"DDQN wins: "
        f"{random_wins}"
    )

    print(
        f"Random wins: "
        f"{random_losses}"
    )

    print(
        f"Draws: "
        f"{random_draws}"
    )

    print(
        f"DDQN win rate: "
        f"{(
            random_wins
            / random_games
        ) * 100:.2f}%"
    )


print(
    f"\nFinal epsilon: "
    f"{agent.epsilon:.4f}"
)


print(
    "\nUpgraded model saved at:"
)

print(
    UPGRADED_MODEL_PATH
)

print("\n" + "=" * 60)