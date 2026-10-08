import os
import sys

# --------------------------------------------------
# Project root
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
# SETTINGS
# ==================================================

NUM_GAMES = 1000

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x_minimax_shaped.pth"
)


# ==================================================
# EVALUATE ONE OPPONENT
# ==================================================

def evaluate(opponent_type):

    print()
    print(
        "======================================"
    )

    print(
        f"Evaluating against: {opponent_type.upper()}"
    )

    print(
        "======================================"
    )

    # ----------------------------------------------
    # Environment
    # ----------------------------------------------

    env = TicTacToe4x4Env(
        opponent_type=opponent_type
    )

    # ----------------------------------------------
    # Agent
    # ----------------------------------------------

    agent = DDQNAgent(
        state_size=16,
        action_size=16,

        learning_rate=0.001,
        gamma=0.99,

        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,

        buffer_size=1000,
        batch_size=64,

        target_update_frequency=100
    )

    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    agent.load(
        MODEL_PATH
    )

    # No exploration
    agent.epsilon = 0.0

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    wins = 0
    losses = 0
    draws = 0

    # ----------------------------------------------
    # Games
    # ----------------------------------------------

    for game in range(
        1,
        NUM_GAMES + 1
    ):

        state = env.reset()

        done = False

        while not done:

            valid_actions = (
                env.get_valid_actions()
            )

            action = agent.select_action(
                state,
                valid_actions
            )

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            state = next_state

        # ------------------------------------------
        # Result
        # ------------------------------------------

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
    # Percentages
    # ----------------------------------------------

    win_rate = (
        wins / NUM_GAMES
    ) * 100

    loss_rate = (
        losses / NUM_GAMES
    ) * 100

    draw_rate = (
        draws / NUM_GAMES
    ) * 100

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    print(
        f"Games : {NUM_GAMES}"
    )

    print(
        f"Wins  : {wins}"
    )

    print(
        f"Losses: {losses}"
    )

    print(
        f"Draws : {draws}"
    )

    print(
        f"Win rate : {win_rate:.2f}%"
    )

    print(
        f"Loss rate: {loss_rate:.2f}%"
    )

    print(
        f"Draw rate: {draw_rate:.2f}%"
    )

    return (
        win_rate,
        loss_rate,
        draw_rate
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # Random opponent
    random_results = evaluate(
        "random"
    )

    # Rule-based opponent
    rule_results = evaluate(
        "rule"
    )

    # Minimax opponent
    minimax_results = evaluate(
        "minimax"
    )

    # ----------------------------------------------
    # Final comparison
    # ----------------------------------------------

    print()
    print(
        "======================================"
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "======================================"
    )

    print(
        f"Random   -> "
        f"Win {random_results[0]:.2f}% | "
        f"Loss {random_results[1]:.2f}% | "
        f"Draw {random_results[2]:.2f}%"
    )

    print(
        f"Rule     -> "
        f"Win {rule_results[0]:.2f}% | "
        f"Loss {rule_results[1]:.2f}% | "
        f"Draw {rule_results[2]:.2f}%"
    )

    print(
        f"Minimax  -> "
        f"Win {minimax_results[0]:.2f}% | "
        f"Loss {minimax_results[1]:.2f}% | "
        f"Draw {minimax_results[2]:.2f}%"
    )