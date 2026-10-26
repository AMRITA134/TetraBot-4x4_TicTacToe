
import os
import sys

# ------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.tictactoe_env import TicTacToe4x4Env
from agent.ddqn_agent import DDQNAgent


# ------------------------------------------------------
# EVALUATION SETTINGS
# ------------------------------------------------------

NUM_GAMES = 100

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ddqn_4x4_x_minimax_shaped.pth",
)


# ------------------------------------------------------
# LOAD TRAINED AGENT
# ------------------------------------------------------

def load_agent():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    agent = DDQNAgent(
        state_size=16,
        action_size=16,
    )

    agent.load(MODEL_PATH)

    # Disable random exploration during evaluation.
    agent.epsilon = 0

    agent.policy_net.eval()
    agent.target_net.eval()

    return agent


# ------------------------------------------------------
# EVALUATE AGAINST ONE OPPONENT
# ------------------------------------------------------

def evaluate(opponent_type, minimax_depth=3, num_games=None):
    if num_games is None:
        num_games = NUM_GAMES

    if num_games <= 0:
        raise ValueError("num_games must be greater than zero.")

    agent = load_agent()

    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        env = TicTacToe4x4Env(
            opponent_type=opponent_type,
            minimax_depth=minimax_depth,
        )

        state = env.reset()
        done = False
        info = {}

        while not done:
            valid_actions = env.get_valid_actions()

            # Normally a full board ends the game inside step().
            # This check also prevents selecting from no valid moves.
            if not valid_actions:
                draws += 1
                break

            action = agent.select_action(
                state,
                valid_actions,
            )

            state, reward, done, info = env.step(action)

        else:
            # The while loop ended normally because done=True.
            pass

        if info:
            winner = info.get("winner")

            if winner == "X":
                wins += 1
            elif winner == "O":
                losses += 1
            elif winner == "draw":
                draws += 1

    win_rate = wins / num_games * 100
    loss_rate = losses / num_games * 100
    draw_rate = draws / num_games * 100

    print("\n" + "=" * 55)
    print(f"Opponent: {opponent_type.upper()}")

    if opponent_type == "minimax":
        print(f"Minimax depth: {minimax_depth}")

    print(f"Games: {num_games}")
    print(f"Wins: {wins} ({win_rate:.2f}%)")
    print(f"Losses: {losses} ({loss_rate:.2f}%)")
    print(f"Draws: {draws} ({draw_rate:.2f}%)")
    print("=" * 55)

    return {
        "opponent": opponent_type,
        "depth": minimax_depth if opponent_type == "minimax" else None,
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
    }


# ------------------------------------------------------
# RUN ALL EVALUATIONS
# ------------------------------------------------------

def main():
    print("\nDDQN 4x4 Tic-Tac-Toe Evaluation")
    print(f"Model: {MODEL_PATH}")
    print(f"Games per opponent: {NUM_GAMES}")

    results = []

    results.append(
        evaluate(
            opponent_type="random",
            num_games=NUM_GAMES,
        )
    )

    results.append(
        evaluate(
            opponent_type="rule",
            num_games=NUM_GAMES,
        )
    )

    results.append(
        evaluate(
            opponent_type="minimax",
            minimax_depth=5,
            num_games=NUM_GAMES,
        )
    )

    # Summary table
    print("\n\nFINAL COMPARISON")
    print("-" * 75)
    print(
        f"{'Opponent':<15}"
        f"{'Wins %':>12}"
        f"{'Losses %':>12}"
        f"{'Draws %':>12}"
    )
    print("-" * 75)

    for result in results:
        print(
            f"{result['opponent']:<15}"
            f"{result['win_rate']:>11.2f}%"
            f"{result['loss_rate']:>11.2f}%"
            f"{result['draw_rate']:>11.2f}%"
        )

    print("-" * 75)


if __name__ == "__main__":
    main()
