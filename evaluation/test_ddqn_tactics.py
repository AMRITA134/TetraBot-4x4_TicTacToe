
import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports when running from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.ddqn_agent import DDQNAgent


MODEL_PATH = PROJECT_ROOT / "models" / "ddqn_4x4_x_minimax_shaped.pth"


# All possible winning lines on a 4x4 board
WINNING_LINES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [3, 6, 9, 12],
]


def make_agent():
    """Load the saved DDQN checkpoint for deterministic evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DDQNAgent(state_size=16, action_size=16)
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Load the checkpoint in the format used by this project.
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        agent.policy_net.load_state_dict(checkpoint["policy_net"])

        if "target_net" in checkpoint:
            agent.target_net.load_state_dict(checkpoint["target_net"])

        if "learn_step" in checkpoint:
            agent.learn_step = checkpoint["learn_step"]

        if "epsilon" in checkpoint:
            agent.epsilon = checkpoint["epsilon"]
    else:
        # Support a checkpoint containing only the model weights.
        agent.policy_net.load_state_dict(checkpoint)

    agent.policy_net.to(device)
    agent.target_net.to(device)

    # Disable exploration for repeatable evaluation.
    agent.epsilon = 0.0
    agent.policy_net.eval()
    agent.target_net.eval()

    return agent


def get_valid_actions(board):
    """Return indices of all empty cells."""
    return [i for i, value in enumerate(board) if value == 0]


def find_immediate_winning_moves(board, player):
    """
    Return empty cells where the specified player can win immediately.

    player = 1  for X
    player = -1 for O
    """
    winning_moves = []

    for action in get_valid_actions(board):
        test_board = board.copy()
        test_board[action] = player

        for line in WINNING_LINES:
            if all(test_board[cell] == player for cell in line):
                winning_moves.append(action)
                break

    return winning_moves


def select_action(agent, board):
    """Select an action using the existing DDQN agent."""
    valid_actions = get_valid_actions(board)

    if not valid_actions:
        return None

    state = np.asarray(board, dtype=np.float32)

    # Support the two common select_action signatures.
    try:
        action = agent.select_action(state, valid_actions)
    except TypeError:
        action = agent.select_action(state)

    if isinstance(action, (tuple, list, np.ndarray)):
        action = np.asarray(action).item()

    return int(action)


def make_board(rows):
    """Convert four rows into a flat 16-cell board."""
    assert len(rows) == 4, "Board must have four rows."
    assert all(len(row) == 4 for row in rows), (
        "Each board row must contain four cells."
    )

    board = np.array(
        [cell for row in rows for cell in row],
        dtype=np.int8,
    )

    assert all(cell in (-1, 0, 1) for cell in board), (
        "Cells must contain only -1, 0, or 1."
    )

    return board


def run_test(name, board, expected=None, acceptable=None):
    """
    Run an action-selection test.

    expected: exact expected action, if applicable.
    acceptable: list of actions that satisfy the test criterion.
    """
    agent = make_agent()
    valid_actions = get_valid_actions(board)

    print(f"\n{name}")
    print(board.reshape(4, 4))

    if not valid_actions:
        action = select_action(agent, board)
        passed = action is None

        print(f"Selected action: {action}")
        print("PASS" if passed else "FAIL: expected no legal action")
        return passed

    action = select_action(agent, board)

    # Every selected action must be an empty cell.
    legal_action = action in valid_actions
    passed = legal_action

    print(f"Selected cell: {action}")
    print(f"Legal cells: {valid_actions}")

    if expected is not None:
        print(f"Expected cell: {expected}")
        passed = passed and action == expected

    if acceptable is not None:
        print(f"Acceptable cells: {acceptable}")
        passed = passed and action in acceptable

    if not legal_action:
        print("ERROR: The agent selected an occupied or invalid cell.")

    print("PASS" if passed else "FAIL")
    return passed


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    results = []

    # ---------------------------------------------------------
    # Test 1: X has an immediate winning move at cell 3.
    # ---------------------------------------------------------
    board = make_board([
        [1, 1, 1, 0],
        [-1, -1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    results.append(
        run_test("Test 1: Immediate X win", board, expected=3)
    )

    # ---------------------------------------------------------
    # Test 2: O threatens to win at cell 3.
    # X should block that cell.
    # ---------------------------------------------------------
    board = make_board([
        [-1, -1, -1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    results.append(
        run_test("Test 2: Block O's immediate win", board, expected=3)
    )

    # ---------------------------------------------------------
    # Test 3: O has two immediate winning opportunities:
    # cells 3 and 12. One move can block only one threat.
    # ---------------------------------------------------------
    board = make_board([
        [-1, -1, -1, 0],
        [-1, 1, 1, 0],
        [-1, 1, 1, 0],
        [0, 0, 0, 0],
    ])

    o_threats = find_immediate_winning_moves(board, -1)
    print("\nDouble-threat verification")
    print(f"O immediate winning moves: {o_threats}")

    double_threat_setup_valid = set(o_threats) == {3, 12}
    print(
        "Threat setup:",
        "PASS" if double_threat_setup_valid else "FAIL",
    )

    test_result = run_test(
        "Test 3: Respond to a double threat",
        board,
        acceptable=[3, 12],
    )

    results.append(double_threat_setup_valid and test_result)

    # ---------------------------------------------------------
    # Test 4: The agent must select an empty cell.
    # ---------------------------------------------------------
    board = make_board([
        [1, -1, 1, 0],
        [-1, 1, -1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    results.append(
        run_test("Test 4: Avoid occupied cells", board)
    )

    # ---------------------------------------------------------
    # Test 5: Both players have an immediate winning move.
    # X can win at cell 3; O can win at cell 7.
    # This tests whether the model prioritizes its own win.
    # ---------------------------------------------------------
    board = make_board([
        [1, 1, 1, 0],
        [-1, -1, -1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    x_wins = find_immediate_winning_moves(board, 1)
    o_wins = find_immediate_winning_moves(board, -1)

    print("\nTest 5: Verify immediate winning opportunities")
    print(board.reshape(4, 4))
    print(f"X immediate winning moves: {x_wins}")
    print(f"O immediate winning moves: {o_wins}")

    setup_valid = 3 in x_wins and 7 in o_wins
    print(
        "Board setup:",
        "PASS" if setup_valid else "FAIL",
    )

    if setup_valid:
        test_result = run_test(
            "Test 5: Prioritize an immediate win",
            board,
            expected=3,
        )
        results.append(test_result)
    else:
        print("Test 5 skipped because its board setup is invalid.")
        results.append(False)

    # ---------------------------------------------------------
    # Test 6: Cell 15 is the only empty cell.
    # ---------------------------------------------------------
    board = make_board([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, -1, 1, -1],
        [-1, 1, -1, 0],
    ])

    results.append(
        run_test("Test 6: Choose the only empty cell", board, expected=15)
    )

    # ---------------------------------------------------------
    # Test 7: Full board should produce no legal action.
    # This tests the selection helper, not environment termination.
    # ---------------------------------------------------------
    board = make_board([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
    ])

    results.append(
        run_test("Test 7: Full board", board)
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    passed = sum(results)
    failed = len(results) - passed

    print("\n" + "=" * 45)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print("=" * 45)


if __name__ == "__main__":
    main()
