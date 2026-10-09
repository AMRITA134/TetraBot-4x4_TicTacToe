
import numpy as np

from environment.board import Board
from evaluation.opponents import (
    random_opponent,
    rule_based_opponent,
    minimax_opponent,
)


class TicTacToe4x4Env:
    """
    4x4 Tic-Tac-Toe Reinforcement Learning Environment.

    X (1)  -> DDQN agent
    O (-1) -> Opponent

    Supported opponents:
        random, rule, minimax

    Rewards:
        +10 : X wins
        -10 : X loses
        +1  : Draw
        -10 : Invalid move
        +2  : Blocks an immediate O winning threat
        +2  : Creates an immediate X winning threat
        -2  : Leaves an immediate O winning threat
    """

    def __init__(self, opponent_type="random", minimax_depth=3):
        self.board = Board()

        self.agent_player = 1
        self.opponent_player = -1

        self.opponent_type = opponent_type
        self.minimax_depth = max(1, int(minimax_depth))

        self.done = False

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------

    def reset(self):
        self.board.reset()
        self.done = False
        return self.get_state()

    # ------------------------------------------------------
    # STATE AND VALID ACTIONS
    # ------------------------------------------------------

    def get_state(self):
        return self.board.get_state()

    def get_valid_actions(self):
        return self.board.get_valid_moves()

    # ------------------------------------------------------
    # IMMEDIATE WIN DETECTION
    # ------------------------------------------------------

    def has_immediate_winning_move(self, player):
        """Return True if player can win with one move."""

        for move in self.board.get_valid_moves():
            self.board.state[move] = player

            wins = self.board.check_winner() == player

            self.board.state[move] = 0

            if wins:
                return True

        return False

    def is_immediate_winning_move(self, action, player):
        """Check whether action immediately wins for player."""

        if not self.board.is_valid_move(action):
            return False

        self.board.state[action] = player
        wins = self.board.check_winner() == player
        self.board.state[action] = 0

        return wins

    def blocks_opponent_threat(self, action):
        """
        Return True if O had an immediate winning opportunity
        and X's move removes all immediate O winning moves.
        """

        opponent_had_threat = self.has_immediate_winning_move(
            self.opponent_player
        )

        if not opponent_had_threat:
            return False

        self.board.state[action] = self.agent_player

        opponent_still_has_threat = self.has_immediate_winning_move(
            self.opponent_player
        )

        self.board.state[action] = 0

        return not opponent_still_has_threat

    # ------------------------------------------------------
    # STEP
    # ------------------------------------------------------

    def step(self, action):
        """Execute X's move, then O's move if the game continues."""

        if self.done:
            return (
                self.get_state(),
                0,
                True,
                {"message": "Game is already over"},
            )

        if not self.board.is_valid_move(action):
            self.done = True
            return (
                self.get_state(),
                -10,
                True,
                {
                    "winner": "O",
                    "invalid_move": True,
                },
            )

        # Check whether X's move blocks an existing O threat.
        blocks_threat = self.blocks_opponent_threat(action)

        # ---------------- X / DDQN move ----------------

        self.board.make_move(action, self.agent_player)

        if self.board.check_winner() == self.agent_player:
            self.done = True
            return (
                self.get_state(),
                10,
                True,
                {
                    "winner": "X",
                    "winning_move": True,
                    "blocked_threat": blocks_threat,
                },
            )

        if self.board.is_full():
            self.done = True
            return (
                self.get_state(),
                1,
                True,
                {
                    "winner": "draw",
                    "blocked_threat": blocks_threat,
                },
            )

        # ---------------- Tactical reward shaping ----------------

        created_threat = self.has_immediate_winning_move(
            self.agent_player
        )

        opponent_has_threat = self.has_immediate_winning_move(
            self.opponent_player
        )

        reward = 0

        if blocks_threat:
            reward += 2

        if created_threat:
            reward += 2

        if opponent_has_threat:
            reward -= 2

        # ---------------- O / Opponent move ----------------

        opponent_action = self.opponent_move()

        if opponent_action is None:
            self.done = True
            return (
                self.get_state(),
                1,
                True,
                {
                    "winner": "draw",
                    "blocked_threat": blocks_threat,
                    "created_threat": created_threat,
                },
            )

        self.board.make_move(
            opponent_action,
            self.opponent_player,
        )

        if self.board.check_winner() == self.opponent_player:
            self.done = True
            return (
                self.get_state(),
                -10,
                True,
                {
                    "winner": "O",
                    "blocked_threat": blocks_threat,
                    "created_threat": created_threat,
                    "opponent_winning_move": opponent_action,
                },
            )

        if self.board.is_full():
            self.done = True
            return (
                self.get_state(),
                1,
                True,
                {
                    "winner": "draw",
                    "blocked_threat": blocks_threat,
                    "created_threat": created_threat,
                },
            )

        return (
            self.get_state(),
            reward,
            False,
            {
                "winner": None,
                "blocked_threat": blocks_threat,
                "created_threat": created_threat,
                "opponent_has_threat": opponent_has_threat,
            },
        )

    # ------------------------------------------------------
    # OPPONENT MOVE
    # ------------------------------------------------------

    def opponent_move(self):
        """Choose O's move using the configured opponent."""

        if self.opponent_type == "random":
            return random_opponent(self.board)

        if self.opponent_type == "rule":
            return rule_based_opponent(
                self.board,
                player=self.opponent_player,
            )

        if self.opponent_type == "minimax":
            return minimax_opponent(
                self.board,
                depth=self.minimax_depth,
                player=self.opponent_player,
            )

        raise ValueError(
            f"Unknown opponent type: {self.opponent_type!r}. "
            "Use 'random', 'rule', or 'minimax'."
        )

    # ------------------------------------------------------
    # RENDER AND STATUS
    # ------------------------------------------------------

    def render(self):
        """Print the current board."""

        symbols = {
            0: ".",
            1: "X",
            -1: "O",
        }

        state = self.board.get_state()

        for row in range(4):
            start = row * 4
            print(
                " | ".join(
                    symbols[int(value)]
                    for value in state[start:start + 4]
                )
            )

            if row < 3:
                print("-" * 13)

        print()

    def is_done(self):
        return self.done
