import numpy as np

from environment.board import Board

from evaluation.opponents import (
    random_opponent,
    rule_based_opponent,
    minimax_opponent
)


class TicTacToe4x4Env:
    """
    4x4 Tic-Tac-Toe Reinforcement Learning Environment.

    X (1)  -> DDQN agent
    O (-1) -> Opponent

    Supported opponents:
        - random
        - rule
        - minimax

    State representation:
        0  = Empty
        1  = X
       -1  = O

    Actions:
        0-15 -> board positions

    Reward structure:

        +10 -> X wins
        -10 -> X loses
        +1  -> Draw
        -10 -> Invalid move

        +2  -> X blocks an immediate O winning move
        +2  -> X creates an immediate winning threat

        -2  -> X leaves an immediate O winning move available
         0  -> Normal move
    """

    def __init__(self, opponent_type="random"):

        # --------------------------------------------------
        # Board
        # --------------------------------------------------

        self.board = Board()

        # --------------------------------------------------
        # Players
        # --------------------------------------------------

        # DDQN agent
        self.agent_player = 1

        # Opponent
        self.opponent_player = -1

        # --------------------------------------------------
        # Opponent type
        # --------------------------------------------------

        self.opponent_type = opponent_type

        # --------------------------------------------------
        # Game status
        # --------------------------------------------------

        self.done = False

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):
        """
        Reset the board and start a new game.

        Returns:
            state: 16-element NumPy array
        """

        self.board.reset()

        self.done = False

        return self.get_state()

    # ======================================================
    # STATE
    # ======================================================

    def get_state(self):
        """
        Return the current board state.

        Returns:
            NumPy array of length 16.
        """

        return self.board.get_state()

    # ======================================================
    # VALID ACTIONS
    # ======================================================

    def get_valid_actions(self):
        """
        Return all currently available cells.

        Returns:
            List containing values from 0 to 15.
        """

        return self.board.get_valid_moves()

    # ======================================================
    # IMMEDIATE WIN DETECTION
    # ======================================================

    def has_immediate_winning_move(self, player):
        """
        Check whether the specified player currently has
        a move that would immediately win the game.

        Parameters:
            player:
                1  -> X
               -1  -> O

        Returns:
            True  -> an immediate winning move exists
            False -> no immediate winning move exists
        """

        valid_moves = self.board.get_valid_moves()

        for move in valid_moves:

            self.board.state[move] = player

            wins = (
                self.board.check_winner()
                == player
            )

            self.board.state[move] = 0

            if wins:
                return True

        return False

    # ======================================================
    # CHECK SPECIFIC IMMEDIATE WIN
    # ======================================================

    def is_immediate_winning_move(self, action, player):
        """
        Check whether placing 'player' at 'action'
        immediately wins the game.

        Parameters:
            action: board position
            player: 1 or -1

        Returns:
            True / False
        """

        if not self.board.is_valid_move(action):
            return False

        self.board.state[action] = player

        wins = (
            self.board.check_winner()
            == player
        )

        self.board.state[action] = 0

        return wins

    # ======================================================
    # CHECK WHETHER ACTION BLOCKS O
    # ======================================================

    def blocks_opponent_threat(self, action):
        """
        Determine whether the DDQN action blocks an
        immediate winning opportunity for O.

        Returns:
            True if O had an immediate winning move before
            X played, and X's action removes that threat.
        """

        # Check whether O currently has an immediate win
        opponent_had_threat = self.has_immediate_winning_move(
            self.opponent_player
        )

        if not opponent_had_threat:
            return False

        # Temporarily make X's move
        self.board.state[action] = self.agent_player

        # Check whether O still has an immediate win
        opponent_still_has_threat = (
            self.has_immediate_winning_move(
                self.opponent_player
            )
        )

        # Undo X move
        self.board.state[action] = 0

        return not opponent_still_has_threat

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):
        """
        Perform one DDQN action.

        The DDQN always plays X.

        Sequence:

            1. Check action validity
            2. Check whether X blocks an O threat
            3. Make X move
            4. Check X win
            5. Check draw
            6. Check whether X created a threat
            7. Check whether X left an O threat
            8. Opponent selects O move
            9. Check O win
            10. Check draw
            11. Return next state

        Returns:
            next_state
            reward
            done
            info
        """

        # --------------------------------------------------
        # Check if game is already finished
        # --------------------------------------------------

        if self.done:

            return (
                self.get_state(),
                0,
                True,
                {
                    "message": "Game is already over"
                }
            )

        # --------------------------------------------------
        # Check whether action is valid
        # --------------------------------------------------

        if not self.board.is_valid_move(action):

            self.done = True

            return (
                self.get_state(),
                -10,
                True,
                {
                    "winner": "O",
                    "invalid_move": True
                }
            )

        # ==================================================
        # ANALYZE X'S MOVE BEFORE MAKING IT
        # ==================================================

        # Was O threatening to win?
        blocks_threat = self.blocks_opponent_threat(
            action
        )

        # ==================================================
        # X / DDQN MOVE
        # ==================================================

        self.board.make_move(
            action,
            self.agent_player
        )

        # --------------------------------------------------
        # Check whether X won
        # --------------------------------------------------

        if (
            self.board.check_winner()
            == self.agent_player
        ):

            self.done = True

            return (
                self.get_state(),
                10,
                True,
                {
                    "winner": "X",
                    "winning_move": True,
                    "blocked_threat": blocks_threat
                }
            )

        # --------------------------------------------------
        # Check whether board is full
        # --------------------------------------------------

        if self.board.is_full():

            self.done = True

            return (
                self.get_state(),
                1,
                True,
                {
                    "winner": "draw",
                    "blocked_threat": blocks_threat
                }
            )

        # ==================================================
        # CHECK TACTICAL EFFECT OF X'S MOVE
        # ==================================================

        # Did X create an immediate winning threat?
        created_threat = self.has_immediate_winning_move(
            self.agent_player
        )

        # Did X leave an immediate winning move for O?
        opponent_has_threat = self.has_immediate_winning_move(
            self.opponent_player
        )

        # --------------------------------------------------
        # Determine shaped reward
        # --------------------------------------------------

        reward = 0

        if blocks_threat:
            reward += 2

        if created_threat:
            reward += 2

        if opponent_has_threat:
            reward -= 2

        # ==================================================
        # O / OPPONENT MOVE
        # ==================================================

        opponent_action = self.opponent_move()

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if opponent_action is None:

            self.done = True

            return (
                self.get_state(),
                1,
                True,
                {
                    "winner": "draw",
                    "blocked_threat": blocks_threat,
                    "created_threat": created_threat
                }
            )

        # --------------------------------------------------
        # Make O move
        # --------------------------------------------------

        self.board.make_move(
            opponent_action,
            self.opponent_player
        )

        # --------------------------------------------------
        # Check whether O won
        # --------------------------------------------------

        if (
            self.board.check_winner()
            == self.opponent_player
        ):

            self.done = True

            return (
                self.get_state(),
                -10,
                True,
                {
                    "winner": "O",
                    "blocked_threat": blocks_threat,
                    "created_threat": created_threat,
                    "opponent_winning_move": opponent_action
                }
            )

        # --------------------------------------------------
        # Check whether board is full
        # --------------------------------------------------

        if self.board.is_full():

            self.done = True

            return (
                self.get_state(),
                1,
                True,
                {
                    "winner": "draw",
                    "blocked_threat": blocks_threat,
                    "created_threat": created_threat
                }
            )

        # ==================================================
        # GAME CONTINUES
        # ==================================================

        return (
            self.get_state(),
            reward,
            False,
            {
                "winner": None,
                "blocked_threat": blocks_threat,
                "created_threat": created_threat,
                "opponent_has_threat": opponent_has_threat
            }
        )

    # ======================================================
    # OPPONENT MOVE
    # ======================================================

    def opponent_move(self):
        """
        Select O's move according to opponent_type.

        Supported:
            random
            rule
            minimax
        """

        # --------------------------------------------------
        # RANDOM
        # --------------------------------------------------

        if self.opponent_type == "random":

            return random_opponent(
                self.board
            )

        # --------------------------------------------------
        # RULE BASED
        # --------------------------------------------------

        elif self.opponent_type == "rule":

            return rule_based_opponent(
                self.board,
                self.opponent_player
            )

        # --------------------------------------------------
        # MINIMAX
        # --------------------------------------------------

        elif self.opponent_type == "minimax":

            return minimax_opponent(
                self.board,
                depth=3,
                player=self.opponent_player
            )

        # --------------------------------------------------
        # UNKNOWN OPPONENT
        # --------------------------------------------------

        else:

            raise ValueError(
                f"Unknown opponent type: "
                f"{self.opponent_type}"
            )

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):
        """
        Display the current board.
        """

        self.board.display()

    # ======================================================
    # GAME STATUS
    # ======================================================

    def is_done(self):
        """
        Check whether the game has ended.

        Returns:
            True  -> game over
            False -> game still running
        """

        return self.done