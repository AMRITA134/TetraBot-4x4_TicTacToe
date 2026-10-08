import numpy as np


class Board:
    """
    4x4 Tic-Tac-Toe board.

    Board representation:
        0  -> Empty
        1  -> X
       -1  -> O
    """

    SIZE = 4

    # All possible winning combinations
    WINNING_LINES = [
        # Rows
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],

        # Columns
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],

        # Diagonals
        [0, 5, 10, 15],
        [3, 6, 9, 12]
    ]

    def __init__(self):
        """Create an empty board."""
        self.reset()

    def reset(self):
        """Reset the board to an empty state."""
        self.state = np.zeros(16, dtype=np.int8)

    def make_move(self, position, player):
        """
        Place a player's mark on the board.

        Parameters:
            position: Cell number from 0 to 15
            player: 1 for X, -1 for O

        Returns:
            True if the move was successful.
            False if the move was invalid.
        """

        # Check position
        if position < 0 or position >= 16:
            return False

        # Check player
        if player not in [1, -1]:
            return False

        # Check whether cell is already occupied
        if self.state[position] != 0:
            return False

        # Make the move
        self.state[position] = player

        return True

    def is_valid_move(self, position):
        """Check whether a cell is available."""
        return (
            0 <= position < 16
            and self.state[position] == 0
        )

    def get_valid_moves(self):
        """Return a list of all currently available cells."""
        return [
            position
            for position in range(16)
            if self.is_valid_move(position)
        ]

    def check_winner(self):
        """
        Check whether a player has won.

        Returns:
            1  -> X wins
           -1  -> O wins
            0  -> No winner
        """

        for line in self.WINNING_LINES:
            values = self.state[line]

            # All four cells belong to the same player
            if values[0] != 0 and np.all(values == values[0]):
                return int(values[0])

        return 0

    def is_full(self):
        """Check whether the board has no empty cells."""
        return not np.any(self.state == 0)

    def is_draw(self):
        """Check whether the game is a draw."""
        return self.is_full() and self.check_winner() == 0

    def is_terminal(self):
        """Check whether the game has ended."""
        return self.check_winner() != 0 or self.is_full()

    def get_state(self):
        """Return a copy of the current board state."""
        return self.state.copy()

    def display(self):
        """Print the board in a human-readable format."""

        symbols = {
            1: "X",
            -1: "O",
            0: " "
        }

        print()

        for row in range(4):
            cells = []

            for col in range(4):
                position = row * 4 + col
                cells.append(symbols[self.state[position]])

            print(" | ".join(cells))

            if row < 3:
                print("--+---+---+--")

        print()