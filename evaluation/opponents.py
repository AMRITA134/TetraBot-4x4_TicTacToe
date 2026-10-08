import random
import math


# ============================================================
# RANDOM OPPONENT
# ============================================================

def random_opponent(board):
    """
    Select a random valid move.
    """

    valid_moves = board.get_valid_moves()

    if not valid_moves:
        return None

    return random.choice(valid_moves)


# ============================================================
# RULE-BASED OPPONENT
# ============================================================

def rule_based_opponent(board, player=-1):
    """
    Rule-based opponent.

    Priority:
        1. Win if possible
        2. Block opponent's winning move
        3. Take center
        4. Take a corner
        5. Random move
    """

    valid_moves = board.get_valid_moves()

    if not valid_moves:
        return None

    # --------------------------------------------------------
    # 1. Try to win
    # --------------------------------------------------------

    for move in valid_moves:

        board.state[move] = player

        if board.check_winner() == player:

            board.state[move] = 0

            return move

        board.state[move] = 0

    # --------------------------------------------------------
    # 2. Block opponent
    # --------------------------------------------------------

    opponent = -player

    for move in valid_moves:

        board.state[move] = opponent

        if board.check_winner() == opponent:

            board.state[move] = 0

            return move

        board.state[move] = 0

    # --------------------------------------------------------
    # 3. Take center cells
    # --------------------------------------------------------

    center_cells = [5, 6, 9, 10]

    available_centers = [
        move
        for move in center_cells
        if move in valid_moves
    ]

    if available_centers:

        return random.choice(
            available_centers
        )

    # --------------------------------------------------------
    # 4. Take corners
    # --------------------------------------------------------

    corners = [0, 3, 12, 15]

    available_corners = [
        move
        for move in corners
        if move in valid_moves
    ]

    if available_corners:

        return random.choice(
            available_corners
        )

    # --------------------------------------------------------
    # 5. Random valid move
    # --------------------------------------------------------

    return random.choice(valid_moves)


# ============================================================
# MINIMAX
# ============================================================

def minimax(
    board,
    depth,
    maximizing_player,
    ai_player=-1
):
    """
    Depth-limited Minimax.

    ai_player:
        -1 -> O

    maximizing_player:
        True  -> maximize
        False -> minimize
    """

    winner = board.check_winner()

    # --------------------------------------------------------
    # Terminal states
    # --------------------------------------------------------

    if winner == ai_player:

        return 1000 + depth

    if winner == -ai_player:

        return -1000 - depth

    if board.is_full():

        return 0

    # Depth limit
    if depth == 0:

        return evaluate_board(
            board,
            ai_player
        )

    valid_moves = board.get_valid_moves()

    # --------------------------------------------------------
    # Maximizing
    # --------------------------------------------------------

    if maximizing_player:

        best_score = -math.inf

        for move in valid_moves:

            board.state[move] = ai_player

            score = minimax(
                board,
                depth - 1,
                False,
                ai_player
            )

            board.state[move] = 0

            best_score = max(
                best_score,
                score
            )

        return best_score

    # --------------------------------------------------------
    # Minimizing
    # --------------------------------------------------------

    else:

        best_score = math.inf

        for move in valid_moves:

            board.state[move] = -ai_player

            score = minimax(
                board,
                depth - 1,
                True,
                ai_player
            )

            board.state[move] = 0

            best_score = min(
                best_score,
                score
            )

        return best_score


# ============================================================
# BOARD EVALUATION
# ============================================================

def evaluate_board(board, ai_player):
    """
    Simple heuristic evaluation.

    Rewards:
        2 or 3 marks in a potential winning line.

    Penalizes:
        Opponent control of lines.
    """

    opponent = -ai_player

    score = 0

    for line in board.WINNING_LINES:

        values = [
            board.state[position]
            for position in line
        ]

        ai_count = values.count(ai_player)
        opponent_count = values.count(opponent)

        empty_count = values.count(0)

        # AI potential
        if opponent_count == 0:

            if ai_count == 3 and empty_count == 1:
                score += 50

            elif ai_count == 2 and empty_count == 2:
                score += 10

            elif ai_count == 1 and empty_count == 3:
                score += 1

        # Opponent potential
        if ai_count == 0:

            if opponent_count == 3 and empty_count == 1:
                score -= 50

            elif opponent_count == 2 and empty_count == 2:
                score -= 10

            elif opponent_count == 1 and empty_count == 3:
                score -= 1

    return score


# ============================================================
# MINIMAX MOVE
# ============================================================

def minimax_opponent(
    board,
    depth=3,
    player=-1
):
    """
    Select the best move using depth-limited Minimax.
    """

    valid_moves = board.get_valid_moves()

    if not valid_moves:
        return None

    best_move = None
    best_score = -math.inf

    for move in valid_moves:

        board.state[move] = player

        score = minimax(
            board,
            depth - 1,
            False,
            player
        )

        board.state[move] = 0

        if score > best_score:

            best_score = score
            best_move = move

    return best_move