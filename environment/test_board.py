from board import Board


board = Board()

print("Initial board:")
board.display()

print("Valid moves:")
print(board.get_valid_moves())

print("Making moves...")

board.make_move(0, 1)    # X
board.make_move(1, -1)   # O
board.make_move(5, 1)    # X
board.make_move(2, -1)   # O
board.make_move(10, 1)   # X
board.make_move(3, -1)   # O
board.make_move(15, 1)   # X

board.display()

print("Winner:", board.check_winner())
print("Game over:", board.is_terminal())