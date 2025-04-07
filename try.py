import pygame
import numpy as np
import random

# Initialize pygame
pygame.init()

# Game Constants
SIZE = 4  # 4x4 grid
TILE_SIZE = 100
MARGIN = 10
WIDTH = HEIGHT = SIZE * (TILE_SIZE + MARGIN) + MARGIN
FONT = pygame.font.Font(None, 80)

# Colors
BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (205, 192, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

def initialize_board():
    board = np.zeros((SIZE, SIZE), dtype=int)
    add_new_tile(board)
    add_new_tile(board)
    return board

def add_new_tile(board):
    empty_cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r, c] == 0]
    if empty_cells:
        r, c = random.choice(empty_cells)
        board[r, c] = 2 if random.random() < 0.9 else 4

def compress(board):
    new_board = np.zeros((SIZE, SIZE), dtype=int)
    for r in range(SIZE):
        pos = 0
        for c in range(SIZE):
            if board[r, c] != 0:
                new_board[r, pos] = board[r, c]
                board[r,c]=0
                pos += 1
    return new_board

def merge(board):
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r, c] == board[r, c + 1] and board[r, c] != 0:
                board[r, c] *= 2
                board[r, c + 1] = 0
    return board

def move_left(board):
    board = compress(board)
    board = merge(board)
    board = compress(board)
    return board

def move_right(board):
    board = np.fliplr(board)
    board = move_left(board)
    return np.fliplr(board)

def move_up(board):
    board = np.rot90(board, 1)
    board = move_left(board)
    return np.rot90(board, -1)

def move_down(board):
    board = np.rot90(board, -1)
    board = move_left(board)
    return np.rot90(board, 1)

def is_game_over(board):
    if np.any(board == 2048):
        return True  # Win condition
    if np.any(board == 0):
        return False
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r, c] == board[r, c + 1]:
                return False
    for r in range(SIZE - 1):
        for c in range(SIZE):
            if board[r, c] == board[r + 1, c]:
                return False
    return True  # No valid moves left

def draw_board(board, screen):
    screen.fill(BACKGROUND_COLOR) #fills the screen with background color
    for r in range(SIZE):
        for c in range(SIZE):
            value = board[r, c]
            color = TILE_COLORS.get(value, (60, 58, 50))#gets the color from title_colors if not found sets to a default value
            pygame.draw.rect(screen, color, (c * (TILE_SIZE + MARGIN) + MARGIN,
                                             r * (TILE_SIZE + MARGIN) + MARGIN, TILE_SIZE, TILE_SIZE))
            if value != 0:
                text = FONT.render(str(value), True, (100, 100, 100))
                text_rect = text.get_rect(center=((c * (TILE_SIZE + MARGIN) + MARGIN + TILE_SIZE // 2),
                                                  (r * (TILE_SIZE + MARGIN) + MARGIN + TILE_SIZE // 2)))
                screen.blit(text, text_rect)

def heuristic(board):
    empty = np.count_nonzero(board == 0)
    smoothness = 0
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r, c] != 0 and board[r, c+1] != 0:
                smoothness -= abs(board[r, c] - board[r, c+1])
    for r in range(SIZE - 1):
        for c in range(SIZE):
            if board[r, c] != 0 and board[r+1, c] != 0:
                smoothness -= abs(board[r, c] - board[r+1, c])

    mono_score = 0
    for row in board:
        mono_score += sum([row[i] >= row[i+1] for i in range(SIZE - 1)])
    for col in board.T:
        mono_score += sum([col[i] >= col[i+1] for i in range(SIZE - 1)])

    max_tile = np.max(board)
    max_tile_in_corner = board[0, 0] == max_tile or board[0, -1] == max_tile or board[-1, 0] == max_tile or board[-1, -1] == max_tile
    corner_bonus = max_tile * 1.5 if max_tile_in_corner else 0

    return empty * 250 + mono_score * 20 + smoothness + corner_bonus

def get_possible_moves(board):
    moves = [(move_up, move_up(board.copy())),
             (move_left, move_left(board.copy())),
             (move_right, move_right(board.copy())),
             (move_down, move_down(board.copy()))]
    return [(func, b) for func, b in moves if not np.array_equal(board, b)]

def expectimax(board, depth, is_max):
    if depth == 0 or is_game_over(board):
        return heuristic(board)
    if is_max:
        best = -np.inf
        for _, new_board in get_possible_moves(board):
            best = max(best, expectimax(new_board, depth - 1, False))
        return best
    else:
        empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r, c] == 0]
        if not empty:
            return heuristic(board)
        score = 0
        for r, c in empty:
            for val, prob in [(2, 0.9), (4, 0.1)]:
                board_copy = board.copy()
                board_copy[r, c] = val
                score += prob * expectimax(board_copy, depth - 1, True) / len(empty)
        return score

def best_expectimax_move(board):
    empty_count = np.count_nonzero(board == 0)
    depth = 5 if empty_count >= 6 else 4 if empty_count >= 3 else 3
    best_score = -np.inf
    best_move = None
    for func, new_board in get_possible_moves(board):
        score = expectimax(new_board, depth - 1, False)
        if score > best_score:
            best_score = score
            best_move = func
    return best_move

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048 AI - Expectimax Optimized")
    board = initialize_board()
    clock = pygame.time.Clock()
    steps = 0
    running = True

    while running:
        draw_board(board, screen)
        pygame.display.flip()
        pygame.time.wait(80)

        if np.max(board) >= 2048 or is_game_over(board):
            print(f"Game Over in {steps} steps. Max tile: {np.max(board)}")
            running = False
            continue

        best_move = best_expectimax_move(board)
        if best_move:
            board = best_move(board)
            add_new_tile(board)
            steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
