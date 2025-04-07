import pygame
import numpy as np
import random
import time

pygame.init()

# Constants & Display Setup
SIZE = 4
TILE_SIZE = 100
MARGIN = 10
WIDTH = HEIGHT = SIZE * (TILE_SIZE + MARGIN) + MARGIN
FONT = pygame.font.Font(None, 80)

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

# -------------------------
# Game Functions
# -------------------------
def initialize_board():
    board = np.zeros((SIZE, SIZE), dtype=int)
    add_new_tile(board)
    add_new_tile(board)
    return board

def add_new_tile(board):
    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r, c] == 0]
    if empty:
        r, c = random.choice(empty)
        board[r, c] = 2 if random.random() < 0.9 else 4

def compress(board):
    new_board = np.zeros((SIZE, SIZE), dtype=int)
    for r in range(SIZE):
        pos = 0
        for c in range(SIZE):
            if board[r, c] != 0:
                new_board[r, pos] = board[r, c]
                pos += 1
    return new_board

def merge(board):
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r, c] != 0 and board[r, c] == board[r, c + 1]:
                board[r, c] *= 2
                board[r, c + 1] = 0
    return board

def move_left(board):
    board = compress(board)
    board = merge(board)
    return compress(board)

def move_right(board):
    return np.fliplr(move_left(np.fliplr(board)))

def move_up(board):
    return np.rot90(move_left(np.rot90(board, 1)), -1)

def move_down(board):
    return np.rot90(move_left(np.rot90(board, -1)), 1)

def is_game_over(board):
    if np.any(board == 2048):
        return True  # win condition reached
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
    return True

def draw_board(board, screen):
    screen.fill(BACKGROUND_COLOR)
    for r in range(SIZE):
        for c in range(SIZE):
            value = board[r, c]
            color = TILE_COLORS.get(value, (60, 58, 50))
            rect = (c * (TILE_SIZE + MARGIN) + MARGIN,
                    r * (TILE_SIZE + MARGIN) + MARGIN,
                    TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, color, rect)
            if value != 0:
                text = FONT.render(str(value), True, (100, 100, 100))
                text_rect = text.get_rect(center=(c * (TILE_SIZE + MARGIN) + MARGIN + TILE_SIZE // 2,
                                                  r * (TILE_SIZE + MARGIN) + MARGIN + TILE_SIZE // 2))
                screen.blit(text, text_rect)

# -------------------------
# Caching & Helper Functions
# -------------------------
def board_to_tuple(board):
    return tuple(map(tuple, board))

cache = {}  # Global cache for Expectimax evaluations

def potential_merges(board):
    merges = 0
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r, c] != 0 and board[r, c] == board[r, c + 1]:
                merges += 1
    for r in range(SIZE - 1):
        for c in range(SIZE):
            if board[r, c] != 0 and board[r, c] == board[r + 1, c]:
                merges += 1
    return merges

def improved_heuristic(board):
    # Empty cells and max tile
    empty = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    
    # Smoothness: penalize differences between adjacent tiles
    smoothness = 0
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r, c] and board[r, c + 1]:
                smoothness -= abs(board[r, c] - board[r, c + 1])
    for r in range(SIZE - 1):
        for c in range(SIZE):
            if board[r, c] and board[r + 1, c]:
                smoothness -= abs(board[r, c] - board[r + 1, c])
    
    # Monotonicity: reward rows/cols that are non-increasing
    mono_score = 0
    for row in board:
        mono_score += sum(row[i] >= row[i+1] for i in range(SIZE - 1))
    for col in board.T:
        mono_score += sum(col[i] >= col[i+1] for i in range(SIZE - 1))
    
    # Corner bonus: encourage max tile to be in a corner
    corners = [board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]]
    corner_bonus = max_tile * 4 if max_tile in corners else -1500
    
    # Merge bonus: reward potential merges
    merge_bonus = potential_merges(board) * 200
    
    # Pattern bonus: snake/spiral pattern weight matrix
    WEIGHTS = np.array([
        [16,  8,  4,  2],
        [7,   6,  5,  1],
        [3,   2,  1,  0],
        [1,   0, -1, -2]
    ])
    pattern_score = np.sum(board * WEIGHTS)
    
    # Combine factors (tweak multipliers as needed)
    return (empty * 270 +
            smoothness * 1.0 +
            mono_score * 50 +
            corner_bonus +
            merge_bonus +
            pattern_score)

def get_possible_moves(board):
    # Prioritize moves: down > right > left > up
    return [
        (move_down, move_down(board.copy())),
        (move_right, move_right(board.copy())),
        (move_left, move_left(board.copy())),
        (move_up, move_up(board.copy()))
    ]

# -------------------------
# Expectimax with Caching
# -------------------------
def expectimax(board, depth, is_max):
    key = (board_to_tuple(board), depth, is_max)
    if key in cache:
        return cache[key]
    
    if depth == 0 or is_game_over(board):
        score = improved_heuristic(board)
        cache[key] = score
        return score

    if is_max:
        best = -np.inf
        for _, new_board in get_possible_moves(board):
            if np.array_equal(board, new_board):
                continue
            best = max(best, expectimax(new_board, depth - 1, False))
        cache[key] = best
        return best
    else:
        empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r, c] == 0]
        if not empty:
            score = improved_heuristic(board)
            cache[key] = score
            return score
        score = 0
        for r, c in empty:
            for val, prob in [(2, 0.9), (4, 0.1)]:
                board_copy = board.copy()
                board_copy[r, c] = val
                score += prob * expectimax(board_copy, depth - 1, True) / len(empty)
        cache[key] = score
        return score

# -------------------------
# Monte Carlo Rollout for Tie-breaking
# -------------------------
def monte_carlo_rollout(board, num_rollouts=5, rollout_depth=5):
    total = 0
    for _ in range(num_rollouts):
        b = board.copy()
        d = rollout_depth
        while d > 0 and not is_game_over(b):
            moves = get_possible_moves(b)
            if not moves:
                break
            move_func, new_board = random.choice(moves)
            b = new_board
            add_new_tile(b)
            d -= 1
        total += improved_heuristic(b)
    return total / num_rollouts

# -------------------------
# Best Move via Iterative Deepening
# -------------------------
def best_expectimax_move_depth(board, depth):
    moves = get_possible_moves(board)
    candidate_scores = []
    for func, new_board in moves:
        if np.array_equal(board, new_board):
            continue
        score = expectimax(new_board, depth - 1, False)
        candidate_scores.append((score, func))
    # Sort candidates by score descending
    candidate_scores.sort(key=lambda x: x[0], reverse=True)
    if candidate_scores:
        # If top two scores are close, run Monte Carlo rollouts for tie-breaker
        if len(candidate_scores) > 1 and abs(candidate_scores[0][0] - candidate_scores[1][0]) < 50:
            m1 = monte_carlo_rollout(candidate_scores[0][1](board.copy()))
            m2 = monte_carlo_rollout(candidate_scores[1][1](board.copy()))
            best_move = candidate_scores[0][1] if m1 > m2 else candidate_scores[1][1]
        else:
            best_move = candidate_scores[0][1]
        return best_move
    return None

def best_move_iterative(board, time_limit=0.1):
    start_time = time.time()
    best_move = None
    depth = 2  # starting depth
    while time.time() - start_time < time_limit:
        move = best_expectimax_move_depth(board, depth)
        if move is not None:
            best_move = move
        depth += 1
    return best_move

# -------------------------
# Main Game Loop
# -------------------------
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048 AI - Iterative Deepening + Monte Carlo")
    board = initialize_board()
    clock = pygame.time.Clock()
    steps = 0
    global cache
    cache = {}  # Reset cache for each game

    running = True
    while running:
        draw_board(board, screen)
        pygame.display.flip()
        pygame.time.wait(50)

        if np.max(board) >= 2048 or is_game_over(board):
            print(f"Game Over in {steps} steps. Max tile: {np.max(board)}")
            running = False
            continue

        # Use iterative deepening to choose the best move within a time limit
        best_move = best_move_iterative(board, time_limit=0.1)
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
