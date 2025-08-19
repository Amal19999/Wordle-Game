import os
import argparse
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

# Load words from files
def load_words(solutions_path, guesses_path):
    if not os.path.isfile(solutions_path):
        raise FileNotFoundError(f"Solution file not found: {solutions_path}")
    if not os.path.isfile(guesses_path):
        raise FileNotFoundError(f"Guess file not found: {guesses_path}")
    with open(solutions_path) as f:
        solution_words = [line.strip().lower() for line in f if line.strip()]
    with open(guesses_path) as f:
        guess_words = [line.strip().lower() for line in f if line.strip()]
    return solution_words, guess_words

# Feedback color mapping
COLORS = {
    'G': '#6aaa64',  # green
    'Y': '#c9b458',  # yellow
    'B': '#787c7e'   # gray
}

# Compute Wordle-style feedback on-the-fly
def get_feedback(guess, secret):
    fb = [''] * 5
    secret_chars = list(secret)
    # First pass: greens
    for i, c in enumerate(guess):
        if c == secret[i]:
            fb[i] = 'G'
            secret_chars[i] = None
    # Second pass: yellows and blanks
    for i, c in enumerate(guess):
        if fb[i] == '':
            if c in secret_chars:
                fb[i] = 'Y'
                secret_chars[secret_chars.index(c)] = None
            else:
                fb[i] = 'B'
    return fb

# Bellman DP solver with candidate reduction (solutions-only)
class BellmanSolver:
    def __init__(self, solutions):
        self.solutions = solutions

    @lru_cache(maxsize=None)
    def dp(self, possible_tuple):
        possible = list(possible_tuple)
        if len(possible) <= 1:
            return [possible[0]] if possible else []
        best_guess, best_worst = None, float('inf')
        # Restrict guesses to current possible solutions
        for g in possible:
            partition_counts = {}
            for s in possible:
                feedback = tuple(get_feedback(g, s))
                partition_counts[feedback] = partition_counts.get(feedback, 0) + 1
            worst = max(partition_counts.values())
            if worst < best_worst:
                best_worst = worst
                best_guess = g
                if best_worst == 1:
                    break
        return [best_guess]

    def solve(self, secret):
        possible = set(self.solutions)
        guesses = []
        for _ in range(6):
            guess = self.dp(tuple(sorted(possible)))[0]
            guesses.append(guess)
            if guess == secret:
                break
            fb = tuple(get_feedback(guess, secret))
            possible = {w for w in possible if tuple(get_feedback(guess, w)) == fb}
        return guesses

# OCT-H solver with CART warm start
def solve_wordle_octh(secret, solutions, guesses):
    remaining = list(solutions)
    sequence = []
    for _ in range(6):
        if len(remaining) == 1:
            sequence.append(remaining[0])
            break
        # Build feature matrix
        X, y = [], []
        for w in remaining:
            vec = np.zeros(26 * 5, dtype=int)
            for i, ch in enumerate(w):
                vec[ord(ch) - ord('a') + 26 * i] = 1
            X.append(vec)
            y.append(1 if w == secret else 0)
        X, y = np.array(X), np.array(y)
        # If only one class present
        if len(set(y)) == 1:
            candidate = remaining[0]
            sequence.append(candidate)
            if candidate == secret:
                break
            fb = get_feedback(candidate, secret)
            remaining = [w for w in remaining if get_feedback(candidate, w) == fb]
            continue
        # Train CART classifier
        clf = DecisionTreeClassifier(max_depth=10).fit(X, y)
        # Score all guess words
        scores = []
        for g in guesses:
            vec = np.zeros(26 * 5, dtype=int)
            for i, ch in enumerate(g):
                vec[ord(ch) - ord('a') + 26 * i] = 1
            prob = clf.predict_proba([vec])[0][list(clf.classes_).index(1)]
            scores.append((prob, g))
        guess = max(scores)[1]
        sequence.append(guess)
        if guess == secret:
            break
        fb = get_feedback(guess, secret)
        remaining = [w for w in remaining if get_feedback(guess, w) == fb]
    return sequence

# Visualization: two side-by-side 6×5 grids
def visualize(bellman_seq, octh_seq, secret):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    titles = ['DP Policy', 'OCT-H Policy']
    for ax, seq, title in zip(axes, [bellman_seq, octh_seq], titles):
        ax.set_title(title)
        ax.axis('off')
        # Draw 6 rows × 5 cols
        for i in range(6):
            for j in range(5):
                if i < len(seq):
                    fb = get_feedback(seq[i], secret)[j]
                    color = COLORS[fb]
                    letter = seq[i][j].upper()
                else:
                    color = 'white'
                    letter = ''
                rect = plt.Rectangle((j, 5 - i - 1), 1, 1,
                                     facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                ax.text(j + 0.5, 5 - i - 1 + 0.5,
                        letter, ha='center', va='center', fontsize=16, weight='bold')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wordle solver with Bellman DP and OCT-H methods'
    )
    parser.add_argument(
        '--solutions', default='data/wordle_solutions.txt',
        help='Path to solution words file'
    )
    parser.add_argument(
        '--guesses', default='data/wordle_guesses.txt',
        help='Path to guess words file'
    )
    args = parser.parse_args()
    sol, gue = load_words(args.solutions, args.guesses)
    secret = input('Enter the secret word: ').strip().lower()
    if secret not in sol:
        raise ValueError(f"Secret '{secret}' not in solution list")
    bellman_seq = BellmanSolver(sol).solve(secret)
    octh_seq = solve_wordle_octh(secret, sol, gue)
    visualize(bellman_seq, octh_seq, secret)

