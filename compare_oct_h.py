import os
import argparse
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ——— Wordle feedback helpers ———

# Return status list ['G','Y','B']
def get_status(guess, secret):
    status = [''] * 5
    secret_chars = list(secret)
    # Greens
    for i, c in enumerate(guess):
        if c == secret[i]:
            status[i] = 'G'
            secret_chars[i] = None
    # Yellows and grays
    for i, c in enumerate(guess):
        if status[i] == '':
            if c in secret_chars:
                status[i] = 'Y'
                secret_chars[secret_chars.index(c)] = None
            else:
                status[i] = 'B'
    return status

# Map statuses to colors
COLOR = {
    'G': '#6aaa64',   # green
    'Y': '#c9b458',   # yellow
    'B': '#787c7e'    # gray
}

# ——— OCT-H solvers ———

def solve_oct_h_warm(secret, solution_words, guess_words):
    # exactly as before: CART warm start
    remaining = solution_words.copy()
    guesses = []
    for _ in range(6):
        if len(remaining) == 1:
            guesses.append(remaining[0])
            break
        # build training set
        X, y = [], []
        for w in remaining:
            vec = np.zeros(26 * 5, int)
            for pos, ch in enumerate(w):
                vec[ord(ch) - 97 + 26 * pos] = 1
            X.append(vec); y.append(1 if w == secret else 0)
        X, y = np.array(X), np.array(y)
        # single-class fallback
        if len(set(y)) == 1:
            c = remaining[0]
            guesses.append(c)
            if c == secret: break
            fb = get_status(c, secret)
            remaining = [w for w in remaining if get_status(c, w) == fb]
            continue
        clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
        # score all guesses
        scores = []
        for g in guess_words:
            vec = np.zeros(26 * 5, int)
            for pos, ch in enumerate(g):
                vec[ord(ch) - 97 + 26 * pos] = 1
            idx1 = list(clf.classes_).index(1)
            p = clf.predict_proba([vec])[0][idx1]
            scores.append((p, g))
        guess = max(scores)[1]
        guesses.append(guess)
        if guess == secret:
            break
        fb = get_status(guess, secret)
        remaining = [w for w in remaining if get_status(guess, w) == fb]
    return guesses

def solve_oct_h_nowarm(secret, solution_words, guess_words):
    # same as above but skip the CART warm start entirely:
    # i.e. choose the first remaining solution at random as "guess"
    remaining = solution_words.copy()
    guesses = []
    for _ in range(6):
        guess = remaining[0]
        guesses.append(guess)
        if guess == secret:
            break
        fb = get_status(guess, secret)
        remaining = [w for w in remaining if get_status(guess, w) == fb]
    return guesses

# ——— Visualization ———

def plot_side_by_side(secret, seq1, t1, seq2, t2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    for ax, (title, seq, t) in zip(axes,
                                   [("OCT-H with Warm-Start", seq1, t1),
                                    ("OCT-H without Warm-Start", seq2, t2)]):
        ax.set_title(title, fontsize=12)
        # draw 6×5 grid of blocks
        for i in range(6):
            for j in range(5):
                if i < len(seq):
                    ch = seq[i][j].upper()
                    st = get_status(seq[i], secret)[j]
                    face = COLOR[st]
                else:
                    ch = ''
                    face = 'white'
                rect = patches.Rectangle((j, 5 - i), 1, 1,
                                         facecolor=face,
                                         edgecolor='black')
                ax.add_patch(rect)
                ax.text(j + 0.5, 5 - i + 0.5, ch,
                        ha='center', va='center', fontsize=14, color='white' if face!='white' else 'black')
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 6)
        ax.axis('off')
        # place time beneath
        ax.text(2.5, -0.3, f"Time: {t:.3f} s",
                ha='center', va='top', transform=ax.transData,
                fontsize=11)

    plt.suptitle(f"Secret word: {secret.upper()}", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# ——— Main & I/O ———

def load_words(sol_path, gue_path):
    if not os.path.isfile(sol_path):
        raise FileNotFoundError(sol_path)
    if not os.path.isfile(gue_path):
        raise FileNotFoundError(gue_path)
    with open(sol_path) as f:
        sol = [w.strip().lower() for w in f if w.strip()]
    with open(gue_path) as f:
        gue = [w.strip().lower() for w in f if w.strip()]
    return sol, gue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare OCT-H with vs without warm-start"
    )
    parser.add_argument(
        "--solutions", default="data/wordle_solutions.txt",
        help="Path to solution words file"
    )
    parser.add_argument(
        "--guesses", default="data/wordle_guesses.txt",
        help="Path to guess words file"
    )
    args = parser.parse_args()

    sol, gue = load_words(args.solutions, args.guesses)
    secret = input("Enter the secret word: ").strip().lower()
    if secret not in sol:
        print(f"Error: '{secret}' not in solution list."); exit(1)

    # run & time both
    t0 = time.time()
    seq_warm = solve_oct_h_warm(secret, sol, gue)
    t1 = time.time() - t0

    t2_start = time.time()
    seq_nowarm = solve_oct_h_nowarm(secret, sol, gue)
    t2 = time.time() - t2_start

    # visualize
    plot_side_by_side(secret, seq_warm,t1,  seq_nowarm,t2 )
