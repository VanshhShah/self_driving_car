import numpy as np

# 3x3 "picture": bright road (8) on the left, dark grass (1 or 2) on the right
picture = np.array([[8, 8, 1],
                    [8, 8, 1],
                    [8, 8, 2]])

x = picture[:, 2].sum() - picture[:, 0].sum()   # right column total - left column total = 4 - 24
label = -15          # what the human did: road bends left
w = -1.0             # starting weight (a deliberately bad guess)
LEARNING_RATE = 0.001

print(f"x = {x}, label = {label}")
print(f"{'round':>5} {'w':>8} {'guess':>8} {'miss':>8} {'loss':>9} {'slope':>9}")
for round_no in range(0, 6):
    guess = w * x                 # the model
    miss = guess - label          # how wrong
    loss = miss ** 2              # squared, so big misses hurt a lot
    slope = 2 * miss * x          # how steeply the loss rises as w rises
    print(f"{round_no:>5} {w:>8.3f} {guess:>8.2f} {miss:>8.2f} {loss:>9.2f} {slope:>9.1f}")
    w = w - LEARNING_RATE * slope # nudge w downhill

# Edge detection from the document: a filter (-1, +1) slid over a row
row = np.array([1, 1, 8, 8, 8])
edges = row[1:] - row[:-1]
print("\nEdge filter on", row.tolist(), "->", edges.tolist(), "(lights up where brightness jumps)")
