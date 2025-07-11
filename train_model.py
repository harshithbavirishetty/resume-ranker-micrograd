# train_model.py

from micrograd.nn import MLP
from micrograd.engine import Value
from utils import build_vocab, vectorize

import numpy as np
import pickle

# Dummy training data
data = [
    ("python developer", "python django flask", 1.0),
    ("data analyst", "excel sql tableau", 1.0),
    ("frontend developer", "photoshop illustrator", 0.0),
    ("backend engineer", "nodejs express mongodb", 1.0),
    ("java dev", "photoshop sketch", 0.0),
]

# Build vocab
all_text = [jd for jd, _, _ in data] + [res for _, res, _ in data]
vocab = build_vocab(all_text)

# Prepare training data
X, Y = [], []
for jd, res, score in data:
    x_vec = np.concatenate([vectorize(jd, vocab), vectorize(res, vocab)])
    X.append([Value(v) for v in x_vec])
    Y.append(Value(score))

model = MLP(len(X[0]), [8, 1])  # input size, hidden layer, output layer

# Training loop
for epoch in range(50):
    total_loss = 0
    for x, y in zip(X, Y):
        y_pred = model(x)
        loss = (y_pred - y) * (y_pred - y)
        total_loss += loss.data

        for p in model.parameters():
            p.grad = 0
        loss.backward()

        for p in model.parameters():
            p.data -= 0.05 * p.grad  # learning rate

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.4f}")

# Save model + vocab
with open("trained_model.pkl", "wb") as f:
    pickle.dump((model, vocab), f)
