# train_model.py

from micrograd.nn import MLP
from micrograd.engine import Value
from utils import build_vocab, vectorize

import numpy as np
import pickle

# Dummy training data
data = [
    # Good Matches
    ("python developer", "python django flask", 1.0),
    ("data analyst", "excel sql tableau", 1.0),
    ("backend engineer", "nodejs express mongodb", 1.0),
    ("frontend developer", "html css javascript react", 1.0),
    ("cybersecurity analyst", "network security ethical hacking kali linux", 1.0),
    ("machine learning engineer", "python numpy pandas sklearn", 1.0),
    ("devops engineer", "aws docker kubernetes jenkins", 1.0),

    # Bad Matches
    ("frontend developer", "illustrator photoshop", 0.0),
    ("java developer", "photoshop figma", 0.0),
    ("data analyst", "bootstrap javascript", 0.0),
    ("cybersecurity analyst", "react html", 0.0),
    ("backend engineer", "corel draw photoshop", 0.0),

    # Partial Matches
    ("data analyst", "excel python", 0.6),
    ("backend engineer", "nodejs sql", 0.7),
    ("python developer", "python numpy", 0.5),
    ("devops engineer", "docker bash shell", 0.6),
    ("cybersecurity analyst", "linux wireshark", 0.7),
    ("frontend developer", "html css", 0.5),
    ("machine learning engineer", "tensorflow keras", 0.8),
    ("java developer", "java springboot", 1.0),
    ("java developer", "c++ python", 0.3),
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
