import numpy as np

states = np.arange(7)  # [0, 1, 2, 3, 4, 5, 6], 0 and 6 are terminal
V = np.zeros(7)
V[1:6] = 0.5  # Initialize middle states
alpha = 0.1
gamma = 1.0

def td_0():
    state = 3  # Start from middle
    while state not in [0, 6]:
        next_state = state + np.random.choice([-1, 1])
        reward = 1 if next_state == 6 else 0
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state

for episode in range(100):
    td_0()

print("Estimated values:", V)
