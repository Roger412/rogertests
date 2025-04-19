import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import intel_extension_for_pytorch as ipex

# Define simple model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    # Generate toy dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Move data to XPU
    device = "cpu"
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Optimize for Intel XPU
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Loss function
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                preds = model(X_test).to("cpu")
                y_true = y_test.to("cpu")
                acc = ((preds > 0.5) == y_true).float().mean()
                print(f"📉 Epoch {epoch}: Loss={loss.item():.4f} | Accuracy={acc.item()*100:.2f}%")

# ✅ Prevent accidental execution when imported or restarted
if __name__ == "__main__":
    main()
