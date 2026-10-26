
import time
import random
import numpy as np
import torch
from torch import nn

from dataset import load_facebook, split_nodes
from modules import GCN


def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # Forward pass using the entire graph
    logits = model(data.x, data.edge_index)

    # Only training nodes contribute to the loss
    loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask]
    )

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()

    logits = model(data.x, data.edge_index)
    predictions = logits.argmax(dim=1)

    correct = (
        predictions[mask] == data.y[mask]
    ).sum().item()

    total = mask.sum().item()
    return correct / total


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load and split data
    data = load_facebook()
    data = split_nodes(data, seed=seed)
    data = data.to(device)

    # Initialize model
    model = GCN(
        num_features=data.num_node_features,
        num_classes=int(data.y.max().item()) + 1,
        hidden_channels=64,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()

    num_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", num_params)
    print("Training nodes:", data.train_mask.sum().item())
    print("Validation nodes:", data.val_mask.sum().item())

    # Short smoke test
    epochs = 5

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, data, optimizer, criterion
        )

        train_acc = evaluate(
            model, data, data.train_mask
        )

        val_acc = evaluate(
            model, data, data.val_mask
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Training and evaluation time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
