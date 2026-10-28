
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from dataset import load_facebook, split_nodes
from modules import GCN, MLP


SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()

    logits = model(data.x, data.edge_index)
    predictions = logits[mask].argmax(dim=1)

    y_true = data.y[mask].cpu().numpy()
    y_pred = predictions.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1


def run_experiment(name, model, data, device):
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    waiting = 0

    num_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n===== {name} =====")
    print(f"Parameters: {num_params}")

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(
            model, data, optimizer, criterion
        )

        val_acc, val_f1 = evaluate(
            model, data, data.val_mask
        )

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            waiting = 0
        else:
            waiting += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val Macro-F1: {val_f1:.4f}"
            )

        if waiting >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore the best validation checkpoint
    model.load_state_dict(best_state)

    final_acc, final_f1 = evaluate(
        model, data, data.val_mask
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Validation Accuracy: {final_acc:.4f}")
    print(f"Validation Macro-F1: {final_f1:.4f}")

    # Save locally for later prediction and testing
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        checkpoint_dir / f"{name.lower()}_best.pt"
    )

    return {
        "model": model,
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "best_epoch": best_epoch
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    data = split_nodes(
        load_facebook(),
        seed=SEED
    ).to(device)

    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    print("Training:", data.train_mask.sum().item())
    print("Validation:", data.val_mask.sum().item())
    print("Testing:", data.test_mask.sum().item())

    # Train MLP baseline
    set_seed(SEED)
    mlp_result = run_experiment(
        "MLP",
        MLP(num_features, num_classes),
        data,
        device
    )

    # Train GCN
    set_seed(SEED)
    gcn_result = run_experiment(
        "GCN",
        GCN(num_features, num_classes),
        data,
        device
    )

    print("\n===== Validation Comparison =====")
    print(
        f"{'Model':<8} {'Accuracy':<12} {'Macro-F1':<12}"
    )

    for name, result in [
        ("MLP", mlp_result),
        ("GCN", gcn_result)
    ]:
        print(
            f"{name:<8} "
            f"{result['accuracy']:<12.4f} "
            f"{result['macro_f1']:<12.4f}"
        )


if __name__ == "__main__":
    main()
