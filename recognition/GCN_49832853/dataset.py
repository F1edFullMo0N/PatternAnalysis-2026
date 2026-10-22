import numpy as np
import torch
from torch_geometric.data import Data


def load_facebook(path="data/facebook.npz"):
    with np.load(path, allow_pickle=False) as npz:
        x = torch.tensor(npz["X"], dtype=torch.float32)

        edge_index = torch.tensor(
            np.stack([npz["edges_x"], npz["edges_y"]]),
            dtype=torch.long
        )

        y = torch.tensor(npz["y"], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    data = load_facebook()
    print(data)
    print("Nodes:", data.num_nodes)
    print("Features:", data.num_node_features)
    print("Edges:", data.num_edges)