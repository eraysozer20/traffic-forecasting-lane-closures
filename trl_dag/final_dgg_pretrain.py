
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# === Final Dynamic Graph Generator ===
class DynamicGraphGenerator(nn.Module):
    def __init__(self, num_nodes, num_timesteps, embed_dim, num_layers=2):
        super().__init__()
        self.N = num_nodes
        self.T = num_timesteps
        self.d = embed_dim
        self.L = num_layers

        self.E_node = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E_time = nn.Parameter(torch.randn(num_timesteps, embed_dim))
        self.W = nn.ParameterList([nn.Parameter(torch.randn(embed_dim, embed_dim)) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim)) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, X):
        T, N, d = X.shape
        device = X.device

        Z_all = torch.zeros(T, N, d, device=device)
        A_all = torch.zeros(T, N, N, device=device)

        Et_all = self.E_time[:T]  # safe indexing of E_time

        for t in range(T):
            Et = Et_all[t]  # independent time embedding
            E_comb = self.ln(self.E_node + Et)
            A_t = torch.matmul(E_comb, E_comb.T)
            A_t = F.softmax(A_t, dim=1)
            A_all[t] = A_t.detach()

            Z = X[t]
            I = torch.eye(N, device=device)

            for l in range(self.L):
                Z = torch.matmul((I + A_t), Z)
                Z = torch.matmul(Z, self.W[l]) + self.b[l]

            Z_all[t] = Z

        return Z_all, A_all

# === Example Training ===
def train_dgg_from_file():
    with h5py.File("data/example.h5", "r") as f:
        data = f["df/block0_values"][:]

    data = torch.tensor(data, dtype=torch.float32)
    mean, std = data.mean(), data.std()
    data = (data - mean) / std

    T, N = data.shape
    d = 32
    input_proj = nn.Linear(1, d)
    X = input_proj(data.unsqueeze(-1))  # (T, N, d)

    dgg = DynamicGraphGenerator(num_nodes=N, num_timesteps=T, embed_dim=d)
    optimizer = torch.optim.Adam(list(dgg.parameters()) + list(input_proj.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 101):
        dgg.train()
        optimizer.zero_grad()
        Z, _ = dgg(X)
        loss = criterion(Z, X)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Reconstruction Loss: {loss.item():.6f}")

if __name__ == "__main__":
    train_dgg_from_file()
