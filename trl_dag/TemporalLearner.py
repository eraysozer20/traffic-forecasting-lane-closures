
import torch
import torch.nn as nn
import torch.optim as optim
import h5py


# === Temporal Representation Learner with auxiliary decoder ===
class TRLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=2, num_layers=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.reconstruct_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_unmasked):
        B, N, S, F = x_unmasked.shape
        x = self.fc(x_unmasked)  # (B, N, S, H)
        x_flat = x.view(B * N, S, -1).permute(1, 0, 2)  # (S, B*N, H)
        encoded = self.encoder(x_flat)  # (S, B*N, H)
        encoded = encoded.permute(1, 0, 2).view(B, N, S, -1)  # (B, N, S, H)
        x_hat_unmasked = self.reconstruct_fc(encoded)  # (B, N, S, F)
        return encoded, x_hat_unmasked


# === Task Head (Transformer decoder) ===
class TaskHead(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers=1, num_heads=2):
        super().__init__()
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, S_all):
        B, N, S, H = S_all.shape
        x = S_all.view(B * N, S, H).permute(1, 0, 2)  # (S, B*N, H)
        x = self.decoder(x)
        x = x.permute(1, 0, 2).view(B, N, S, H)
        return self.fc(x)  # (B, N, S, F)


# === Full TRL Model ===
class TRLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, subseq_len, N_subseq):
        super().__init__()
        self.subseq_len = subseq_len
        self.N_subseq = N_subseq
        self.trl = TRLearner(input_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, subseq_len, hidden_dim))
        self.task_head = TaskHead(hidden_dim, input_dim)

    def forward(self, X, mask):
        B, T, F = X.shape
        S = self.subseq_len
        N = self.N_subseq
        X = X.view(B, N, S, F)

        unmask_idx = mask.sum(dim=(2, 3))[:, :, None, None] > 0
        mask_idx = ~unmask_idx

        X_unmasked = X[unmask_idx.expand_as(X)].view(B, -1, S, F)

        Sunmasked, X_hat_unmasked_aux = self.trl(X_unmasked)

        N_masked = mask_idx.sum(dim=1).max().item()
        Smasked = self.mask_token.expand(B, N_masked, S, -1)

        S_all = torch.zeros(B, N, S, Sunmasked.shape[-1], device=X.device)
        unmask_indices = unmask_idx.squeeze(-1).squeeze(-1)
        mask_indices = mask_idx.squeeze(-1).squeeze(-1)

        for b in range(B):
            S_all[b][unmask_indices[b]] = Sunmasked[b]
            S_all[b][mask_indices[b]] = Smasked[b, :mask_indices[b].sum()]

        X_hat = self.task_head(S_all)
        return X_hat, X, X_hat_unmasked_aux, X_unmasked


# === Mask Generator ===
def generate_mask(B, N, S, F, keep_ratio=0.25):
    mask = (torch.rand(B, N) < keep_ratio).float()
    return mask.view(B, N, 1, 1).expand(-1, -1, S, F)


# === Full training and evaluation ===
if __name__ == "__main__":
    with h5py.File("data/example.h5", "r") as f:
        data = f["df/block0_values"][:]

    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    mean, std = data.mean(), data.std()
    data = (data - mean) / std

    B, T, F = data.shape
    subseq_len = 4
    N_subseq = T // subseq_len
    hidden_dim = 32

    model = TRLModel(input_dim=F, hidden_dim=hidden_dim, subseq_len=subseq_len, N_subseq=N_subseq)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    lambda_aux = 0.1
    best_loss = float("inf")

    for epoch in range(1, 201):
        model.train()
        mask = generate_mask(B, N_subseq, subseq_len, F)
        X_hat, X_true, X_hat_aux, X_unmasked = model(data, mask)

        main_loss = criterion(X_hat[mask == 0], X_true.view(B, N_subseq, subseq_len, F)[mask == 0])
        aux_loss = criterion(X_hat_aux, X_unmasked)
        total_loss = main_loss + lambda_aux * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(), "best_trl_model.pth")

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f} | Main: {main_loss.item():.4f} | Aux: {aux_loss.item():.4f}")

    # Evaluation
    model.load_state_dict(torch.load("best_trl_model.pth"))
    model.eval()

    with torch.no_grad():
        mask = generate_mask(B, N_subseq, subseq_len, F)
        X_hat, X_true, _, _ = model(data, mask)
        y_pred = X_hat[mask == 0]
        y_true = X_true.view(B, N_subseq, subseq_len, F)[mask == 0]

        mse = nn.functional.mse_loss(y_pred, y_true).item()
        mae = nn.functional.l1_loss(y_pred, y_true).item()
        rmse = mse ** 0.5

    print("\n[Test Metrics on masked regions]")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
