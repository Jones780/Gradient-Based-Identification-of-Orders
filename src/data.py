"""Training-data generation and surrogate training."""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .solver import solve_adams


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_multi_alpha_data(f, alphas, h: float = 0.05, T: float = 5.0, num_traj_per_alpha: int = 100, noise_level: float = 0.05, domain: tuple[float, float] = (-2.0, 2.0)):
    X_list, t_list, alpha_list, target_list = [], [], [], []
    t_grid = np.arange(0, T + h, h)
    m = len(t_grid)
    for alpha in alphas:
        for _ in range(num_traj_per_alpha):
            init = np.random.uniform(*domain, size=2)
            _, u_clean = solve_adams(f, alpha, h, [0, T], init)
            rms = np.sqrt(np.mean(u_clean**2, axis=1, keepdims=True))
            u_noisy = u_clean + noise_level * rms * np.random.randn(*u_clean.shape)
            dclean = np.array([f(t_grid[j], u_clean[:, j]) for j in range(m)]).T
            for j in range(m):
                X_list.append(u_noisy[:, j])
                t_list.append(t_grid[j])
                alpha_list.append(alpha)
                target_list.append(dclean[:, j])
    return (np.asarray(X_list, dtype=np.float32), np.asarray(t_list, dtype=np.float32).reshape(-1, 1), np.asarray(alpha_list, dtype=np.float32).reshape(-1, 1), np.asarray(target_list, dtype=np.float32))


def train_model(model, X, t, alpha, target, *, device, epochs=150, batch_size=2048, lr=1e-3):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(t, dtype=torch.float32), torch.tensor(alpha, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, tb, ab, yb in loader:
            xb, tb, ab, yb = xb.to(device), tb.to(device), ab.to(device), yb.to(device)
            pred = model(xb, tb, ab)
            loss = crit(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        sched.step()
        if (ep + 1) % 30 == 0:
            print(f"Epoch {ep+1:3d}/{epochs}, loss={epoch_loss/len(loader):.3e}")
    return model
