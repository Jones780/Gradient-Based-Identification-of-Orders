"""Train the three surrogate models used by the paper."""
from __future__ import annotations
import argparse
import os
import torch

from src.data import set_seed, generate_multi_alpha_data, train_model
from src.models import DeepONetAlpha
from src.systems import linear_system, nonlinear_system, brusselator_system

SYSTEMS = {"linear": linear_system, "polynomial": nonlinear_system, "brusselator": brusselator_system}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SYSTEMS, default="linear")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    f = SYSTEMS[args.system]
    alphas = [0.25, 0.50, 0.75, 0.95]
    X, t, a, y = generate_multi_alpha_data(f, alphas, num_traj_per_alpha=100, noise_level=0.05)
    model = DeepONetAlpha(state_dim=2)
    train_model(model, X, t, a, y, device=device, epochs=args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"deeponet_{args.system}.pt")
    torch.save(model.state_dict(), path)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
