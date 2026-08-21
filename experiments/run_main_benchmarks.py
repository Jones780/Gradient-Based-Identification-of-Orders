"""Reproduce the main order-estimation benchmark (Table 1 style)."""
from __future__ import annotations
import argparse
import numpy as np
import torch

from src.data import set_seed
from src.models import DeepONetAlpha
from src.order_estimation import estimate_alpha_known_F
from src.systems import linear_system, nonlinear_system, brusselator_system

SYSTEMS = {"linear": linear_system, "polynomial": nonlinear_system, "brusselator": brusselator_system}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=SYSTEMS, default="linear")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONetAlpha(state_dim=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    f = SYSTEMS[args.system]

    errors = []
    for _ in range(args.repeats):
        ic = np.random.uniform(-2.0, 2.0, size=2)
        est, _ = estimate_alpha_known_F(f, 0.85, 0.05, ic, model, device)
        errors.append(abs(est - 0.85))
    print(f"{args.system}: MAE={np.mean(errors):.4f} +- {np.std(errors):.4f}")


if __name__ == "__main__":
    main()
