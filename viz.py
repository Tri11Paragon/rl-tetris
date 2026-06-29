import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from pathlib import Path


def rolling_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return np.arange(len(values)), values
    if window > len(values):
        return np.array([]), np.array([])

    averaged = np.convolve(values, np.ones(window) / window, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, averaged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", '-l', type=str, default="experiments")
    parser.add_argument("file")
    args = parser.parse_args()

    rolling_window = 10

    with open(str(Path(args.location) / args.file / "state.json"), "r") as f:
        data = json.load(f)

    loss_profiles = []
    if "actor_loss" in data and len(data["actor_loss"]) > 0:
        loss_profiles.append(("actor_loss", data["actor_loss"]))
    if "critic_loss" in data and len(data["critic_loss"]) > 0:
        loss_profiles.append(("critic_loss", data["critic_loss"]))
    if "loss" in data and len(data["loss"]) > 0:
        loss_profiles.append(("loss", data["loss"]))
    if "returns" in data and len(data["returns"]) > 0:
        loss_profiles.append(("returns", data["returns"]))

    averaged_loss_profiles = [
        (name, value, rolling_average(value, rolling_window)) for name, value in loss_profiles
    ]

    fig, axes = plt.subplots(
        max(len(loss_profiles), 2),
        3,
        sharex="col",
        figsize=(18, 8),
    )

    for row, (label, values, avg) in enumerate(averaged_loss_profiles):
        ax = axes[row, 0]
        derivative_ax = axes[row, 1]
        second_derivative_ax = axes[row, 2]

        ax.plot(values, label=label, alpha=0.45)

        x, averaged = avg
        ax.plot(
            x,
            averaged,
            color="tab:gray",
            label=f"{label} rolling avg ({rolling_window})",
            linewidth=2,
        )

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        if len(averaged) > 1:
            derivative = np.gradient(values)
            x2, deravg = rolling_average(derivative, rolling_window)

            derivative_ax.plot(
                derivative,
                label=f"d/dstep rolling avg {label}",
                color="tab:gray",
                linewidth=2,
                alpha=0.45
            )

            derivative_ax.plot(
                x2,
                deravg,
                label=f"d/dstep {label} rolling avg",
                color="tab:red",
                linewidth=2,
            )

            derivative_ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

            if len(deravg) > 1:
                second_derivative = np.gradient(derivative)
                x3, second_deravg = rolling_average(second_derivative, rolling_window)

                second_derivative_ax.plot(
                    second_derivative,
                    label=f"d²/dstep² smoothed {label}",
                    color="tab:gray",
                    linewidth=2,
                    alpha=0.45,
                )

                second_derivative_ax.plot(
                    x3,
                    second_deravg,
                    label=f"d²/dstep² {label} rolling avg",
                    color="tab:blue",
                    linewidth=2,
                )

                second_derivative_ax.axhline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )

        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        derivative_ax.set_ylabel("slope")
        derivative_ax.legend()
        derivative_ax.grid(True, alpha=0.3)

        second_derivative_ax.set_ylabel("curvature")
        second_derivative_ax.legend()
        second_derivative_ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    axes[-1, 2].set_xlabel("step")
    fig.suptitle("Loss Profiles, Rolling Average Derivatives, and Second Derivatives")
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()