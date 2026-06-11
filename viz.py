import matplotlib.pyplot as plt
import numpy as np
import json
import pickle

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
    rolling_window = 10

    with open("ppo_new.pt.json", "r") as f:
        data = json.load(f)
    loss = data["loss"]
    actor_loss = data["actor_loss"]
    critic_loss = data["critic_loss"]

    loss_profiles = [
        ("loss", loss),
        ("actor_loss", actor_loss),
        ("critic_loss", critic_loss),
    ]

    averaged_loss_profiles = [
        (name, value, rolling_average(value, rolling_window)) for name, value in loss_profiles
    ]

    fig, axes = plt.subplots(
        len(loss_profiles),
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

        if len(averaged) > 1:
            derivative = np.gradient(averaged, x)
            x2, deravg = rolling_average(derivative, rolling_window)
            x2 = x2 + rolling_window

            derivative_ax.plot(
                x,
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
                second_derivative = np.gradient(deravg, x2)
                x3, second_deravg = rolling_average(second_derivative, rolling_window)
                x3 = x3 + rolling_window * 2

                second_derivative_ax.plot(
                    x2,
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