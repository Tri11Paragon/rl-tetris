import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import json
import pickle
from pathlib import Path
import config


def rolling_average(values, window):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, False, 0)
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
    parser.add_argument("type",
                        nargs="?",
                        default=None,
                        help="Run command to see a list of options, use + to combine multiple options into a single graph")
    args = parser.parse_args()

    rolling_window = 50

    cfg = config.Config(Path(args.location) / args.file / "config.nix").load()

    with open(str(Path(args.location) / args.file / "state.json"), "r") as f:
        data = json.load(f)

    filtered_data = {name: value for name, value in data.items() if type(value) == list}

    print("Options:")
    for key in filtered_data.keys():
        print(f"\t{key}")
    print()

    expansions = {
        "loss": [data for data in filtered_data.keys() if "_loss" in data],
        "all": [data for data in filtered_data.keys()]
    }

    print("Expansions:")
    for key, values in expansions.items():
        print(f"\t{key}: [{', '.join(values)}]")
    print()

    if args.type is None:
        return

    parts = args.type.split("+")
    parts = [part.strip() for part in parts]
    parts = [
        inner
        for part in parts
        for inner in (expansions[part] if (part in expansions) else [part])
    ]
    parts = list(dict.fromkeys(parts))
    print("Using: ", parts)

    averaged_loss_profiles = []
    for part in parts:
        averaged_loss_profiles += [(name, value, rolling_average(value, rolling_window)) for name, value in filtered_data.items() if part in name]

    fig, axes = plt.subplots(
        len(averaged_loss_profiles),
        sharex="col",
        figsize=(18, 8),
    )

    if len(averaged_loss_profiles) == 1:
        axes = [axes]

    rolling_lines = []

    ax_slider = plt.axes((0.1, 0.01, 0.8, 0.03))
    a_slider = Slider(
        ax=ax_slider,
        label="Rolling Window",
        valmin=10,
        valmax=250,
        valinit=rolling_window,
        valstep=1,
    )

    for row, (label, values, avg) in enumerate(averaged_loss_profiles):
        ax = axes[row]

        ax.plot(values, label=label, alpha=0.45)

        x, averaged = avg
        rolling_line, = ax.plot(
            x,
            averaged,
            color="tab:gray",
            label=f"{label} rolling avg ({rolling_window})",
            linewidth=2,
        )
        rolling_lines.append((rolling_line, values, label, ax))

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")

    def update_rolling_window(value):
        window = int(value)

        for rolling_line, values, label, ax in rolling_lines:
            x, averaged = rolling_average(values, window)
            rolling_line.set_data(x, averaged)
            rolling_line.set_label(f"{label} rolling avg ({window})")
            ax.legend()
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    a_slider.on_changed(update_rolling_window)

    fig.suptitle(f"Recorded attributes for {args.file} with network '{cfg.network.mode}'")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()



if __name__ == "__main__":
    main()