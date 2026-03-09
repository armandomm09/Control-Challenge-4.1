import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot desired vs real positions from a CSV file.")
    parser.add_argument("csvfile", help="Path to the CSV file produced by the controller")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (subplots and 3D in separate windows)")
    parser.add_argument("--outdir", help="Directory to save figures (default: current dir)", default=".")
    args = parser.parse_args()

    df = pd.read_csv(args.csvfile)

    # Prepare lists of plot specifications for reuse
    plots = []  # each entry: (title, ylabel, [(real_col, label1), (des_col, label2)])

    # Cartesian coordinates
    cart_axes = [("x", "p_x", "p_des_x"),
                 ("y", "p_y", "p_des_y"),
                 ("z", "p_z", "p_des_z")]
    for axis, real_col, des_col in cart_axes:
        plots.append((f"Cartesian {axis}-coordinate", axis,
                      [(real_col, f"real {axis}"), (des_col, f"desired {axis}")]))

    # Joint angles
    for i in range(1, 7):
        q_col = f"q_{i}"
        qd_col = f"q_des_{i}"
        if q_col in df.columns and qd_col in df.columns:
            plots.append((f"Joint {i} angle", f"joint {i} (rad)",
                          [(q_col, f"q_{i}"), (qd_col, f"q_des_{i}")]))

    # Create individual figures (for saving)
    individual_figs = []
    for title, ylabel, lines in plots:
        fig = plt.figure()
        for col, lbl in lines:
            plt.plot(df['time'], df[col], label=lbl)
        plt.xlabel('time')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        individual_figs.append((fig, title))

    # Create subplots figure
    n = len(plots)
    cols = 2
    rows = (n + cols - 1) // cols
    subplots_fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()
    for ax, (title, ylabel, lines) in zip(axes, plots):
        for col, lbl in lines:
            ax.plot(df['time'], df[col], label=lbl)
        ax.set_xlabel('time')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    # hide any unused axes
    for ax in axes[len(plots):]:
        ax.set_visible(False)
    subplots_fig.tight_layout()

    # Create 3D figure
    threed_fig = plt.figure()
    ax = threed_fig.add_subplot(111, projection='3d')
    ax.plot(df['p_x'], df['p_y'], df['p_z'], label='real position')
    ax.plot(df['p_des_x'], df['p_des_y'], df['p_des_z'], label='desired position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory: Real vs Desired Positions')
    ax.legend()
    ax.grid(True)

    if args.show:
        # Show subplots and 3D in separate windows
        plt.figure(subplots_fig.number)
        plt.show()
        plt.figure(threed_fig.number)
        plt.show()
    else:
        # Save all: individuals, subplots combined, and 3D
        for fig, title in individual_figs:
            fname = title.lower().replace(' ', '_').replace('-', '_')
            outfile = f"{args.outdir}/{fname}.png"
            fig.savefig(outfile)
            print(f"wrote {outfile}")
        # save subplots
        outc = f"{args.outdir}/all_subplots.png"
        subplots_fig.savefig(outc)
        print(f"wrote {outc}")
        # save 3D
        outfile = f"{args.outdir}/3d_trajectory.png"
        threed_fig.savefig(outfile)
        print(f"wrote {outfile}")


if __name__ == '__main__':
    main()
