import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(csv_path, output=None):
    # Load data
    df = pd.read_csv(csv_path)

    # Determine unique channels
    ch_list = sorted(set(df['ch1']).union(df['ch2']))

    # Initialize matrix
    mat = pd.DataFrame(np.nan, index=ch_list, columns=ch_list)

    # Fill values
    for _, row in df.iterrows():
        i, j, v = row['ch1'], row['ch2'], row['coh_mean']
        mat.at[i, j] = v
        mat.at[j, i] = v  # assume symmetry

    # Optionally fill diagonal
    if 'diag' in df.columns:
        for _, row in df[df['ch1'] == df['ch2']].iterrows():
            mat.at[row['ch1'], row['ch2']] = row['coh_mean']

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(mat, interpolation='nearest', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(ch_list)))
    ax.set_yticks(np.arange(len(ch_list)))
    ax.set_xticklabels(ch_list, rotation=90)
    ax.set_yticklabels(ch_list)

    # Colorbar
    fig.colorbar(cax, ax=ax, label='Coherence Mean')

    plt.tight_layout()

    # Save or show
    if output:
        plt.savefig(output, dpi=300)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot coherence heatmap from CSV.')
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Path to save the figure (e.g. heatmap.png)', default=None)
    args = parser.parse_args()
    main(args.csv_path, args.output)
