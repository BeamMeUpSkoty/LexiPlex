# plotting/circumplex.py
import matplotlib.pyplot as plt

def plot_circumplex(
    df,
    group_col,
    label_map=None,
    point_size=50,
    edge_color='k'
):
    """
    Create a Russell circumplex plot of valence vs. arousal.

    Parameters:
    - df: pandas.DataFrame with 'valence' and 'arousal' columns
      and a grouping column specified by group_col.
    - group_col: name of the column to group by (e.g., 'speaker' or 'topic').
    - label_map: optional dict mapping group values to display labels.
    - point_size: size for scatter points.
    - edge_color: color for point edges.

    Returns:
    - fig: a matplotlib Figure object containing the plot.
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=1)
    ax.add_artist(circle)

    # Draw zero lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)

    # Set axis limits and square aspect
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', adjustable='box')

    # Scatter points for each group
    for name, group in df.groupby(group_col):
        label = label_map.get(name, str(name)) if label_map else str(name)
        ax.scatter(
            group['valence'],
            group['arousal'],
            s=point_size,
            edgecolors=edge_color,
            label=label
        )

    # Label axes and title
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title(f'Circumplex Plot by {group_col.title()}')

    # Place legend outside plot
    ax.legend(title=group_col.title(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig
