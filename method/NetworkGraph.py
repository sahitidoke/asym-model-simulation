import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
  # Assuming application.py is in the same directory and contains the necessary functions

def plot_precision_graph(
    precision_matrix,
    labels=None,
    threshold=1e-6,
    weight_power=6.0,       # steep exponent — push this higher for more extreme contrast
    k=None,
    title="Precision Matrix Graph",
    seed=42,
    figsize=(10, 10),
    filename = None
):
    p = precision_matrix.shape[0]
    if labels is None:
        labels = [str(i) for i in range(p)]

    # Collect all nonzero magnitudes first, so we can normalize
    raw_weights = []
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            w = abs(precision_matrix[i, j])
            if w > threshold:
                edges.append((i, j))
                raw_weights.append(w)

    raw_weights = np.array(raw_weights)

    # --- Normalize to [0, 1] based on min/max observed magnitude ---
    w_min, w_max = raw_weights.min(), raw_weights.max()
    norm_weights = (raw_weights - w_min) / (w_max - w_min + 1e-12)

    # --- Steep nonlinear transform: large -> near 1 (strong pull),
    #     small -> near 0 (almost no pull, drifts far away) ---
    scaled_weights = norm_weights ** weight_power

    # Floor so weakest edges still have *some* spring, avoiding
    # total disconnection/blowup, but keep it tiny
    scaled_weights = np.clip(scaled_weights, 1e-4, None)

    G = nx.Graph()
    G.add_nodes_from(range(p))
    for (i, j), w in zip(edges, scaled_weights):
        G.add_edge(i, j, weight=w)

    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    community_id = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            community_id[node] = cid

    n_comms = len(communities)
    cmap = colormaps["tab20"].resampled(n_comms)
    node_colors = [cmap(community_id.get(n, 0)) for n in G.nodes()]

    if k is None:
        k = 0.3 / np.sqrt(p)   # tight core, long stretch for weak nodes

    pos = nx.spring_layout(
        G, seed=seed, k=k, weight="weight", iterations=500,  # more iterations to fully settle extremes
    )

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.4, edge_color="gray")
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=40,
        edgecolors="black", linewidths=0.4,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        print(f"Graph saved to {filename}")

    plt.show()
    return G, community_id