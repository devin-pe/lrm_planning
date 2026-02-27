import networkx as nx
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np

def draw_wavy_hanoi_path():
    # 1. Setup the Path (Your provided sequence)
    custom_path = [
        "1111", "2111", "2311", "3311", "3321", "1321", "1221", "2221",
        "2223", "3223", "3123", "1123", "1133", "2133", "2333", "3333"
    ]
    
    n = len(custom_path[0])
    print(f"Generating Sierpinski graph for {n} disks...")
    G = nx.Graph()

    # 2. Generate all nodes (D1, D2, D3, D4 notation)
    nodes = ["".join(seq) for seq in itertools.product("123", repeat=n)]
    G.add_nodes_from(nodes)

    # 3. Define Valid Edges (Standard Hanoi Rules)
    for u, v in itertools.combinations(nodes, 2):
        diff_indices = [i for i in range(n) if u[i] != v[i]]
        if len(diff_indices) == 1:
            disk_idx = diff_indices[0] # The disk being moved
            
            # Validity Check: Are there any smaller disks on the source or dest?
            # Smaller disks are at indices LESS than disk_idx in this notation
            u_peg, v_peg = u[disk_idx], v[disk_idx]
            is_valid = True
            for smaller_idx in range(0, disk_idx):
                if u[smaller_idx] == u_peg or u[smaller_idx] == v_peg:
                    is_valid = False
                    break
            
            if is_valid:
                G.add_edge(u, v)

    # 4. Layout
    print("Calculating layout...")
    pos = nx.kamada_kawai_layout(G)

    # 5. Prepare the visual highlighting
    path_edges = list(zip(custom_path, custom_path[1:]))
    path_edge_set = set(frozenset(e) for e in path_edges)
    background_edges = [e for e in G.edges() if frozenset(e) not in path_edge_set]

    # 6. Plotting
    plt.figure(figsize=(15, 12), facecolor='white')
    
    # Draw non-path edges
    nx.draw_networkx_edges(G, pos, edgelist=background_edges, 
                           edge_color='lightgray', alpha=0.3, width=1)
    
    # Draw YOUR specific path in Green
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                           edge_color='#00FF00', width=5, alpha=1.0)

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue', alpha=0.7)
    
    # Highlight Start and End
    nx.draw_networkx_nodes(G, pos, nodelist=[custom_path[0]], node_size=500, node_color='gold')
    nx.draw_networkx_nodes(G, pos, nodelist=[custom_path[-1]], node_size=500, node_color='red')

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')

    plt.title(f"Custom Path Highlighted (Notation: D1 D2 D3 D4)", fontsize=15)
    plt.axis('off')
    output_dir = 'graphs'
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, 'optimal_sierpinski')
    plt.savefig(f'{out_base}.png', dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved {out_base}.png")


def draw_sierpinski_top_down(user_labels, optimal_states=None, suboptimal_states=None):
    if len(user_labels) != 81:
        raise ValueError(f"You must provide exactly 81 labels. You provided {len(user_labels)}.")

    # 1. Geometry Setup
    h = np.sqrt(3) / 2
    corners = {
        '1': np.array([0.5, h]),  # Top
        '2': np.array([0, 0]),    # Bottom Left
        '3': np.array([1, 0])     # Bottom Right
    }

    # 2. Generate all 81 coordinates
    all_coords = []
    state_addresses = list(itertools.product(['1', '2', '3'], repeat=4))
    
    for state in state_addresses:
        pos = np.array([0.0, 0.0])
        for i, peg in enumerate(state):
            weight = 0.5**(i + 1)
            pos += weight * corners[peg]
        all_coords.append(pos)

    # 3. SORTING LOGIC: Top-to-Bottom, then Left-to-Right
    # We sort by Y (descending: -p[1]) then X (ascending: p[0])
    sorted_coords = sorted(all_coords, key=lambda p: (-p[1], p[0]))

    # 4. Visualization
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    
    # Adjust this if labels are still too close
    label_offset = 0.012

    label_to_coord = {}
    for i, (x, y) in enumerate(sorted_coords):
        label = user_labels[i]
        label_to_coord[label] = (x, y)
        
        # Draw node
        ax.scatter(x, y, color='black', s=15, zorder=3)
        
        # Draw label
        ax.text(x, y + label_offset, str(label), 
                fontsize=7, 
                ha='center', 
                va='bottom', 
                color='black', 
                fontweight='bold',
                zorder=4)

    def draw_path_edges(path_states, color):
        if not path_states:
            return
        for u, v in zip(path_states, path_states[1:]):
            if u not in label_to_coord or v not in label_to_coord:
                continue
            x1, y1 = label_to_coord[u]
            x2, y2 = label_to_coord[v]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, alpha=0.95, zorder=2)

    draw_path_edges(optimal_states, 'green')
    draw_path_edges(suboptimal_states, 'red')

    ax.set_aspect('equal')
    ax.axis('off')
    output_dir = 'graphs'
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, 'optimal_sierpinski')
    plt.savefig(f'{out_base}.png', dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved {out_base}.png")


state_labels = [
"1111",
"3111", "2111",
"3211", "2311",
"2211", "1211", "1311", "3311",
"2231", "3321",
"1231", "3231", "2321", "1321",
"1331", "3131", "2121", "1221",
"3331", "2331", "2131", "1131", "1121", "3121", "3221", "2221",
"3332", "2223",
"2332", "1332", "1223", "3223",
"2132", "1232", "1323", "3123",
"1132", "3132", "3232", "2232", "3323", "2323", "2123", "1123",
"1122", "2212", "3313", "1133",
"3122", "2122", "1212", "3212", "2313", "1313", "3133", "2133",
"3222", "2322", "1312", "3112", "2113", "1213", "3233", "2333",
"2222", "1222", "1322", "3322", "3312", "2312", "2112", "1112", "1113", "3113", "3213", "2213", "2233", "1233", "1333", "3333"
]

kimi_failed_states = [
    '1111', '3111', '3211', '2211', '2231', '1231', '1331', '3331',
    '3332', '2332', '2132', '1132', '1122', '3122', '3222', '2222',
]

optimal_states = [
    '1111', '2111', '2311', '3311', '3321', '1321', '1221', '2221',
    '2223', '3223', '3123', '1123', '1133', '2133', '2333', '3333',
]

draw_sierpinski_top_down(
    state_labels,
    optimal_states=optimal_states,
    suboptimal_states=kimi_failed_states,
)


