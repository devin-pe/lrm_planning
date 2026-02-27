"""
Visualize the full 3-disk Towers of Hanoi state space as a graph.
The state space forms a Sierpinski triangle structure.

Edges in the model's solution are colored red.
Edges in the optimal (BFS) solution are colored green.
"""

import itertools
import os
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque


def all_states(num_disks):
    """Generate all valid TOH states for num_disks disks on 3 pegs."""
    # Each disk can be on peg 0, 1, or 2
    # We represent a state as a tuple of peg assignments for each disk (1..n)
    states = []
    for assignment in itertools.product(range(3), repeat=num_disks):
        # assignment[i] = peg for disk (i+1)
        # Build peg lists
        pegs = [[], [], []]
        # Place disks from largest to smallest
        for disk in range(num_disks, 0, -1):
            peg = assignment[disk - 1]
            pegs[peg].append(disk)
        states.append(tuple(tuple(p) for p in pegs))
    return list(set(states))


def get_neighbors(state):
    """Get all valid neighbor states (one move away)."""
    neighbors = []
    pegs = [list(p) for p in state]
    for from_peg in range(3):
        if not pegs[from_peg]:
            continue
        disk = pegs[from_peg][-1]
        for to_peg in range(3):
            if from_peg == to_peg:
                continue
            if pegs[to_peg] and pegs[to_peg][-1] < disk:
                continue
            new_pegs = [list(p) for p in pegs]
            new_pegs[from_peg] = new_pegs[from_peg][:-1]
            new_pegs[to_peg] = new_pegs[to_peg] + [disk]
            neighbors.append(tuple(tuple(p) for p in new_pegs))
    return neighbors


def moves_to_edges(initial_state, moves):
    """Convert a move sequence to a list of (state_from, state_to) edges."""
    edges = []
    state = [list(p) for p in initial_state]
    for disk, from_peg, to_peg in moves:
        s_before = tuple(tuple(p) for p in state)
        state[from_peg].remove(disk)
        state[to_peg].append(disk)
        # Re-sort isn't needed since we always move top disk
        s_after = tuple(tuple(p) for p in state)
        edges.append((s_before, s_after))
    return edges


def _extract_moves_block(text):
    for match in re.finditer(r'moves\s*=\s*\[', text, flags=re.IGNORECASE):
        start = text.find('[', match.start())
        if start == -1:
            continue

        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    block = text[start:idx + 1]
                    if block.strip() != '[]':
                        return block
                    break
    return None


def _parse_moves_json(moves_str):
    cleaned = re.sub(r'#[^\n]*', '', moves_str)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    return json.loads(cleaned)


def ensure_zero_based_moves(moves):
    for move in moves:
        if len(move) != 3:
            raise ValueError(f"Invalid move format: {move}")
        disk, from_peg, to_peg = move
        if not isinstance(disk, int) or not isinstance(from_peg, int) or not isinstance(to_peg, int):
            raise ValueError(f"Move must contain ints: {move}")
        if from_peg not in {0, 1, 2} or to_peg not in {0, 1, 2}:
            raise ValueError(
                f"Expected 0-based peg indexing (0..2), found move: {move}"
            )
    return moves


def state_from_label(label, num_disks, largest_to_smallest=True):
    """
    Convert a label like '1113' to state tuple-of-tuples.

    largest_to_smallest=True means label[0] is disk n, label[-1] is disk 1.
    """
    if len(label) != num_disks:
        raise ValueError(f"Label '{label}' must have length {num_disks}")

    pegs = [[], [], []]
    for idx, ch in enumerate(label):
        if ch not in {'1', '2', '3'}:
            raise ValueError(f"Invalid peg digit '{ch}' in label '{label}'")
        peg = int(ch) - 1
        disk = (num_disks - idx) if largest_to_smallest else (idx + 1)
        pegs[peg].append(disk)

    return tuple(tuple(p) for p in pegs)


def labels_to_edges(labels, num_disks, largest_to_smallest=True):
    states = [state_from_label(lbl, num_disks, largest_to_smallest=largest_to_smallest) for lbl in labels]
    edges = []
    for i in range(len(states) - 1):
        edges.append((states[i], states[i + 1]))
    return edges


def generate_optimal_moves(num_disks, source=0, target=2, auxiliary=1):
    moves = []

    def _hanoi(n, src, dst, aux):
        if n == 0:
            return
        _hanoi(n - 1, src, aux, dst)
        moves.append([n, src, dst])
        _hanoi(n - 1, aux, dst, src)

    _hanoi(num_disks, source, target, auxiliary)
    return ensure_zero_based_moves(moves)


def load_kimi_failed_moves_4(
    file_path='new_baseline_results/kimi-k2-tower/problem_009.json',
):
    with open(file_path) as f:
        record = json.load(f)

    response = record.get('response', {})
    text = response.get('raw_content') or response.get('answer') or ''
    moves_str = _extract_moves_block(text)
    if moves_str is None:
        raise ValueError(f"No non-empty moves block found in {file_path}")

    moves = _parse_moves_json(moves_str)
    return ensure_zero_based_moves(moves)


def sierpinski_layout(G, num_disks=3):
    """
    Compute Sierpinski-triangle positions for TOH states.

    111 (all on peg 1) at top, 222 (all on peg 2) at bottom-left,
    333 (all on peg 3) at bottom-right.

    Uses the correct recursive peg-to-corner mapping: when zooming into
    a sub-triangle, the two pegs NOT anchoring that corner swap their
    corner assignments.  This produces the proper Sierpinski gasket where
    adjacent sub-triangles touch at the right vertices.
    """
    # Main triangle corners (T = top, BL = bottom-left, BR = bottom-right)
    top = np.array([0.5, np.sqrt(3) / 2])
    bl  = np.array([0.0, 0.0])
    br  = np.array([1.0, 0.0])

    def state_to_assignment(state):
        """Convert state (pegs representation) to disk -> peg assignment."""
        assignment = {}
        for peg_idx, peg in enumerate(state):
            for disk in peg:
                assignment[disk] = peg_idx
        return assignment

    def compute_pos(state):
        assignment = state_to_assignment(state)

        # Physical corner positions of the current (sub-)triangle
        cT  = top.copy()
        cBL = bl.copy()
        cBR = br.copy()

        # Which internal peg index currently sits at each corner
        peg_at_T  = 0   # peg 0 ("1") -> top
        peg_at_BL = 1   # peg 1 ("2") -> bottom-left
        peg_at_BR = 2   # peg 2 ("3") -> bottom-right

        for disk in range(num_disks, 0, -1):
            peg = assignment[disk]

            if peg == peg_at_T:
                # Zoom into the T-corner sub-triangle
                cBL = (cT + cBL) / 2
                cBR = (cT + cBR) / 2
                # cT stays
                peg_at_BL, peg_at_BR = peg_at_BR, peg_at_BL

            elif peg == peg_at_BL:
                # Zoom into the BL-corner sub-triangle
                cT  = (cBL + cT)  / 2
                cBR = (cBL + cBR) / 2
                # cBL stays
                peg_at_T, peg_at_BR = peg_at_BR, peg_at_T

            elif peg == peg_at_BR:
                # Zoom into the BR-corner sub-triangle
                cT  = (cBR + cT)  / 2
                cBL = (cBR + cBL) / 2
                # cBR stays
                peg_at_T, peg_at_BL = peg_at_BL, peg_at_T

        # Position is the centroid of the final tiny triangle
        return (cT + cBL + cBR) / 3

    positions = {}
    for node in G.nodes():
        positions[node] = compute_pos(node)

    return positions


def format_state(state, largest_to_smallest=False):
    """Format state as compact label.

    largest_to_smallest=False: disk1..diskN order
    largest_to_smallest=True:  diskN..disk1 order
    """
    assignment = {}
    for peg_idx, peg in enumerate(state):
        for disk in peg:
            assignment[disk] = peg_idx
    num_disks = len(assignment)
    if largest_to_smallest:
        disk_range = range(num_disks, 0, -1)
    else:
        disk_range = range(1, num_disks + 1)
    return ''.join(str(assignment[d] + 1) for d in disk_range)


def sierpinski_layout_by_label(G, largest_to_smallest=False):
    """
    Label-driven Sierpinski layout.

    The first label digit selects the major triangle (1=top, 2=bottom-left,
    3=bottom-right), the next digit selects sub-triangle, and so on.
    This matches row patterns like:
      row1: 1111
      row2: 1112, 1113
      ...
      row9+: 2*** on left and 3*** on right.
    """
    top = np.array([0.5, np.sqrt(3) / 2])
    bl = np.array([0.0, 0.0])
    br = np.array([1.0, 0.0])

    positions = {}
    for node in G.nodes():
        label = format_state(node, largest_to_smallest=largest_to_smallest)

        cT = top.copy()
        cBL = bl.copy()
        cBR = br.copy()

        for ch in label:
            if ch == '1':
                cBL = (cT + cBL) / 2
                cBR = (cT + cBR) / 2
            elif ch == '2':
                cT = (cBL + cT) / 2
                cBR = (cBL + cBR) / 2
            elif ch == '3':
                cT = (cBR + cT) / 2
                cBL = (cBR + cBL) / 2

        positions[node] = (cT + cBL + cBR) / 3

    return positions


def build_state_graph(num_disks):
    states = all_states(num_disks)
    G = nx.Graph()
    G.add_nodes_from(states)

    for s in states:
        for neighbor in get_neighbors(s):
            if not G.has_edge(s, neighbor):
                G.add_edge(s, neighbor)

    return G


def snap_to_triangle_grid(pos, num_disks):
    """
    Snap continuous Sierpinski coordinates to a discrete triangle grid.

    For 4 disks, this yields 16 visible rows with gap columns so labels appear
    in row-by-row Sierpinski form (with blank spaces between sub-triangles).

    After snapping, each row's x-coordinates are linearly rescaled so that the
    outermost nodes sit exactly on the straight edges of the outer triangle,
    ensuring fully straight sides from top to bottom.
    """
    rows = 2 ** num_disks
    cols = 2 * rows - 1

    ys = np.array([p[1] for p in pos.values()])
    ymin = float(ys.min())
    ymax = float(ys.max())
    span = max(1e-12, ymax - ymin)

    # First pass: snap every node to its nearest row and column
    snapped_raw = {}   # node -> (row, x_snapped, y_snapped)
    row_nodes = {}     # row -> list of (node, x_snapped)
    for node, p in pos.items():
        x = float(p[0])
        y = float(p[1])

        row = int(round((ymax - y) / span * (rows - 1)))
        row = max(0, min(rows - 1, row))

        col = int(round(x * (cols - 1)))
        col = max(0, min(cols - 1, col))

        x_new = col / (cols - 1)
        y_new = 1.0 - (row / (rows - 1))
        snapped_raw[node] = (row, x_new, y_new)
        row_nodes.setdefault(row, []).append((node, x_new))

    # Second pass: rescale x per row so outermost nodes lie on the triangle edges
    snapped = {}
    for node, (row, x_snap, y_new) in snapped_raw.items():
        t = row / (rows - 1)            # 0 at top, 1 at bottom
        left_x  = 0.5 - 0.5 * t         # left edge of outer triangle
        right_x = 0.5 + 0.5 * t         # right edge of outer triangle

        xs_in_row = [xx for _, xx in row_nodes[row]]
        min_x = min(xs_in_row)
        max_x = max(xs_in_row)

        if max_x - min_x > 1e-12:
            x_final = left_x + (x_snap - min_x) / (max_x - min_x) * (right_x - left_x)
        else:
            x_final = (left_x + right_x) / 2   # single-node row → centre

        snapped[node] = np.array([x_final, y_new])

    return snapped


def draw_graph(
    G,
    num_disks,
    initial_state,
    goal_state,
    model_moves,
    optimal_moves,
    title,
    out_prefix,
    model_color='red',
    model_label='Model',
    optimal_color='#00aa00',
    optimal_label='Optimal',
    fig_size=(16, 14),
    label_font_size=13,
    node_size_regular=350,
    snap_grid=False,
    use_label_layout=False,
    show_paths=True,
    model_state_labels=None,
    optimal_state_labels=None,
    labels_largest_to_smallest=False,
):
    print(f"State space ({num_disks} disks): {G.number_of_nodes()} states, {G.number_of_edges()} edges")

    if model_state_labels is not None:
        model_edges = labels_to_edges(
            model_state_labels,
            num_disks,
            largest_to_smallest=labels_largest_to_smallest,
        )
    else:
        model_edges = moves_to_edges(initial_state, model_moves)

    if optimal_state_labels is not None:
        optimal_edges = labels_to_edges(
            optimal_state_labels,
            num_disks,
            largest_to_smallest=labels_largest_to_smallest,
        )
    else:
        optimal_edges = moves_to_edges(initial_state, optimal_moves)

    for a, b in model_edges + optimal_edges:
        if not G.has_edge(a, b):
            raise ValueError(f"Provided state sequence contains non-adjacent states: {a} -> {b}")

    model_edge_set = set()
    for a, b in model_edges:
        model_edge_set.add((a, b) if a < b else (b, a))

    optimal_edge_set = set()
    for a, b in optimal_edges:
        optimal_edge_set.add((a, b) if a < b else (b, a))

    if use_label_layout:
        pos = sierpinski_layout_by_label(
            G,
            largest_to_smallest=labels_largest_to_smallest,
        )
    else:
        pos = sierpinski_layout(G, num_disks)
    if snap_grid:
        pos = snap_to_triangle_grid(pos, num_disks)

    both_edges = []
    model_only_edges = []
    optimal_only_edges = []
    normal_edges = []

    for u, v in G.edges():
        key = (u, v) if u < v else (v, u)
        in_model = key in model_edge_set
        in_optimal = key in optimal_edge_set

        if in_model and in_optimal:
            both_edges.append((u, v))
        elif in_model:
            model_only_edges.append((u, v))
        elif in_optimal:
            optimal_only_edges.append((u, v))
        else:
            normal_edges.append((u, v))

    model_states = set()
    for a, b in model_edges:
        model_states.add(a)
        model_states.add(b)

    optimal_states = set()
    for a, b in optimal_edges:
        optimal_states.add(a)
        optimal_states.add(b)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        edge_color='#d8d8d8',
        width=0.6,
        alpha=0.35,
        ax=ax,
    )

    if show_paths:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=model_only_edges,
            edge_color=model_color,
            width=2.6,
            alpha=0.9,
            ax=ax,
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=optimal_only_edges,
            edge_color=optimal_color,
            width=3.0,
            alpha=0.9,
            ax=ax,
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=both_edges,
            edge_color='#8bc34a',
            width=3.2,
            alpha=0.95,
            ax=ax,
        )

    regular_nodes = [n for n in G.nodes() if n not in model_states and n not in optimal_states]
    both_nodes = [
        n for n in G.nodes()
        if n in model_states and n in optimal_states and n != initial_state and n != goal_state
    ]
    model_only_nodes = [
        n for n in model_states if n not in optimal_states and n != initial_state and n != goal_state
    ]
    optimal_only_nodes = [
        n for n in optimal_states if n not in model_states and n != initial_state and n != goal_state
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=regular_nodes,
        node_color='#ececec',
        node_size=node_size_regular,
        edgecolors='#999999',
        linewidths=0.4,
        ax=ax,
    )

    if show_paths:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=model_only_nodes,
            node_color='#dddddd',
            node_size=int(node_size_regular * 1.35),
            edgecolors=model_color,
            linewidths=1.8,
            ax=ax,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=optimal_only_nodes,
            node_color='#ccffcc',
            node_size=int(node_size_regular * 1.35),
            edgecolors=optimal_color,
            linewidths=1.8,
            ax=ax,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=both_nodes,
            node_color='#e6f5d0',
            node_size=int(node_size_regular * 1.35),
            edgecolors='#7cb342',
            linewidths=1.8,
            ax=ax,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[initial_state],
        node_color='#4444ff',
        node_size=int(node_size_regular * 2.0),
        edgecolors='black',
        linewidths=2.2,
        ax=ax,
        node_shape='s',
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[goal_state],
        node_color='gold',
        node_size=int(node_size_regular * 2.2),
        edgecolors='black',
        linewidths=2.2,
        ax=ax,
        node_shape='*',
    )

    labels = {
        node: format_state(node, largest_to_smallest=labels_largest_to_smallest)
        for node in G.nodes()
    }
    label_pos = {node: p + np.array([0.0, -0.018]) for node, p in pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=labels,
        font_size=label_font_size,
        font_color='#222222',
        font_family='monospace',
        font_weight='bold',
        ax=ax,
    )

    offset = np.array([0.0, 0.028])
    ax.annotate(
        "START",
        pos[initial_state] + offset,
        fontsize=9,
        ha='center',
        va='bottom',
        fontweight='bold',
        color='#4444ff',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85),
    )
    ax.annotate(
        "GOAL",
        pos[goal_state] + offset,
        fontsize=9,
        ha='center',
        va='bottom',
        fontweight='bold',
        color='#886600',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85),
    )

    if show_paths:
        legend_elements = [
            mpatches.Patch(facecolor=model_color, edgecolor=model_color, label=model_label),
            mpatches.Patch(facecolor=optimal_color, edgecolor=optimal_color, label=optimal_label),
            mpatches.Patch(facecolor='#8bc34a', edgecolor='#7cb342', label='Shared'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    ax.set_title(title, fontsize=15, pad=16)
    ax.axis('off')

    output_dir = 'graphs'
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, out_prefix)

    plt.tight_layout()
    plt.savefig(f'{out_base}.png', dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved {out_base}.png")
    plt.close(fig)


def main():
    # =========================
    # 3-disk figure (existing)
    # =========================
    num_disks = 3

    initial_state = ((1,), (3,), (2,))
    goal_state = ((1,), (), (3, 2))

    model_moves = [
        [2, 2, 1], [1, 0, 2], [2, 1, 0], [1, 2, 1],
        [1, 1, 0], [3, 1, 2], [1, 0, 1], [2, 0, 2], [1, 1, 0],
    ]

    optimal_moves = [
        [1, 0, 1], [2, 2, 0], [1, 1, 0], [3, 1, 2],
        [1, 0, 1], [2, 0, 2], [1, 1, 0],
    ]

    G = build_state_graph(num_disks)
    draw_graph(
        G=G,
        num_disks=num_disks,
        initial_state=initial_state,
        goal_state=goal_state,
        model_moves=model_moves,
        optimal_moves=optimal_moves,
        title='3-disk TOH state space (model vs optimal)',
        out_prefix='state_space_graph',
        model_color='red',
        model_label='Model (9 moves)',
        optimal_color='#00aa00',
        optimal_label='Optimal (7 moves)',
        fig_size=(16, 14),
        label_font_size=15,
        node_size_regular=350,
    )

    # ===============================
    # 4-disk figure (new, requested)
    # ===============================
    num_disks_4 = 4
    initial_state_4 = ((4, 3, 2, 1), (), ())
    goal_state_4 = ((), (), (4, 3, 2, 1))

    # -----------------------------------------------------------------
    # 4-disk manual state-sequence input (edit these two lists directly)
    # Convention used here: largest -> smallest, e.g., 1111 -> 1113 -> ...
    # -----------------------------------------------------------------
    kimi_failed_states_4 = [
        '1111', '1113', '1123', '1122', '1322', '1321', '1331', '1333',
        '2333', '2332', '2312', '2311', '2211', '2213', '2223', '2222',
    ]

    optimal_states_4 = [
        '1111', '1112', '1132', '1133', '1233', '1231', '1221', '1222',
        '3222', '3223', '3213', '3211', '3311', '3312', '3332', '3333',
    ]

    G4 = build_state_graph(num_disks_4)
    draw_graph(
        G=G4,
        num_disks=num_disks_4,
        initial_state=initial_state_4,
        goal_state=goal_state_4,
        model_moves=[],
        optimal_moves=[],
        model_state_labels=kimi_failed_states_4,
        optimal_state_labels=optimal_states_4,
        labels_largest_to_smallest=True,
        title='4-disk TOH Sierpinski state space (Kimi failed path vs optimal)',
        out_prefix='state_space_graph_4disk',
        model_color='#6b6b6b',
        model_label='Kimi failed attempt (15 moves)',
        optimal_color='#00aa00',
        optimal_label='Optimal (15 moves)',
        fig_size=(28, 24),
        label_font_size=13,
        node_size_regular=170,
        snap_grid=True,
        use_label_layout=True,
        show_paths=True,
    )

    print('Generated both 3-disk and 4-disk Sierpinski visualizations.')


if __name__ == "__main__":
    main()
