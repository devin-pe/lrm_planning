"""
Visualize the full 3-disk Towers of Hanoi state space as a graph.
The state space forms a Sierpinski triangle structure.

Edges in the model's solution are colored red.
Edges in the optimal (BFS) solution are colored green.
"""

import itertools
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


def format_state(state):
    """Format state as compact label e.g. '012' meaning disk1->peg0, disk2->peg1, disk3->peg2."""
    assignment = {}
    for peg_idx, peg in enumerate(state):
        for disk in peg:
            assignment[disk] = peg_idx
    num_disks = len(assignment)
    return ''.join(str(assignment[d] + 1) for d in range(1, num_disks + 1))


def main():
    num_disks = 3
    
    # Initial and goal states from problem_003
    initial_state = ((1,), (3,), (2,))
    goal_state = ((1,), (), (3, 2))
    
    # Model's moves (9 moves)
    model_moves = [
        [2, 2, 1], [1, 0, 2], [2, 1, 0], [1, 2, 1],
        [1, 1, 0], [3, 1, 2], [1, 0, 1], [2, 0, 2], [1, 1, 0],
    ]
    
    # Optimal moves (7 moves from BFS)
    optimal_moves = [
        [1, 0, 1], [2, 2, 0], [1, 1, 0], [3, 1, 2],
        [1, 0, 1], [2, 0, 2], [1, 1, 0],
    ]
    
    # Build the full state space graph
    states = all_states(num_disks)
    G = nx.Graph()
    G.add_nodes_from(states)
    
    for s in states:
        for neighbor in get_neighbors(s):
            if not G.has_edge(s, neighbor):
                G.add_edge(s, neighbor)
    
    print(f"State space: {G.number_of_nodes()} states, {G.number_of_edges()} edges")
    
    # Convert move sequences to edge sets
    model_edges = moves_to_edges(initial_state, model_moves)
    optimal_edges = moves_to_edges(initial_state, optimal_moves)
    
    model_edge_set = set()
    for a, b in model_edges:
        model_edge_set.add((a, b) if a < b else (b, a))
    
    optimal_edge_set = set()
    for a, b in optimal_edges:
        optimal_edge_set.add((a, b) if a < b else (b, a))
    
    # Compute Sierpinski layout
    pos = sierpinski_layout(G, num_disks)
    
    # Classify edges
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
    
    # Classify nodes
    model_states = set()
    for a, b in model_edges:
        model_states.add(a)
        model_states.add(b)
    
    optimal_states = set()
    for a, b in optimal_edges:
        optimal_states.add(a)
        optimal_states.add(b)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # Draw normal edges (gray, thin)
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='#d0d0d0',
                           width=0.8, alpha=0.5, ax=ax)
    
    # Draw model-only edges (red)
    nx.draw_networkx_edges(G, pos, edgelist=model_only_edges, edge_color='red',
                           width=3.0, alpha=0.85, ax=ax)
    
    # Draw optimal-only edges (green)
    nx.draw_networkx_edges(G, pos, edgelist=optimal_only_edges, edge_color='#00aa00',
                           width=3.0, alpha=0.85, ax=ax)
    
    # Draw shared edges (orange/yellow - both solutions use them)
    nx.draw_networkx_edges(G, pos, edgelist=both_edges, edge_color='#ff8800',
                           width=3.5, alpha=0.9, ax=ax)
    
    # Draw nodes
    # Regular nodes
    regular_nodes = [n for n in G.nodes() if n not in model_states and n not in optimal_states]
    both_nodes = [n for n in G.nodes() if n in model_states and n in optimal_states
                  and n != initial_state and n != goal_state]
    model_only_nodes = [n for n in model_states if n not in optimal_states
                        and n != initial_state and n != goal_state]
    optimal_only_nodes = [n for n in optimal_states if n not in model_states
                          and n != initial_state and n != goal_state]
    
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_color='#e8e8e8',
                           node_size=350, edgecolors='#999999', linewidths=0.5, ax=ax)
    
    nx.draw_networkx_nodes(G, pos, nodelist=model_only_nodes, node_color='#ffcccc',
                           node_size=500, edgecolors='red', linewidths=2.0, ax=ax)
    
    nx.draw_networkx_nodes(G, pos, nodelist=optimal_only_nodes, node_color='#ccffcc',
                           node_size=500, edgecolors='#00aa00', linewidths=2.0, ax=ax)
    
    nx.draw_networkx_nodes(G, pos, nodelist=both_nodes, node_color='#ffffaa',
                           node_size=500, edgecolors='#ff8800', linewidths=2.0, ax=ax)
    
    # Draw start and goal nodes prominently
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_state], node_color='#4444ff',
                           node_size=700, edgecolors='black', linewidths=2.5, ax=ax,
                           node_shape='s')
    nx.draw_networkx_nodes(G, pos, nodelist=[goal_state], node_color='gold',
                           node_size=800, edgecolors='black', linewidths=2.5, ax=ax,
                           node_shape='*')
    
    # Add vertex labels offset below/above nodes for better visibility
    labels = {node: format_state(node) for node in G.nodes()}
    label_pos = {}
    for node, p in pos.items():
        # Offset labels slightly below each node
        label_pos[node] = p + np.array([0.0, -0.022])
    nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=13, font_color='#222222',
                            font_family='monospace', font_weight='bold', ax=ax)
    
    # Label start and goal
    offset = np.array([0.0, 0.03])
    ax.annotate("START", pos[initial_state] + offset,
                fontsize=9, ha='center', va='bottom', fontweight='bold',
                color='#4444ff',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.annotate("GOAL", pos[goal_state] + offset,
                fontsize=9, ha='center', va='bottom', fontweight='bold',
                color='#886600',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='red', edgecolor='red', label=f"Model (9 moves)"),
        mpatches.Patch(facecolor='#00aa00', edgecolor='#00aa00', label=f'Optimal (7 moves)'),
        mpatches.Patch(facecolor='#ff8800', edgecolor='#ff8800', label='Shared'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc')
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('state_space_graph.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('state_space_graph.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved state_space_graph.png and state_space_graph.pdf")
    plt.show()


if __name__ == "__main__":
    main()
