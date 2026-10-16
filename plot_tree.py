"""
Visualization for Literati-learned Shape Generalized Trees (SGTs).

Usage:
    from plot_tree import plot_literati_tree
    fig = plot_literati_tree(clf, class_names=['Class A', 'Class B'])
    fig.savefig('tree.png', dpi=150, bbox_inches='tight')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_sample_bin(x_val, thresholds):
    """Determine which bin a feature value falls into.

    Bin boundaries (left-closed, right-open except last):
      bin 0: x <= t[0]
      bin i: t[i-1] < x <= t[i]
      last:  x > t[-1]
    """
    if len(thresholds) == 0:
        return 0
    if x_val <= thresholds[0]:
        return 0
    for i in range(1, len(thresholds)):
        if thresholds[i - 1] < x_val <= thresholds[i]:
            return i
    return len(thresholds)  # last bin


def _compute_layout(nodes):
    """Compute (x_norm, y_norm) positions for all nodes.

    Returns dict: {node_id: (x_norm, y_norm)}.
    Root is at the top (y≈1), leaves at the bottom (y≈0.15).
    x positions are spread so that leaves are evenly spaced.
    """
    # Find root
    root_id = next(nid for nid, n in nodes.items() if n['parent_id'] is None)
    max_depth = max(n['depth'] for n in nodes.values())

    # Assign x by leaf counter (depth-first, left-to-right)
    x_positions = {}
    leaf_counter = [0]

    def _assign_x(nid):
        node = nodes[nid]
        if not node['children']:
            x_positions[nid] = leaf_counter[0]
            leaf_counter[0] += 1
            return x_positions[nid]
        child_xs = [_assign_x(cid) for cid in node['children']]
        x_positions[nid] = sum(child_xs) / len(child_xs)
        return x_positions[nid]

    _assign_x(root_id)

    x_max = max(x_positions.values()) if x_positions else 0

    layout = {}
    for nid, node in nodes.items():
        x_raw = x_positions[nid]
        x_norm = x_raw / max(x_max, 1)
        if max_depth > 0:
            y_norm = 1.0 - (node['depth'] / max_depth) * 0.85
        else:
            y_norm = 0.5
        layout[nid] = (x_norm, y_norm)

    return layout


def _collect_nodes(clf, X_train, feature_dict):
    """Collect node descriptors from a fitted LiteratiClassifier.

    Uses clf.export_tree() to get node dicts, then routes training samples
    through the tree to compute feature_values for histogram display.
    """
    raw_nodes = clf.export_tree()

    # Convert list of dicts to {id: dict} with feature_values added
    nodes = {}
    for d in raw_nodes:
        node = dict(d)  # copy
        nid = node['id']
        node['children'] = list(node.get('children', []))
        nodes[nid] = node

    # Route samples through the tree to compute feature_values at each node
    # Start with all samples at the root
    n_samples = X_train.shape[0]
    node_sample_indices = {0: np.arange(n_samples)}

    # BFS to assign samples to each node
    from collections import deque
    queue = deque([0])
    while queue:
        nid = queue.popleft()
        node = nodes[nid]
        sample_idx = node_sample_indices.get(nid, np.array([], dtype=int))

        if not node.get('is_leaf', True) and 'feature' in node:
            thresholds = node['thresholds']
            split = node['split']
            feature_idx = node.get('feature_idx')

            # If feature_idx not in export, look it up from feature_dict
            if feature_idx is None and feature_dict is not None:
                fname = node['feature']
                for name, indices in feature_dict.items():
                    if name == fname:
                        feature_idx = indices[0]
                        break

            if feature_idx is not None and len(sample_idx) > 0:
                x_feature = X_train[sample_idx, feature_idx]
                node['feature_values'] = x_feature

                # Route samples to children
                bin_assignments = np.array([_get_sample_bin(v, thresholds) for v in x_feature])
                split_arr = np.array(split)
                left_mask = split_arr[bin_assignments] == 1
                left_indices = sample_idx[left_mask]
                right_indices = sample_idx[~left_mask]

                children = node['children']
                if len(children) >= 2:
                    node_sample_indices[children[0]] = left_indices
                    node_sample_indices[children[1]] = right_indices

        for cid in node['children']:
            queue.append(cid)

    return nodes


def _draw_shape_function_node(ax, thresholds, split, feature_values, feature_name,
                               left_color, right_color, fontsize, n_bins=20,
                               show_sample_count=True):
    """Draw shape-function split as background shading + sample histogram.

    Background spans (axvspan) show which region of the feature domain routes
    left (left_color) or right (right_color).  Histogram bars on top show the
    sample count distribution within each fine bin, also coloured by routing.

    Parameters
    ----------
    thresholds     : list of shape-function bin boundaries
    split          : list of 0/1 — 1 = routes LEFT
    feature_values : 1-D numpy array of feature values at this node
    n_bins         : target number of histogram bins; routing-change boundaries
                     are always bin edges (actual bin count may exceed n_bins
                     if n_bins < number of distinct routing regions)
    """
    # ── 1. Data range ─────────────────────────────────────────────────────────
    if len(feature_values) > 0:
        x_min, x_max = float(feature_values.min()), float(feature_values.max())
    else:
        x_min = float(thresholds[0]) - 1 if thresholds else 0.0
        x_max = float(thresholds[-1]) + 1 if thresholds else 1.0
    if x_min >= x_max:
        x_min -= 1.0
        x_max += 1.0

    # ── 2. Build routing regions (merge consecutive same-direction bins) ───────
    regions = []   # (x_start, x_end, is_left: bool)
    i = 0
    region_start = x_min
    while i < len(split):
        cur_dir = split[i]
        j = i + 1
        while j < len(split) and split[j] == cur_dir:
            j += 1
        # boundary between bin j-1 and bin j is thresholds[j-1]
        region_end = float(thresholds[j - 1]) if j < len(split) else x_max
        regions.append((region_start, region_end, cur_dir == 1))
        region_start = region_end
        i = j

    # ── 3. Distribute n_bins across regions (≥ 1 per region, prop. by width) ──
    region_widths = [r[1] - r[0] for r in regions]
    total_width = sum(region_widths) or 1.0
    alloc = [max(1, round(n_bins * w / total_width)) for w in region_widths]
    # Adjust to hit target total
    while sum(alloc) > n_bins and len(alloc) > 1:
        reducible = [(region_widths[k], k) for k in range(len(alloc)) if alloc[k] > 1]
        if not reducible:
            break
        _, idx = max(reducible)
        alloc[idx] -= 1
    while sum(alloc) < n_bins:
        _, idx = max((region_widths[k], k) for k in range(len(alloc)))
        alloc[idx] += 1

    # ── 4. Build bin edges (region boundaries are guaranteed edges) ────────────
    bin_edges = []
    for j, (r_start, r_end, _) in enumerate(regions):
        edges = np.linspace(r_start, r_end, alloc[j] + 1)
        if j == 0:
            bin_edges.extend(edges.tolist())
        else:
            bin_edges.extend(edges[1:].tolist())  # skip duplicate boundary

    # ── 5. Histogram counts ────────────────────────────────────────────────────
    counts, _ = np.histogram(feature_values, bins=bin_edges)

    # ── 6. Per-bar colors ──────────────────────────────────────────────────────
    bar_colors = []
    for j, (_, _, is_left) in enumerate(regions):
        bar_colors.extend([left_color if is_left else right_color] * alloc[j])

    # ── 7. Background shading ─────────────────────────────────────────────────
    for r_start, r_end, is_left in regions:
        ax.axvspan(r_start, r_end,
                   color=left_color if is_left else right_color,
                   alpha=0.5, zorder=0)

    # ── 8. Histogram bars ─────────────────────────────────────────────────────
    centers = [(bin_edges[k] + bin_edges[k + 1]) / 2 for k in range(len(counts))]
    bar_widths = [bin_edges[k + 1] - bin_edges[k] for k in range(len(counts))]
    ax.bar(centers, counts, width=bar_widths, color=bar_colors,
           edgecolor='white', linewidth=0.3, zorder=1)

    # ── 9. Axis decorations ───────────────────────────────────────────────────
    # Only tick at routing-change boundaries (region endpoints except the last),
    # not at every intra-region CART threshold — those produce repeated labels
    # when nearby values round to the same two-decimal-place string.
    disc_thresholds = [r_end for r_start, r_end, _ in regions[:-1]]
    ax.set_xlim(x_min, x_max)
    if disc_thresholds:
        ax.set_xticks(disc_thresholds)
        ax.set_xticklabels(
            [f"{t:.2f}" for t in disc_thresholds],
            fontsize=max(fontsize - 2, 5),
            rotation=30,
            ha='right',
        )
    else:
        ax.set_xticks([])
    ax.set_xlabel(feature_name, fontsize=fontsize, labelpad=1)
    ax.yaxis.set_visible(False)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)

    # Sample count annotation
    if show_sample_count:
        ax.text(1.0, 1.0, f"n={len(feature_values)}", transform=ax.transAxes,
                fontsize=max(fontsize - 2, 5), ha='right', va='top',
                color='#444444')


def _draw_leaf_node(ax, pred_class, n_samples, class_name, side_color, fontsize,
                    show_class_idx=True, show_sample_count=True):
    """Draw a leaf node as a coloured text box."""
    ax.set_facecolor(side_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Top line: class name; bottom line: class index + sample count
    # Sub-label parts
    parts = []
    if show_class_idx:
        parts.append(f"(Class {pred_class})")
    if show_sample_count:
        parts.append(f"n={n_samples}")
    sub_label = "  ".join(parts)

    if sub_label:
        ax.text(0.5, 0.65, class_name, transform=ax.transAxes,
                fontsize=fontsize, ha='center', va='center',
                fontweight='bold', color='#1a1a1a')
        ax.text(0.5, 0.22, sub_label,
                transform=ax.transAxes,
                fontsize=max(fontsize - 1, 5), ha='center', va='center',
                color='#444444')
    else:
        ax.text(0.5, 0.5, class_name, transform=ax.transAxes,
                fontsize=fontsize, ha='center', va='center',
                fontweight='bold', color='#1a1a1a')

    ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_literati_tree(
    clf,
    class_names=None,
    figsize=None,
    left_color='#E8A0BF',
    right_color='#FAC898',
    fontsize=12,
    node_aspect_ratio=2.5,
    title=None,
    n_bins=20,
    show_node_labels=True,
    show_sample_counts=True,
):
    """Visualize the best tree found by a fitted LiteratiClassifier.

    Each internal node shows a two-layer display: background spans coloured by
    routing direction (left/right) and a sample histogram on top.  Leaf nodes
    show the predicted class.

    Parameters
    ----------
    clf              : fitted LiteratiClassifier
    class_names      : list[str] | None
        Human-readable class labels. Defaults to clf.le.classes_ as strings.
    figsize          : (float, float) | None
        Figure size in inches. Inferred from tree structure if None.
    left_color       : str  — colour for regions/bars routing samples LEFT  (default: pink)
    right_color      : str  — colour for regions/bars routing samples RIGHT (default: orange)
    fontsize         : int  — base font size for node labels
    node_aspect_ratio: float — width/height ratio for each node box
    title            : str | None — overall figure title
    n_bins           : int  — target number of histogram bins per internal node (default: 20).
        Routing-change boundaries are always bin edges, so the actual count may
        exceed n_bins when n_bins < number of distinct routing regions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if clf._engine is None:
        raise ValueError("Classifier has not been fitted. Call fit() first.")

    # Resolve class names (track whether user supplied them)
    user_provided_class_names = class_names is not None
    if class_names is None:
        class_names = [str(c) for c in clf.le.classes_]

    nodes = _collect_nodes(clf, clf._X_train, clf._feature_dict)

    if not nodes:
        raise RuntimeError("No nodes collected from tree. Is the model fitted?")

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = _compute_layout(nodes)

    n_leaves = sum(1 for n in nodes.values() if n['is_leaf'])
    max_d = max(n['depth'] for n in nodes.values())

    if figsize is None:
        fig_w = max(n_leaves * 3.0, 8.0)
        fig_h = max((max_d + 1) * 2.8, 4.0)
        figsize = (fig_w, fig_h)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)

    # Invisible background axes covering the full figure for arrows + node labels
    bg_ax = fig.add_axes([0, 0, 1, 1])
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.axis('off')
    bg_ax.set_zorder(0)

    # Node box dimensions in figure-fraction units; shrink when tree is wide.
    # Leaf nodes use the base size; decision nodes are 1.5× wider (and taller).
    node_w = min(0.14, 0.85 / max(n_leaves, 1))
    node_h = node_w / node_aspect_ratio
    internal_node_w = node_w * 1.5
    internal_node_h = internal_node_w / node_aspect_ratio

    def _nw_nh(node):
        """Return (width, height) for a node in figure-fraction units."""
        if node['is_leaf']:
            return node_w, node_h
        return internal_node_w, internal_node_h

    # Vertical layout constants
    # Reserve top 12 % for the figure title + node labels, 3 % at the bottom.
    TOP_MARGIN = 0.12
    BOT_MARGIN = 0.03
    x_margin = internal_node_w * 0.6  # horizontal padding based on largest node
    LABEL_GAP = 0.008                 # gap between top of box and "Node N" text

    def _fig_xy(x_norm, y_norm):
        """Map normalised layout coords → figure fractions."""
        fx = x_margin + x_norm * (1.0 - 2.0 * x_margin)
        fy = BOT_MARGIN + y_norm * (1.0 - TOP_MARGIN - BOT_MARGIN)
        return fx, fy

    # ── Draw edges (arrows) ──────────────────────────────────────────────────
    for nid, node in nodes.items():
        if node['parent_id'] is None:
            continue
        parent_id = node['parent_id']
        pfx, pfy = _fig_xy(*layout[parent_id])
        cfx, cfy = _fig_xy(*layout[nid])

        _, p_nh = _nw_nh(nodes[parent_id])
        _, c_nh = _nw_nh(node)
        arrow_color = left_color if node['side'] == 'left' else right_color

        arrow = FancyArrowPatch(
            posA=(pfx, pfy - p_nh / 2),
            posB=(cfx, cfy + c_nh / 2),
            arrowstyle='-|>',
            color=arrow_color,
            linewidth=2.0,
            mutation_scale=10,
            transform=fig.transFigure,
            figure=fig,
            zorder=1,
        )
        fig.add_artist(arrow)

    # ── Draw nodes + labels ───────────────────────────────────────────────────
    for nid, node in nodes.items():
        fx, fy = _fig_xy(*layout[nid])
        nw, nh = _nw_nh(node)

        ax_node = fig.add_axes([fx - nw / 2, fy - nh / 2, nw, nh])
        ax_node.set_zorder(10)

        if node['is_leaf']:
            side_color = (left_color if node['side'] == 'left'
                          else right_color if node['side'] == 'right'
                          else '#DDDDDD')
            class_idx = int(node['pred_class'])
            class_name = (class_names[class_idx]
                          if class_idx < len(class_names) else str(class_idx))
            _draw_leaf_node(ax_node, class_idx, node['n_samples'],
                            class_name, side_color, fontsize,
                            show_class_idx=not user_provided_class_names,
                            show_sample_count=show_sample_counts)
        else:
            _draw_shape_function_node(
                ax_node,
                thresholds=node['thresholds'],
                split=node['split'],
                feature_values=node['feature_values'],
                feature_name=node['feature'],
                left_color=left_color,
                right_color=right_color,
                fontsize=fontsize,
                n_bins=n_bins,
                show_sample_count=show_sample_counts,
            )

        # "Node N" label added directly to figure so it renders above arrows
        if show_node_labels:
            fig.text(
                fx, fy + nh / 2 + LABEL_GAP,
                f"Node {nid}",
                fontsize=max(fontsize - 1, 5),
                ha='center', va='bottom',
                color='#333333',
                zorder=20,
            )

    if title:
        fig.text(0.5, 0.985, title,
                 fontsize=fontsize + 2, ha='center', va='top',
                 transform=fig.transFigure)

    return fig
