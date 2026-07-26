"""FAR vs. abstention trade-off for the popQA reward schemes.

Authored at final print size for an AAAI two-column layout: 7.0in wide, so the
figure drops into a `figure*` environment at 1:1 with no rescaling. Type sizes
below are therefore the sizes that appear on the page (10-13.5pt).

Encoding
--------
colour + shape -> scheme family (A / B / I-CALM), plus a neutral X for Pure Eval
fill           -> penalty for an incorrect answer (solid = -1, open = 0)
size           -> abstention reward (+0.2 ... +0.8); reference configs have none

The family is deliberately *doubly* encoded. B and I-CALM interleave in the
middle of the range, so hue alone -- or a nearby text label -- leaves individual
points ambiguous; the marker shape settles membership locally. B and I-CALM take
the two most separable slots (orange / blue) because theirs is the comparison
the figure exists to make. Labels stay in ink and are attached by proximity, so
nothing on the canvas can be mistaken for a data point.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

# --------------------------------------------------------------------------
# Chrome
# --------------------------------------------------------------------------
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
# Keep mathtext (beta / gamma) in the same face as the surrounding labels.
rcParams['mathtext.fontset'] = 'dejavusans'
# Camera-ready requirement: embed TrueType, never Type 3.
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

SURFACE = '#ffffff'
INK = '#0b0b0b'
INK_2 = '#3d3c3a'
MUTED = '#8a8983'
AXIS = '#9a9993'
FRONTIER = '#cfcec8'

# Categorical slots 1-3: the only three that clear the all-pairs CVD gates.
# B and I-CALM take blue/orange -- the most separable pair -- since that is the
# comparison being made; A is spatially isolated and can afford aqua.
C_ICALM = '#2a78d6'  # blue
C_B = '#eb6834'      # orange
C_A = '#1baf7a'      # aqua
C_REF = '#7a7973'    # Pure Eval is a reference point, not a competing scheme

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
TOTAL = 14267

# (family, reward, answered, far_answered) per incorrect-answer penalty
runs = {
    -1: [
        ('A',      None, 12093, 0.486),
        ('B',       0.2,  9810, 0.416),
        ('B',       0.4,  9823, 0.418),
        ('B',       0.6,  9735, 0.418),
        ('B',       0.8,  9247, 0.4),
        ('I-CALM',  0.2,  8398, 0.367),
        ('I-CALM',  0.4,  8043, 0.355),
        ('I-CALM',  0.6,  8428, 0.376),
        ('I-CALM',  0.8,  8637, 0.379),
    ],
    0: [
        ('A',      None, 12527, 0.496),
        ('B',       0.2, 10938, 0.449),
        ('B',       0.4, 10905, 0.451),
        ('B',       0.6, 10482, 0.437),
        ('B',       0.8, 10515, 0.435),
        ('I-CALM',  0.2,  9419, 0.408),
        ('I-CALM',  0.4,  9660, 0.414),
        ('I-CALM',  0.6,  8914, 0.389),
        ('I-CALM',  0.8,  9589, 0.414),
    ],
}

pure_eval = (1 - 13843 / TOTAL, 0.533)

COLORS = {'A': C_A, 'B': C_B, 'I-CALM': C_ICALM}
MARKERS = {'A': 'D', 'B': 'o', 'I-CALM': '^'}
# Equalise apparent weight across marker shapes.
MARKER_SCALE = {'D': 0.8, 'o': 1.0, '^': 1.35}
# Marker areas in pt^2, chosen for print: ~8pt to ~13pt across.
SIZES = {0.2: 62, 0.4: 92, 0.6: 128, 0.8: 170}
SIZE_NONE = 112  # A / Pure Eval carry no abstention reward
SIZE_REF = 150   # Pure Eval's X reads smaller than a disc at equal area

FS_LABEL = 13.5    # axis titles
FS_TICK = 12       # tick labels
FS_LEGEND = 9.5    # legend entries
FS_DIRECT = 11.5   # on-plot scheme labels


def abstention(answered):
    return 1 - answered / TOTAL


# --------------------------------------------------------------------------
# Frontier over every configuration (lower FAR and lower abstention are both
# better, so the boundary is the running minimum in FAR).
# --------------------------------------------------------------------------
all_points = [pure_eval] + [
    (abstention(ans), far) for rows in runs.values() for _, _, ans, far in rows
]

frontier = []
best = float('inf')
for x, y in sorted(all_points):
    if y < best:
        frontier.append((x, y))
        best = y
fx, fy = zip(*frontier)

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_axisbelow(True)

# The frontier is computed above but left undrawn -- the markers alone carry
# the trade-off. Re-enable by plotting (fx, fy).

for penalty, rows in runs.items():
    for family, reward, answered, far in rows:
        marker = MARKERS[family]
        size = SIZES.get(reward, SIZE_NONE) * MARKER_SCALE[marker]
        if penalty == -1:
            face, edge, lw = COLORS[family], SURFACE, 1.1
        else:  # open marker -- surface face keeps overlaps readable
            face, edge, lw = SURFACE, COLORS[family], 1.5
        ax.scatter(
            abstention(answered), far,
            marker=marker, s=size,
            facecolors=face, edgecolors=edge, linewidth=lw,
            zorder=5,
        )

ax.scatter(*pure_eval, marker='X', s=SIZE_REF, color=C_REF,
           edgecolors=SURFACE, linewidth=1.1, zorder=5)

# Scheme names are carried by the legend alone -- repeating them on the canvas
# duplicated every entry and made the text compete with the marks.

# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------
ax.set_xlabel('Abstention rate (1 - coverage)', fontsize=FS_LABEL,
              fontweight='bold', labelpad=6, color=INK)
ax.set_ylabel('FAR (answered)', fontsize=FS_LABEL,
              fontweight='bold', labelpad=6, color=INK)

ax.set_xlim(0.0, 0.52)
ax.set_ylim(0.33, 0.55)
ax.set_xticks(np.arange(0.0, 0.51, 0.05))
ax.set_yticks(np.arange(0.35, 0.551, 0.05))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(AXIS)
    ax.spines[side].set_linewidth(0.8)

ax.tick_params(axis='both', labelsize=FS_TICK, colors=INK_2,
               length=5.5, width=1.1, pad=3.5, top=False, right=False)

# --------------------------------------------------------------------------
# Legends -- in the empty lower-left corner, stacked upward
# --------------------------------------------------------------------------
def proxy(marker, size, face, edge, lw, label):
    return Line2D([], [], linestyle='none', marker=marker, markersize=size,
                  markerfacecolor=face, markeredgecolor=edge,
                  markeredgewidth=lw, label=label)


scheme_handles = [
    proxy(MARKERS[f], np.sqrt(SIZE_NONE * MARKER_SCALE[MARKERS[f]]),
          COLORS[f], SURFACE, 1.1, f)
    for f in ('A', 'B', 'I-CALM')
] + [proxy('X', np.sqrt(SIZE_REF), C_REF, SURFACE, 1.1, 'Pure Eval')]

penalty_handles = [
    proxy('o', np.sqrt(SIZE_NONE), INK_2, SURFACE, 1.1, '$-1$'),
    proxy('o', np.sqrt(SIZE_NONE), SURFACE, INK_2, 1.5, '$0$'),
]
reward_handles = [
    proxy('o', np.sqrt(SIZES[r]), MUTED, SURFACE, 1.1, f'$+{r}$')
    for r in (0.2, 0.4, 0.6, 0.8)
]

# The scheme key sits in the empty top-right corner; the two reward-setting
# keys stay together in the lower-left. Each cluster gets its own frame.
legend_specs = [
    # handles, title, loc, anchor, ncol, columnspacing, cluster
    (scheme_handles, 'Scheme',
     'upper right', (0.988, 0.972), 2, 1.4, 'top-right'),
    (penalty_handles, 'Penalty for an incorrect answer',
     'lower left', (0.030, 0.145), 2, 1.4, 'lower-left'),
    (reward_handles, 'Abstention reward',
     'lower left', (0.030, 0.020), 4, 0.8, 'lower-left'),
]
clusters = {}
for handles, title, loc, anchor, ncol, colspace, cluster in legend_specs:
    leg = ax.legend(
        handles=handles, title=title,
        loc=loc, bbox_to_anchor=anchor, ncol=ncol,
        bbox_transform=ax.transAxes,
        fontsize=FS_LEGEND, title_fontsize=FS_LEGEND, frameon=False,
        handletextpad=0.45, columnspacing=colspace,
        labelspacing=0.4, borderpad=0.0, handlelength=1.0,
    )
    leg.get_title().set_color(INK)
    leg.get_title().set_fontweight('bold')
    leg._legend_box.align = 'left'
    leg.set_zorder(8)
    for text in leg.get_texts():
        text.set_color(INK_2)
    ax.add_artist(leg)
    clusters.setdefault(cluster, []).append(leg)

fig.subplots_adjust(left=0.132, right=0.992, top=0.975, bottom=0.165)

# matplotlib can only frame a single legend, so each cluster gets a patch
# fitted to the union of its legends' rendered extents.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transAxes.inverted()
pad_x, pad_y = 0.020, 0.030
for legs in clusters.values():
    union = Bbox.union([leg.get_window_extent(renderer) for leg in legs])
    (x0, y0) = inv.transform((union.x0, union.y0))
    (x1, y1) = inv.transform((union.x1, union.y1))
    ax.add_patch(FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y),
        (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y,
        boxstyle='round,pad=0,rounding_size=0.010',
        transform=ax.transAxes, facecolor=SURFACE, edgecolor=AXIS,
        linewidth=0.7, zorder=7, clip_on=False,
    ))

save_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(save_dir, 'far_vs_abstention.pdf'), facecolor=SURFACE)
fig.savefig(os.path.join(save_dir, 'far_vs_abstention.png'), dpi=400,
            facecolor=SURFACE)
plt.close(fig)
print('Saved far_vs_abstention.pdf / .png  (7.0 x 3.5 in, AAAI full width)')
