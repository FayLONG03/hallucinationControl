import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

models = ['gpt-5-mini', 'gemini-3.1-flash-lite', 'Meta-Llama-3-8B-Instruct']
model_labels = ['GPT-5 mini', 'Gemini-3.1\nFlash-Lite', 'Meta-Llama-3\n8B-Instruct']
schemes = ['Pure Eval', 'Scheme A', 'Scheme B', 'Scheme B w/ norms']

far_vals = {
    'Meta-Llama-3-8B-Instruct':  [0.683, 0.691, 0.65, 0.515],
    'gemini-3.1-flash-lite':     [0.444, 0.388, 0.395, 0.322],
    'gpt-5-mini':                [0.533, 0.486, 0.418, 0.355],
}
far_errs = {
    'Meta-Llama-3-8B-Instruct':  [0.008, 0.008, 0.010, 0.014],
    'gemini-3.1-flash-lite':     [0.008, 0.009, 0.009, 0.009],
    'gpt-5-mini':                [0.008, 0.008, 0.009, 0.012],
}

coverage_vals = {
    'Meta-Llama-3-8B-Instruct':  [0.963, 0.931, 0.775, 0.447],
    'gemini-3.1-flash-lite':     [0.999, 0.850, 0.839, 0.706],
    'gpt-5-mini':                [0.97, 0.848, 0.689, 0.564],
}
coverage_errs = {
    'Meta-Llama-3-8B-Instruct':  [np.nan, np.nan, np.nan, np.nan],
    'gemini-3.1-flash-lite':     [np.nan, np.nan, np.nan, np.nan],
    'gpt-5-mini':                [np.nan, np.nan, np.nan, np.nan],
}

ece_vals = {
    'Meta-Llama-3-8B-Instruct':  [np.nan, 0.6659, 0.6296, 0.4974],
    'gemini-3.1-flash-lite':     [np.nan, 0.3698, 0.3735, 0.3059],
    'gpt-5-mini':                [np.nan, 0.1436, 0.1485, 0.1474],
}
ece_errs = {m: [np.nan, np.nan, np.nan, np.nan] for m in models}

jabs_vals = {
    'Meta-Llama-3-8B-Instruct':  [np.nan, 0.082, 0.238, 0.49],
    'gemini-3.1-flash-lite':     [np.nan, 0.216, 0.224, 0.38],
    'gpt-5-mini':                [np.nan, 0.212, 0.381, 0.444],
}
jabs_errs = {
    'Meta-Llama-3-8B-Instruct':  [np.nan, np.nan, np.nan, np.nan],
    'gemini-3.1-flash-lite':     [np.nan, np.nan, np.nan, np.nan],
    'gpt-5-mini':                [np.nan, np.nan, np.nan, np.nan],
}

metrics = [
    ('FAR (Answered)', far_vals,   far_errs),
    ('Coverage',        coverage_vals, coverage_errs),
    ('Empirical ECE (Answered)',   ece_vals,   ece_errs),
    ('J_abs',           jabs_vals,  jabs_errs),
]

colors  = ['#003F5C', '#2F7EB1', '#5BB8D4', '#A8DDE9']
x       = np.arange(len(models))
n       = len(schemes)
width   = 0.16
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * (width + 0.01)

fig, axes = plt.subplots(2, 2, figsize=(18, 8))

for ax, (title, vals, errs) in zip(axes.flat, metrics):
    for i, (scheme, color) in enumerate(zip(schemes, colors)):
        v  = [vals[m][i] for m in models]
        e  = [errs[m][i] for m in models]

        v_plot = [vi if not np.isnan(vi) else 0 for vi in v]
        e_plot = [ei if not np.isnan(ei) else 0 for ei in e]
        mask   = [not np.isnan(vi) for vi in v]

        bars = ax.bar(
            x + offsets[i], v_plot, width,
            color=color, alpha=0.85, label=scheme, zorder=3,
        )
        for j, (bar, has_data, vi, ei) in enumerate(zip(bars, mask, v_plot, e_plot)):
            if not has_data:
                bar.set_height(0)
                bar.set_visible(False)
            else:
                if ei > 0:
                    ax.errorbar(
                        bar.get_x() + bar.get_width()/2, vi,
                        yerr=ei, fmt='none', ecolor='#333', elinewidth=1.2, capsize=3, capthick=1.2, zorder=4
                    )

    ax.set_title(title, fontsize=30, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=25)
    ax.tick_params(axis='y', labelsize=24)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_v = [vals[m][i] for m in models for i in range(n) if not np.isnan(vals[m][i])]
    ax.set_ylim(0, (max(all_v) * 1.22) if all_v else 1.0)

handles = [mpatches.Patch(color=c, label=s, alpha=0.85) for c, s in zip(colors, schemes)]
fig.legend(handles=handles, title='Scheme', fontsize=25, title_fontsize=26,
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.20),
           framealpha=0.9, edgecolor='#ccc')

plt.tight_layout(h_pad=3.0)
plt.subplots_adjust(bottom=0.12)
save_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, 'model_comparison.pdf'), bbox_inches='tight')
print("Saved.")