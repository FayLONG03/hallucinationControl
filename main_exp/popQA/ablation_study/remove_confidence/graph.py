import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import gaussian_kde, binom


# Increase default font sizes for readability across all plots
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

def plot_idk_confidence_hist(df, file_label):
    """
    Plots a combined histogram where bar heights represent proportion
    of total samples (true density = count / total).
    For file_label 'B', uses best_guess_confidence for IDK=1;
    otherwise uses first_confidence for both.
    Shows correct vs incorrect breakdown for each confidence bin.
    """

    df_plot = df.copy()
    df_plot.loc[df_plot['idk_flag'] == 0, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 0, 'first_confidence']
    df_plot.loc[df_plot['idk_flag'] == 1, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 1, 'best_guess_confidence']
    if file_label.upper() == 'B':
        output_name = "idk_confidence_density_B.png"
    else:
        # df_plot = df.copy()
        # df_plot['confidence'] = df_plot['first_confidence']
        output_name = "idk_confidence_density_A.png"

    # Clean data
    df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
    df_plot = df_plot.dropna(subset=['confidence'])
    df_plot['confidence'] = df_plot['confidence'].clip(0, 1)

    # Create bins for 0-1 range (20 bins of 0.05 each)
    bins = np.linspace(0, 1, 21)
    
    # Split by IDK flag and correctness
    idk_0_correct = df_plot.loc[(df_plot['idk_flag'] == 0) & (df_plot['correct'] == 1), 'confidence']
    idk_0_incorrect = df_plot.loc[(df_plot['idk_flag'] == 0) & (df_plot['correct'] == 0), 'confidence']
    idk_1_correct = df_plot.loc[(df_plot['idk_flag'] == 1) & (df_plot['correct'] == 1), 'confidence']
    idk_1_incorrect = df_plot.loc[(df_plot['idk_flag'] == 1) & (df_plot['correct'] == 0), 'confidence']

    total_count = len(df_plot)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram data manually for stacking
    bin_edges = list(bins)
    
    # For IDK=0 (Answered)
    if len(idk_0_correct) > 0 or len(idk_0_incorrect) > 0:
        hist_0_correct, _ = np.histogram(idk_0_correct, bins=bin_edges)
        hist_0_incorrect, _ = np.histogram(idk_0_incorrect, bins=bin_edges)
        
        # Normalize
        hist_0_correct_norm = hist_0_correct / total_count
        hist_0_incorrect_norm = hist_0_incorrect / total_count
        
        # Plot as stacked bars
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04  # bar width (adjusted for 0-1 scale)
        
        ax.bar(bin_centers, hist_0_correct_norm, width=width, 
               label='Answered (Correct)', color='green', alpha=0.6, edgecolor='black')
        ax.bar(bin_centers, hist_0_incorrect_norm, width=width, bottom=hist_0_correct_norm,
               label='Answered (Incorrect)', color='red', alpha=0.6, edgecolor='black')
    
    # For IDK=1 
    if len(idk_1_correct) > 0 or len(idk_1_incorrect) > 0:
        hist_1_correct, _ = np.histogram(idk_1_correct, bins=bin_edges)
        hist_1_incorrect, _ = np.histogram(idk_1_incorrect, bins=bin_edges)
        
        # Normalize
        hist_1_correct_norm = hist_1_correct / total_count
        hist_1_incorrect_norm = hist_1_incorrect / total_count
        
        # Offset bars slightly for visibility
        bin_centers_offset = [(bin_edges[i] + bin_edges[i+1])/2 + width for i in range(len(bin_edges)-1)]
        
        if file_label.upper() == 'A':
            label_prefix = "Said IDK"
        else:
            label_prefix = "Best Guess"
            
        ax.bar(bin_centers_offset, hist_1_correct_norm, width=width,
               label=f'{label_prefix} (Correct)', color='lightgreen', alpha=0.6, edgecolor='black')
        ax.bar(bin_centers_offset, hist_1_incorrect_norm, width=width, bottom=hist_1_correct_norm,
               label=f'{label_prefix} (Incorrect)', color='lightcoral', alpha=0.6, edgecolor='black')

    ax.set_title(f'Confidence Distribution with Correctness ({file_label})')
    ax.set_xlabel('Confidence (0-1)')
    ax.set_ylabel('Density (Proportion of Total)')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save
    #output_dir = "experiment_1/initial_exp/results/idk_confidence_density"
    output_dir = "gpt5_all_results/idk_confidence_density"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = output_name.rsplit('.', 1)[0]  # Remove extension
    extension = output_name.rsplit('.', 1)[1]  # Get extension
    output_name_with_time = f"{base_name}_{timestamp}.{extension}"
    
    output_path = os.path.join(output_dir, output_name_with_time)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved true density histogram for file {file_label} → {output_path}")

def plot_idk_confidence_separate(df, file_label, reward_setting):
    """
    Plots separate histograms for Answered (IDK=0) and Said IDK/Best Guess (IDK=1).
    Each plot shows correct vs incorrect breakdown as stacked bars.
    For baseline data (no idk_flag), plots a single histogram of all data.
    """
    # Save
    #output_dir = "experiment_1/initial_exp/results/plot_idk_confidence_separate"
    output_dir = "results/plot_idk_confidence_separate"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{file_label}_({reward_setting})_{timestamp}.png"
    output_path = os.path.join(output_dir, output_name)

    df_plot = df.copy()
    
    # Check if this is baseline data (no idk_flag column or 'Baseline' in label)
    is_baseline = 'Baseline' in file_label
    
    if not is_baseline:
        df_plot.loc[df_plot['idk_flag'] == 0, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 0, 'first_confidence']
        df_plot.loc[df_plot['idk_flag'] == 1, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 1, 'best_guess_confidence']
    
    idk1_label = "Best Guess (after saying IDK)"

    # Clean data
    df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
    df_plot = df_plot.dropna(subset=['confidence'])
    df_plot['confidence'] = df_plot['confidence'].clip(0, 1)

    # Handle baseline case - single plot
    if is_baseline:
        bins = np.linspace(0, 1, 21)
        bin_edges = list(bins)
        total_count = len(df_plot)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get correct and incorrect data
        correct_data = df_plot.loc[df_plot['correct'] == 1, 'confidence']
        incorrect_data = df_plot.loc[df_plot['correct'] == 0, 'confidence']
        
        if len(correct_data) > 0 or len(incorrect_data) > 0:
            hist_correct, _ = np.histogram(correct_data, bins=bin_edges)
            hist_incorrect, _ = np.histogram(incorrect_data, bins=bin_edges)
            
            # Normalize by total dataset count
            hist_correct_norm = hist_correct / total_count
            hist_incorrect_norm = hist_incorrect / total_count
            
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            width = 0.04
            
            ax.bar(bin_centers, hist_correct_norm, width=width,
                   label='Correct', color='green', alpha=0.7, edgecolor='black')
            ax.bar(bin_centers, hist_incorrect_norm, width=width, bottom=hist_correct_norm,
                   label='Incorrect', color='red', alpha=0.7, edgecolor='black')
        
        ax.set_title(f'Confidence Distribution ({file_label}; {reward_setting})', fontsize=18, fontweight='bold')
        ax.set_xlabel('Confidence (0-1)')
        ax.set_ylabel('Proportion of Total Dataset')
        ax.set_ylim(0, 0.6)
        ax.legend()
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"✅ Saved confidence histogram for baseline {file_label}:{reward_setting} → {output_path}")
        return

    # Create bins for 0-1 range (20 bins of 0.05 each)
    bins = np.linspace(0, 1, 21)
    bin_edges = list(bins)
    
    # Get total count for normalization
    total_count = len(df_plot)
    
    # Create two separate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---- Plot 1: Answered (IDK=0) ----
    idk_0 = df_plot[df_plot['idk_flag'] == 0]
    if len(idk_0) > 0:
        idk_0_correct = idk_0.loc[idk_0['correct'] == 1, 'confidence']
        idk_0_incorrect = idk_0.loc[idk_0['correct'] == 0, 'confidence']
        
        hist_0_correct, _ = np.histogram(idk_0_correct, bins=bin_edges)
        hist_0_incorrect, _ = np.histogram(idk_0_incorrect, bins=bin_edges)
        
        # Normalize by total dataset count (not just answered)
        hist_0_correct_norm = hist_0_correct / total_count
        hist_0_incorrect_norm = hist_0_incorrect / total_count
        
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04  # bar width (adjusted for 0-1 scale)
        
        ax1.bar(bin_centers, hist_0_correct_norm, width=width,
               label='Correct', color='green', alpha=0.7, edgecolor='black')
        ax1.bar(bin_centers, hist_0_incorrect_norm, width=width, bottom=hist_0_correct_norm,
               label='Incorrect', color='red', alpha=0.7, edgecolor='black')
        
        ax1.set_title(f'Answered ({file_label}; {reward_setting})', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Confidence (0-1)')
        ax1.set_ylabel('Proportion of Total Dataset')
        ax1.set_ylim(0, 0.6)
        ax1.legend()
        ax1.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax1.text(0.5, 0.5, 'No data for Answered', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Answered ({file_label}; {reward_setting})', fontsize=18, fontweight='bold')
    
    # ---- Plot 2: Said IDK / Best Guess (IDK=1) ----
    idk_1 = df_plot[df_plot['idk_flag'] == 1]
    if len(idk_1) > 0:
        idk_1_correct = idk_1.loc[idk_1['correct'] == 1, 'confidence']
        idk_1_incorrect = idk_1.loc[idk_1['correct'] == 0, 'confidence']
        
        hist_1_correct, _ = np.histogram(idk_1_correct, bins=bin_edges)
        hist_1_incorrect, _ = np.histogram(idk_1_incorrect, bins=bin_edges)
        
        # Normalize by total dataset count (not just IDK)
        hist_1_correct_norm = hist_1_correct / total_count
        hist_1_incorrect_norm = hist_1_incorrect / total_count
        
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04  # bar width (adjusted for 0-1 scale)
        
        ax2.bar(bin_centers, hist_1_correct_norm, width=width,
                   label='Correct', color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.bar(bin_centers, hist_1_incorrect_norm, width=width, bottom=hist_1_correct_norm,
               label='Incorrect', color='lightcoral', alpha=0.7, edgecolor='black')
            
        
        ax2.set_title(f'{idk1_label} ({file_label}; {reward_setting})', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Confidence (0-1)')
        ax2.set_ylabel('Proportion of Total Dataset')
        ax2.set_ylim(0, 0.6)
        ax2.legend()
        ax2.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, f'No data for {idk1_label}', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{idk1_label} ({file_label}; {reward_setting})', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Saved separate confidence histograms for file {file_label}; {reward_setting} → {output_path}")



def plot_confidence_by_correctness(df, exp_label, reward_setting):
    """
    Plots two side-by-side histograms showing confidence distribution
    separated by correctness (correct=0 vs correct=1).
    For Experiment B, combines first_confidence and best_guess_confidence.
    """
    df_plot = df.copy()
    if exp_label != 'A Baseline':
        df_plot.loc[df_plot['idk_flag'] == 0, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 0, 'first_confidence']
        df_plot.loc[df_plot['idk_flag'] == 1, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 1, 'best_guess_confidence']
    
    # Clean data
    df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
    df_plot = df_plot.dropna(subset=['confidence'])
    df_plot['confidence'] = df_plot['confidence'].clip(0, 1)
    
    # Create bins for 0-1 range (20 bins of 0.05 each)
    bins = np.linspace(0, 1, 21)
    bin_edges = list(bins)
    
    # Split by correctness
    correct_data = df_plot.loc[df_plot['correct'] == 1, 'confidence']
    incorrect_data = df_plot.loc[df_plot['correct'] == 0, 'confidence']

    # Create single plot with both distributions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_smooth = np.linspace(0, 1, 300)
    max_density = 0

    # Compute proportions so that correct + incorrect integrate to 1
    n_correct = len(correct_data)
    n_incorrect = len(incorrect_data)
    n_total = n_correct + n_incorrect
    p_correct = n_correct / n_total if n_total > 0 else 0
    p_incorrect = n_incorrect / n_total if n_total > 0 else 0

    # Plot correct answers (green) scaled by p_correct
    density_correct = np.zeros_like(x_smooth)
    if n_correct > 0:
        kde_correct = gaussian_kde(correct_data.values, bw_method='scott')
        density_correct = kde_correct(x_smooth) * p_correct
        max_density = max(max_density, density_correct.max())
        ax.plot(x_smooth, density_correct, color='green', linewidth=2,
                label=f'Correct')
        ax.fill_between(x_smooth, density_correct, alpha=0.3, color='green')

    # Plot incorrect answers (red) scaled by p_incorrect
    density_incorrect = np.zeros_like(x_smooth)
    if n_incorrect > 0:
        kde_incorrect = gaussian_kde(incorrect_data.values, bw_method='scott')
        density_incorrect = kde_incorrect(x_smooth) * p_incorrect
        max_density = max(max_density, density_incorrect.max())
        ax.plot(x_smooth, density_incorrect, color='red', linewidth=2,
                label=f'Incorrect')
        ax.fill_between(x_smooth, density_incorrect, alpha=0.3, color='red')
    
    # Configure plot
    ax.set_title(f'Confidence Distribution({exp_label}; {reward_setting})', fontsize=18, fontweight='bold')
    ax.set_xlabel('Confidence (0-1)')
    ax.set_ylabel('Density')
    ax.set_ylim(0, 3)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save
    #output_dir = "experiment_1/initial_exp/results/confidence_by_correctness"
    output_dir = "results/confidence_by_correctness"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"confidence_by_correctness_{exp_label}_({reward_setting})_{timestamp}.png"
    output_path = os.path.join(output_dir, output_name)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Saved confidence by correctness plot for {exp_label}; {reward_setting} → {output_path}")


def plot_confidence_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, df_b_norm: pd.DataFrame,
                               labels: tuple = ("scheme A", "scheme B", "scheme B with norm")):
    """
    Plot confidence distributions (KDE) for three datasets on the same axes.

    Each dataset is restricted to answered rows (`idk_flag == 0`) and uses
    `first_confidence`. KDEs are normalized so each curve integrates to 1,
    making shape comparisons straightforward.
    """
    datasets = [df_a, df_b, df_b_norm]
    label_names = list(labels)

    prepared = []
    counts = []
    for df in datasets:
        # Follow existing plotting convention: use `first_confidence` when
        # `idk_flag == 0`, otherwise use `best_guess_confidence` for IDK rows.
        d = df.copy()
        d.loc[d['idk_flag'] == 0, 'confidence'] = d.loc[d['idk_flag'] == 0, 'first_confidence']
        d.loc[d['idk_flag'] == 1, 'confidence'] = d.loc[d['idk_flag'] == 1, 'best_guess_confidence']
        d['confidence'] = pd.to_numeric(d['confidence'], errors='coerce')
        d = d.dropna(subset=['confidence'])
        confidences = d['confidence'].clip(0, 1).values
        prepared.append(confidences)
        counts.append(len(confidences))

    x_smooth = np.linspace(0, 1, 400)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for confidences, name, c, n in zip(prepared, label_names, colors, counts):
        if len(confidences) < 2:
            # Not enough data to estimate KDE; skip but note in legend
            ax.plot([], [], color=c, label=f"{name} (n={n}) — insufficient data")
            continue
        kde = gaussian_kde(confidences, bw_method='scott')
        density = kde(x_smooth)
        # Normalize to unit area
        dx = x_smooth[1] - x_smooth[0]
        area = np.trapz(density, x_smooth)
        if area > 0:
            density = density / area
        ax.plot(x_smooth, density, color=c, linewidth=2, label=f"{name}")
        ax.fill_between(x_smooth, density, alpha=0.25, color=c)

    ax.set_title('Confidence Distribution Comparison', fontsize=18, fontweight='bold')
    ax.set_xlabel('Confidence (0-1)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    #output_dir = "experiment_1/initial_exp/results/confidence_comparison"
    output_dir = "gpt5_all_results/confidence_comparison"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"confidence_comparison_{timestamp}.png"
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved confidence comparison plot → {output_path}")


def calculate_ci(successes, trials, confidence=0.95):
    if trials == 0:
        return (0, 0)
    lower = binom.ppf((1 - confidence) / 2, trials, successes / trials) / trials
    upper = binom.ppf(1 - (1 - confidence) / 2, trials, successes / trials) / trials
    return (lower, upper)


# Modify confidence interval output to show +/- format
def calculate_ci_pm(successes, trials, confidence=0.95):
    if trials == 0:
        return 0
    p_hat = successes / trials
    margin = binom.ppf(1 - (1 - confidence) / 2, trials, p_hat) / trials - p_hat
    return margin

def summarize(df, label):

        acc = df["correct"].mean()
        answered = df["idk_flag"] == 0 
        far_e1 = df.loc[answered, "false_answer_flag"].sum() / answered.sum()
        far_e2 = 1 - acc
        #far_e3 = df.loc[answered, "false_answer_flag"].sum() / len(df) if len(df) > 0 else 0
        print(f"\n📊 {label} Summary:")


        far_e1_margin = calculate_ci_pm(df.loc[answered, "false_answer_flag"].sum(), answered.sum())
        far_e2_margin = calculate_ci_pm(len(df) - df["correct"].sum(), len(df))
        #far_e3_margin = calculate_ci_pm(df.loc[answered, "false_answer_flag"].sum(), len(df))

        print(f"  False-Answer Rate - E1: {far_e1:.3f} ± {far_e1_margin:.3f}")
        print(f"  False-Answer Rate - E2: {far_e2:.3f} ± {far_e2_margin:.3f}")
        #print(f"  False-Answer Rate - E3: {far_e3:.3f} ± {far_e3_margin:.3f}")

        # calculate total score
        total_score = df["score"].sum()
        print(f"  Total Score: {total_score} out of {len(df)}")

        answered = df["idk_flag"] == 0
        wrong_answered = df[answered & (df["correct"] == 0)]
        print(f"  Answered questions: {answered.sum()} out of {len(df)}")
        print(f"  Wrong answered questions: {len(wrong_answered)} out of {answered.sum()}")
        
        # calculate number of abstain question out of wrong answered questions
        wrong_answered = df[df["correct"] == 0]
        print(f"  Total wrong answered questions: {len(wrong_answered)}")
        abstain_out_of_wrong = (len(wrong_answered[wrong_answered["idk_flag"] == 1]) / len(wrong_answered)) if len(wrong_answered) > 0 else 0
        print(f"  Abstain out of wrong answered questions: {abstain_out_of_wrong:.3f}")

        # CAR (answered correct / all correct): of the questions the model got
        # right, the share it actually surfaced (didn't abstain on).
        total_correct = (df["correct"] == 1).sum()
        answered_correct = ((df["idk_flag"] == 0) & (df["correct"] == 1)).sum()
        car = (answered_correct / total_correct) if total_correct else float("nan")
        print(f"  CAR (answered correct / all correct): {car:.3f}")

        



def get_latest_file(pattern):
    """Find the most recent file matching the pattern."""
    import glob
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    # Sort by modification time, most recent first
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main():

    correct = "+1, "
    wrong = "-1"
    abstain = ", +0.4"
    reward_setting_A = f"{correct}{wrong}"
    reward_setting_B = f"{correct}{wrong}{abstain}"

    # Summarize every gpt5 result CSV in this ablation folder.
    import glob
    files = sorted(glob.glob("gpt5_results/popqa_*_results_*.csv"))
    if not files:
        raise FileNotFoundError("No gpt5 result CSVs found in gpt5_results/")

    print(f"📊 Loaded data files:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    for f in files:
        summarize(pd.read_csv(f), os.path.basename(f))

    # plot_idk_confidence_separate(df_A_baseline, 'A Baseline', reward_setting_A)
    # plot_idk_confidence_separate(df_A, 'A', reward_setting_A)
    # plot_idk_confidence_separate(df_B, 'B', reward_setting_B)
    # plot_idk_confidence_separate(df_B_control, 'B Control', reward_setting_B)
    # plot_idk_confidence_separate(df_B_norm, 'B norms', reward_setting_B)
    
    # plot_confidence_by_correctness(df_A_baseline, 'A Baseline', reward_setting_A)
    # plot_confidence_by_correctness(df_A, 'A', reward_setting_A)
    # plot_confidence_by_correctness(df_B, 'B', reward_setting_B)
    # plot_confidence_by_correctness(df_B_control, 'B Control', reward_setting_B)
    # plot_confidence_by_correctness(df_B_norm, 'B norms', reward_setting_B)


    # Create a single comparison plot for A, B, and B_norm
    # plot_confidence_comparison(df_A, df_B, df_B_norm)


if __name__ == "__main__":
    main()