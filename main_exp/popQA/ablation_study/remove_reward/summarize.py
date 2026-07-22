import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import gaussian_kde, binom
from typing import Tuple, Dict


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

def plot_idk_confidence_separate(df, file_label):
    """
    Plots separate histograms for Answered (IDK=0) and Said IDK/Best Guess (IDK=1).
    Each plot shows correct vs incorrect breakdown as stacked bars.
    For baseline data (no idk_flag), plots a single histogram of all data.
    """
    # Save
    #output_dir = "experiment_1/initial_exp/results/plot_idk_confidence_separate"
    output_dir = "gpt5_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{file_label}_{timestamp}.png"
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
        
        ax.set_title(f'Confidence Distribution ({file_label})', fontsize=18, fontweight='bold')
        ax.set_xlabel('Confidence (0-1)')
        ax.set_ylabel('Proportion of Total Dataset')
        ax.set_ylim(0, 0.6)
        ax.legend()
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"✅ Saved confidence histogram for baseline {file_label} → {output_path}")
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
        
        ax1.set_title(f'Answered ({file_label})', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Confidence (0-1)')
        ax1.set_ylabel('Proportion of Total Dataset')
        ax1.set_ylim(0, 0.6)
        ax1.legend()
        ax1.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax1.text(0.5, 0.5, 'No data for Answered', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Answered ({file_label})', fontsize=18, fontweight='bold')
    
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
            
        
        ax2.set_title(f'{idk1_label} ({file_label})', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Confidence (0-1)')
        ax2.set_ylabel('Proportion of Total Dataset')
        ax2.set_ylim(0, 0.6)
        ax2.legend()
        ax2.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, f'No data for {idk1_label}', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{idk1_label} ({file_label})', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Saved separate confidence histograms for file {file_label} → {output_path}")



def plot_confidence_by_correctness(df, exp_label):
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
    ax.set_title(f'Confidence Distribution ({exp_label})', fontsize=18, fontweight='bold')
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
    output_dir = "gpt5_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"confidence_by_correctness_{exp_label}_{timestamp}.png"
    output_path = os.path.join(output_dir, output_name)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Saved confidence by correctness plot for {exp_label} → {output_path}")


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
    output_dir = "gpt5_results"
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
        guess = df["idk_flag"] == 1  # second-round best-guess rows
        total = len(df)

        far_e1 = df.loc[answered, "false_answer_flag"].sum() / answered.sum()
        far_e2 = 1 - acc
        far_e1_margin = calculate_ci_pm(df.loc[answered, "false_answer_flag"].sum(), answered.sum())
        far_e2_margin = calculate_ci_pm(len(df) - df["correct"].sum(), len(df))

        coverage = answered.sum() / total if total else float("nan")
        # Best-guess wrongness uses `correct`, not false_answer_flag: for IDK rows the
        # model abstained, so false_answer_flag is 0 even when the best guess is wrong.
        guess_wrong = ((df["idk_flag"] == 1) & (df["correct"] == 0)).sum()
        far_guess = (guess_wrong / guess.sum()) if guess.sum() else float("nan")
        far_guess_margin = calculate_ci_pm(guess_wrong, guess.sum())
        delta_risk = far_guess - far_e1

        # No reward is assigned in this ablation, so there is no "score" column.
        total_utility = df["score"].sum() if "score" in df.columns else float("nan")
        total_wrong = (df["correct"] == 0).sum()
        total_correct = (df["correct"] == 1).sum()
        answered_wrong = ((df["idk_flag"] == 0) & (df["correct"] == 0)).sum()
        answered_correct = ((df["idk_flag"] == 0) & (df["correct"] == 1)).sum()
        surfaced_wrong = df.loc[answered, "false_answer_flag"].sum()
        ecr = (guess_wrong / total_wrong) if total_wrong else float("nan")
        car = (answered_correct / total_correct) if total_correct else float("nan")
        hallucination_rate = (surfaced_wrong / total) if total else float("nan")

        # Printed in the same order as popQA/eval.py's summarize().
        print(f"\n📊 {label} Summary:")
        print(f"  Coverage: {coverage:.3f} ({answered.sum()}/{total})")
        print(f"  False-Answer Rate - Answered: {far_e1:.3f} ± {far_e1_margin:.3f}")
        print(f"  False-Answer Rate - Guess (2nd round): {far_guess:.3f} ± {far_guess_margin:.3f}")
        print(f"  False-Answer Rate - Overall: {far_e2:.3f} ± {far_e2_margin:.3f}")
        print(f"  Delta Risk (FAR guess - FAR answered): {delta_risk:.3f}")
        print(f"  Total Utility: {total_utility}")
        print(f"  Answered questions: {answered.sum()} out of {total}")
        print(f"  Wrong answered questions: {answered_wrong} out of {answered.sum()}")
        print(f"  Total wrong answered questions: {total_wrong}")
        print(f"  ECR (abstained wrong / all wrong): {ecr:.3f}")
        print(f"  CAR (answered correct / all correct): {car:.3f}")
        print(f"  Hallucination Rate (surfaced wrong / total): {hallucination_rate:.3f}")



def calculate_brier_score(df: pd.DataFrame) -> float:
    """
    Calculate the Brier score for the dataset.

    Brier score = mean((confidence - correctness)^2)
    """
    df['confidence'] = df.apply(
        lambda row: row['first_confidence'] if row['idk_flag'] == 0 else row['best_guess_confidence'], axis=1
    )
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['confidence'] = df['confidence'].clip(0, 1)

    brier_score = np.mean((df['confidence'] - df['correct']) ** 2)
    return brier_score


def calculate_brier_score_with_ci(df: pd.DataFrame, confidence: float = 0.95, answered_only: bool = False, file_label = None) -> Tuple[float, float]:
    """
    Calculate the Brier score and its confidence interval for the dataset.

    Args:
        df: DataFrame containing the data.
        confidence: Confidence level for the interval (default: 95%).
        file_label: Optional label for the file (not used in calculation).
    Returns:
        A tuple containing the Brier score and the margin of error (Brier score ± margin).
    """
    # If requested, restrict to answered rows only and use their first_confidence.

    if file_label != "A_Baseline":
        if answered_only:
            df = df[df['idk_flag'] == 0].copy()
            df['confidence'] = pd.to_numeric(df['first_confidence'], errors='coerce')
        else:
            df['confidence'] = df.apply(
            lambda row: row['first_confidence'] if row['idk_flag'] == 0 else row['best_guess_confidence'], axis=1
            )
    else:
        df = df[df['idk_flag'] == 0].copy()
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['confidence'] = df['confidence'].clip(0, 1)

    errors = (df['confidence'] - df['correct']) ** 2
    errors = errors.dropna()  # Remove NaN values

    if len(errors) == 0:
        return float('nan'), float('nan')

    brier_score = np.mean(errors)

    # Calculate confidence interval using binomial approximation
    def calculate_ci_pm(successes, trials, confidence=0.95):
        if trials == 0:
            return 0
        p_hat = successes / trials
        margin = binom.ppf(1 - (1 - confidence) / 2, trials, p_hat) / trials - p_hat
        return margin

    n = len(errors)
    # For the margin calculation we treat the sum of squared-errors as "successes"
    # in a binomial approximation (legacy choice). If you want a different
    # CI (e.g. bootstrap), we can change this.
    margin = calculate_ci_pm(np.sum(errors), n, confidence)

    return brier_score, margin



def calculate_calibration_error(df: pd.DataFrame, file_label: str, num_bins: int = 10) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE = (1/m) * sum over bins: m_b * |o_hat_b - p_hat_b|
    
    where:
    - m is total number of samples
    - m_b is number of samples in bin b
    - o_hat_b is empirical frequency (accuracy) in bin b
    - p_hat_b is average predicted probability (self reported confidence) in bin b
    
    For file_label 'B', uses best_guess_confidence for IDK=1;
    otherwise uses first_confidence for both.
    """
    
    df_calc = df.copy()
    if file_label != "A_Baseline":
        df_calc.loc[df_calc['idk_flag'] == 0, 'confidence'] = df_calc.loc[df_calc['idk_flag'] == 0, 'first_confidence']
        df_calc.loc[df_calc['idk_flag'] == 1, 'confidence'] = df_calc.loc[df_calc['idk_flag'] == 1, 'best_guess_confidence']
    
    # Clean data
    df_calc['confidence'] = pd.to_numeric(df_calc['confidence'], errors='coerce')
    df_calc['confidence'] = df_calc['confidence'].clip(0, 100)

    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    df_calc['bin'] = pd.cut(df_calc['confidence'], bins=bins, include_lowest=True, labels=False)
    
    # Calculate per-bin statistics
    bin_stats = []
    ece = 0.0
    total_samples = len(df_calc)  # m in the formula
    
    for bin_idx in range(num_bins):
        bin_data = df_calc[df_calc['bin'] == bin_idx]
        
        if len(bin_data) > 0:
            bin_count = len(bin_data)  # m_b in the formula
            
            avg_confidence = bin_data['confidence'].mean()

            empirical_frequency = bin_data['correct'].mean()
            
            calibration_gap = abs(empirical_frequency - avg_confidence)
            
            ece += (bin_count / total_samples) * calibration_gap
            
            bin_stats.append({
                'bin_idx': bin_idx,
                'bin_range': f'[{bins[bin_idx]:.2f}, {bins[bin_idx+1]:.2f}]',
                'count': bin_count,
                'avg_predicted_prob': avg_confidence,  # p_hat_b
                'empirical_frequency': empirical_frequency,  # o_hat_b
                'calibration_gap': calibration_gap
            })
        else:
            bin_stats.append({
                'bin_idx': bin_idx,
                'bin_range': f'[{bins[bin_idx]:.2f}, {bins[bin_idx+1]:.2f}]',
                'count': 0,
                'avg_predicted_prob': np.nan,
                'empirical_frequency': np.nan,
                'calibration_gap': np.nan
            })
    
    return {
        'ECE': ece,
        'total_samples': total_samples,
        'num_bins': num_bins,
        'bin_stats': bin_stats
    }


def calculate_calibration_by_category(df: pd.DataFrame, file_label: str, num_bins: int = 10) -> Dict[str, Dict]:
    """
    Calculate calibration error separately for answered questions (idk_flag=0)
    and all questions combined.
    
    Returns:
        Dictionary with separate calibration metrics for each category
    """
    results = {}
    if file_label == "A_Baseline":
        df = df[df['idk_flag'] == 0].copy()
        results['overall'] = calculate_calibration_error(df, file_label, num_bins)
        return results

    # Split by IDK flag
    df_answered = df[df['idk_flag'] == 0].copy()
    
    # Calculate for answered questions
    if len(df_answered) > 0:
        results['answered'] = calculate_calibration_error(df_answered, file_label, num_bins)
    else:
        results['answered'] = None
    
    # Calculate for all combined
    results['overall'] = calculate_calibration_error(df, file_label, num_bins)
    
    return results


def print_calibration_report(results: Dict[str, Dict], label: str):
    """
    Pretty print calibration error results.
    
    Args:
        results: Dictionary returned by calculate_calibration_by_category
        label: Label for the experiment (e.g., 'Experiment A')
    """
    
    print(f"\n{'='*80}")
    print(f"📏 CALIBRATION REPORT: {label}")
    print(f"{'='*80}")
    
    for category, metrics in results.items():
        if metrics is None:
            print(f"\n❌ {category.upper()}: No data available")
            continue
        
        print(f"\n📊 {category.upper()}")
        print(f"   Expected Calibration Error (ECE): {metrics['ECE']:.4f}")
        print(f"   Total samples: {metrics['total_samples']}")
        print(f"   Number of bins: {metrics['num_bins']}")
        
        print(f"\n   Per-bin breakdown:")
        print(f"   {'Bin':<15} {'Count':<8} {'P_hat':<12} {'O_hat':<12} {'Cal Gap':<12}")
        print(f"   {'-'*60}")
        
        for bin_stat in metrics['bin_stats']:
            if bin_stat['count'] > 0:
                print(f"   {bin_stat['bin_range']:<15} "
                      f"{bin_stat['count']:<8} "
                      f"{bin_stat['avg_predicted_prob']:<12.4f} "
                      f"{bin_stat['empirical_frequency']:<12.4f} "
                      f"{bin_stat['calibration_gap']:<12.4f}")



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

    pattern = "outputs/popqa_results_*.csv"
    
    
    file = get_latest_file(pattern)
    
     # Load data

    df = pd.read_csv(file)
    

    print(f"📊 Loaded data files:")
    print(f"  no reward: {file}")

    summarize (df, "no reward")



    brier_score_B_norm, margin_B_norm = calculate_brier_score_with_ci(df)
    brier_score_B_norm_answered, margin_B_norm_answered = calculate_brier_score_with_ci(df, answered_only=True)
    print(f"Brier Score for Experiment B (Normalized overall): {brier_score_B_norm:.4f} ± {margin_B_norm:.4f}")
    print(f"Brier Score for Experiment B (Normalized answered only): {brier_score_B_norm_answered:.4f} ± {margin_B_norm_answered:.4f}")
    

    results_B_norm = calculate_calibration_by_category(df, 'no rewards', num_bins=10)
    print_calibration_report(results_B_norm, "confidence only")

    plot_idk_confidence_separate(df, 'confidence only')
   
    
    plot_confidence_by_correctness(df, 'confidence only')
    
    
if __name__ == "__main__":
    main()