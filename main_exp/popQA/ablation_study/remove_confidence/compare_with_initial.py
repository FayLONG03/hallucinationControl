import pandas as pd
from scipy.stats import binom

def calculate_ci_pm(successes, trials, confidence=0.95):
    """Calculate confidence interval margin (plus/minus)"""
    if trials == 0:
        return 0
    p_hat = successes / trials
    margin = binom.ppf(1 - (1 - confidence) / 2, trials, p_hat) / trials - p_hat
    return margin

# Load the two CSV files
ablation_file = '/Users/binzeli/Desktop/hallucination/Corrigibility-for-Language-Model-Hallucination-Control/experiment_1/ablation_study/remove_confidence/gpt5_results/popqa_B_batch_results_+1_-1_+0.4_20260125_213103.csv'
initial_file = '/Users/binzeli/Desktop/hallucination/Corrigibility-for-Language-Model-Hallucination-Control/experiment_1/initial_exp/gpt5_all_results/popqa_B_results_+1_-1_+0.4_20260109_122206.csv'

ablation_df = pd.read_csv(ablation_file)
initial_df = pd.read_csv(initial_file)

print("=" * 80)
print("COMPARISON: Ablation Study vs Initial Experiment")
print("=" * 80)

# Step 1: Find rows in ablation study where idk_flag = 1
ablation_idk = ablation_df[ablation_df['idk_flag'] == 1]
num_idk_ablation = len(ablation_idk)

print(f"\n1. Number of data points in ablation study with idk_flag = 1: {num_idk_ablation}")
print(f"   (Out of {len(ablation_df)} total data points)")
print(f"   Percentage: {num_idk_ablation / len(ablation_df) * 100:.2f}%")

# Step 2: Get the dataset_ids where ablation has idk_flag = 1
ablation_idk_dataset_ids = set(ablation_idk['dataset_id'].values)

# Step 3: Filter initial_df for those dataset_ids where idk_flag = 0 and correct = 1
initial_filtered = initial_df[
    (initial_df['dataset_id'].isin(ablation_idk_dataset_ids)) &
    (initial_df['idk_flag'] == 0) &
    (initial_df['correct'] == 1)
]

num_correct_in_initial = len(initial_filtered)

# Calculate average first_confidence for those rows
avg_first_confidence = initial_filtered['first_confidence'].mean()

print(f"\n2. For those {num_idk_ablation} dataset_ids from ablation study (idk_flag=1):")
print(f"   Number with idk_flag = 0 AND correct = 1 in initial experiment: {num_correct_in_initial}")
print(f"   Percentage: {num_correct_in_initial / num_idk_ablation * 100:.2f}%")
print(f"   Average first_confidence (for idk=0, correct=1): {avg_first_confidence:.4f}")


# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total ablation study samples: {len(ablation_df)}")
print(f"Ablation study with idk_flag=1: {num_idk_ablation} ({num_idk_ablation/len(ablation_df)*100:.2f}%)")
print(f"Of those, correct in initial (idk_flag=0, correct=1): {num_correct_in_initial} ({num_correct_in_initial/num_idk_ablation*100:.2f}%)")
print(f"This means: {num_correct_in_initial} questions that were answered correctly in the initial")
print(f"            experiment are now saying 'I don't know' in the ablation study.")


print(f"\n3. Metrics on initial_exp for those dataset_ids:")

# ===== NEW ANALYSIS: For questions where ablation has idk_flag=0 (answered) =====
print("\n" + "=" * 80)
print("ANALYSIS FOR QUESTIONS WHERE ABLATION STUDY ANSWERED (idk_flag=0)")
print("=" * 80)

# Step 5: Find rows in ablation study where idk_flag = 0
ablation_answered = ablation_df[ablation_df['idk_flag'] == 0]
num_answered_ablation = len(ablation_answered)

print(f"\n1. Number of data points in ablation study with idk_flag = 0: {num_answered_ablation}")
print(f"   (Out of {len(ablation_df)} total data points)")
print(f"   Percentage: {num_answered_ablation / len(ablation_df) * 100:.2f}%")

# Step 6: Get the dataset_ids where ablation has idk_flag = 0
ablation_answered_dataset_ids = set(ablation_answered['dataset_id'].values)

# Step 7: Calculate metrics for ablation file (answered questions only)
ablation_subset = ablation_df[ablation_df['dataset_id'].isin(ablation_answered_dataset_ids)]

acc_ablation = ablation_subset["correct"].mean()
answered_ablation = ablation_subset["idk_flag"] == 0
far_e1_ablation = ablation_subset.loc[answered_ablation, "false_answer_flag"].sum() / answered_ablation.sum() if answered_ablation.sum() > 0 else 0
far_e2_ablation = 1 - acc_ablation

far_e1_margin_ablation = calculate_ci_pm(ablation_subset.loc[answered_ablation, "false_answer_flag"].sum(), answered_ablation.sum())
far_e2_margin_ablation = calculate_ci_pm(len(ablation_subset) - ablation_subset["correct"].sum(), len(ablation_subset))

print(f"\n2. Metrics on ablation_study for those {num_answered_ablation} dataset_ids:")
print(f"   Accuracy: {acc_ablation:.4f}")
print(f"   False-Answer Rate (E1): {far_e1_ablation:.4f} ± {far_e1_margin_ablation:.4f}")
print(f"   False-Answer Rate (E2): {far_e2_ablation:.4f} ± {far_e2_margin_ablation:.4f}")
print(f"   Answered (idk_flag=0): {answered_ablation.sum()} out of {len(ablation_subset)}")
print(f"   Abstained (idk_flag=1): {len(ablation_subset) - answered_ablation.sum()} out of {len(ablation_subset)}")

# Step 8: Calculate metrics for initial file (same dataset_ids)
initial_subset_answered = initial_df[initial_df['dataset_id'].isin(ablation_answered_dataset_ids)]

acc_initial = initial_subset_answered["correct"].mean()
answered_initial = initial_subset_answered["idk_flag"] == 0
far_e1_initial = initial_subset_answered.loc[answered_initial, "false_answer_flag"].sum() / answered_initial.sum() if answered_initial.sum() > 0 else 0
far_e2_initial = 1 - acc_initial

far_e1_margin_initial = calculate_ci_pm(initial_subset_answered.loc[answered_initial, "false_answer_flag"].sum(), answered_initial.sum())
far_e2_margin_initial = calculate_ci_pm(len(initial_subset_answered) - initial_subset_answered["correct"].sum(), len(initial_subset_answered))

print(f"\n3. Metrics on initial_exp for those {num_answered_ablation} dataset_ids:")
print(f"   Accuracy: {acc_initial:.4f}")
print(f"   False-Answer Rate (E1): {far_e1_initial:.4f} ± {far_e1_margin_initial:.4f}")
print(f"   False-Answer Rate (E2): {far_e2_initial:.4f} ± {far_e2_margin_initial:.4f}")
print(f"   Answered (idk_flag=0): {answered_initial.sum()} out of {len(initial_subset_answered)}")
print(f"   Abstained (idk_flag=1): {len(initial_subset_answered) - answered_initial.sum()} out of {len(initial_subset_answered)}")
