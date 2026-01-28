# Post-Error Slowing Analysis
# Two approaches: (1) Difference score model, (2) Full factorial model

import pandas as pd
import numpy as np
from scipy import stats

# For mixed effects models
# Option A: pymer4 (R's lme4 via rpy2) - most flexible
# Option B: statsmodels MixedLM - pure Python but more limited
# I'll show both syntaxes

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(raw_data):
    """
    Prepare data with previous trial information and error type classification.
    """
    df = raw_data.copy()
    
    # Previous trial info (shift within subject)
    df['prev_acc'] = df.groupby('subject_ID')['acc'].shift(1)
    df['prev_cong'] = df.groupby('subject_ID')['congruency'].shift(1)
    df['prev_sw'] = df.groupby('subject_ID')['switchType'].shift(1)
    
    # Boolean for post-error
    df['is_post_error'] = df['prev_acc'] == 0
    df['is_post_correct'] = df['prev_acc'] == 1

    df['total_subject_errors'] = df.groupby('subject_ID')['is_post_error'].transform('sum')
    df = df[df['total_subject_errors'] >= 25]
    
    # Classify previous ERROR TYPE (only meaningful when prev_acc == 0)
    # iR (incongruent-repeat) = stability failure
    # cS (congruent-switch) = flexibility failure
    # We can also include iS and cR if you want all four
    conditions = [
        (df['prev_cong'] == 'i') & (df['prev_sw'] == 'r'),  # iR - stability failure
        (df['prev_cong'] == 'c') & (df['prev_sw'] == 's'),  # cS - flexibility failure
        (df['prev_cong'] == 'i') & (df['prev_sw'] == 's'),  # iS
        (df['prev_cong'] == 'c') & (df['prev_sw'] == 'r'),  # cR
    ]
    choices = ['iR', 'cS', 'iS', 'cR']
    df['prev_trial_type'] = np.select(conditions, choices, default=None)
    
    # For the binary stability vs flexibility comparison
    df['prev_error_type'] = np.where(
        df['prev_trial_type'].isin(['iR']), 'stability_failure',
        np.where(df['prev_trial_type'].isin(['cS']), 'flexibility_failure', None)
    )
    
    # Current trial conditions (for clarity)
    df['curr_cong'] = df['congruency']
    df['curr_sw'] = df['switchType']
    
    # Filter to valid RTs and correct current trials (typical for RT analyses)
    df = df[(df['RT'] > 0) & (df['acc'] == 1)].copy()
    
    # Drop rows with missing previous trial info (first trial of each subject)
    df = df.dropna(subset=['prev_acc', 'prev_cong', 'prev_sw'])
    
    return df


# =============================================================================
# APPROACH 1: DIFFERENCE SCORE MODEL
# =============================================================================

def compute_slowing_scores(df):
    """
    Compute post-error slowing (post-error RT - post-correct RT) 
    for each subject × previous error type × current congruency × current switch type cell.
    """
    # We need to compute mean RT for post-error and post-correct trials separately,
    # then take the difference
    
    # Group by subject, previous accuracy, previous error type, and current trial conditions
    # But previous error type is only defined for errors, so we need to be careful
    
    # For post-ERROR trials: group by prev_error_type
    post_error = df[df['is_post_error']].copy()
    post_error_means = post_error.groupby(
        ['subject_ID', 'prev_error_type', 'curr_cong', 'curr_sw']
    )['RT'].agg(['mean', 'count']).reset_index()
    post_error_means.columns = ['subject_ID', 'prev_error_type', 'curr_cong', 'curr_sw', 
                                 'RT_post_error', 'n_post_error']
    
    # For post-CORRECT trials: we need a matching structure
    # The question is: what "prev_error_type" do we assign to post-correct trials?
    # Option: Match by the same prev_trial_type (iR, cS, etc.) but where prev trial was correct
    post_correct = df[df['is_post_correct']].copy()
    post_correct_means = post_correct.groupby(
        ['subject_ID', 'prev_trial_type', 'curr_cong', 'curr_sw']
    )['RT'].agg(['mean', 'count']).reset_index()
    post_correct_means.columns = ['subject_ID', 'prev_trial_type', 'curr_cong', 'curr_sw',
                                   'RT_post_correct', 'n_post_correct']
    
    # Map prev_trial_type to error_type for merging
    post_correct_means['prev_error_type'] = post_correct_means['prev_trial_type'].map({
        'iR': 'stability_failure',
        'cS': 'flexibility_failure'
    })
    post_correct_means = post_correct_means.dropna(subset=['prev_error_type'])
    
    # Merge post-error and post-correct
    slowing_df = pd.merge(
        post_error_means,
        post_correct_means[['subject_ID', 'prev_error_type', 'curr_cong', 'curr_sw', 
                            'RT_post_correct', 'n_post_correct']],
        on=['subject_ID', 'prev_error_type', 'curr_cong', 'curr_sw'],
        how='inner'
    )
    
    # Compute slowing
    slowing_df['post_error_slowing'] = slowing_df['RT_post_error'] - slowing_df['RT_post_correct']
    
    return slowing_df


def run_difference_score_model_statsmodels(slowing_df):
    """
    Run mixed model on post-error slowing scores using statsmodels.
    
    PostErrorSlowing ~ PrevErrorType * CurrCong * CurrSw + (1 | Subject)
    
    Note: statsmodels MixedLM has limited support for crossed random effects
    and complex interactions. For full flexibility, use pymer4 or R.
    """
    import statsmodels.formula.api as smf
    
    # Rename for cleaner formula
    model_df = slowing_df.copy()
    model_df['error_type'] = model_df['prev_error_type']
    model_df['cong'] = model_df['curr_cong']
    model_df['sw'] = model_df['curr_sw']
    model_df['slowing'] = model_df['post_error_slowing']
    model_df['subj'] = model_df['subject_ID']
    
    # Filter to cells with enough trials
    model_df = model_df[(model_df['n_post_error'] >= 3) & (model_df['n_post_correct'] >= 3)]
    
    # Statsmodels formula
    # Note: statsmodels uses C() for categorical variables
    formula = 'slowing ~ C(error_type) * C(cong) * C(sw)'
    
    model = smf.mixedlm(formula, model_df, groups=model_df['subj'])
    result = model.fit()
    
    return result


def run_difference_score_model_pymer4(slowing_df):
    """
    Run mixed model using pymer4 (requires R and lme4 installed).
    
    PostErrorSlowing ~ PrevErrorType * CurrCong * CurrSw + (1 | Subject)
    """
    from pymer4.models import Lmer
    
    model_df = slowing_df.copy()
    model_df['error_type'] = model_df['prev_error_type']
    model_df['cong'] = model_df['curr_cong']
    model_df['sw'] = model_df['curr_sw']
    model_df['slowing'] = model_df['post_error_slowing']
    model_df['subj'] = model_df['subject_ID']
    
    # Filter to cells with enough trials
    model_df = model_df[(model_df['n_post_error'] >= 3) & (model_df['n_post_correct'] >= 3)]
    
    # pymer4 formula (R-style)
    formula = 'slowing ~ error_type * cong * sw + (1 | subj)'
    
    model = Lmer(formula, data=model_df)
    model.fit()
    
    return model


# =============================================================================
# APPROACH 2: FULL FACTORIAL MODEL (all trials)
# =============================================================================

def run_full_factorial_model_statsmodels(df):
    """
    Include all trials and model RT with previous accuracy as a factor.
    
    RT ~ PrevAcc * PrevErrorType * CurrCong * CurrSw + (1 | Subject)
    
    The key interactions:
    - PrevAcc:PrevErrorType:CurrCong = does congruency effect on slowing differ by error type?
    - PrevAcc:PrevErrorType:CurrSw = does switch cost on slowing differ by error type?
    """
    import statsmodels.formula.api as smf
    
    # Filter to trials with defined prev_error_type (iR or cS only)
    model_df = df[df['prev_error_type'].isin(['stability_failure', 'flexibility_failure'])].copy()
    
    # Create clean variable names
    model_df['prev_acc'] = model_df['prev_acc'].map({0: 'error', 1: 'correct'})
    model_df['error_type'] = model_df['prev_error_type']
    model_df['cong'] = model_df['curr_cong']
    model_df['sw'] = model_df['curr_sw']
    model_df['subj'] = model_df['subject_ID']
    
    # Formula with all interactions
    # Note: This is a big model - you might want to simplify
    formula = 'RT ~ C(prev_acc) * C(error_type) * C(cong) * C(sw)'
    
    model = smf.mixedlm(formula, model_df, groups=model_df['subj'])
    result = model.fit()
    
    return result


def run_full_factorial_model_pymer4(df, include_block_covariates=False):
    """
    Full factorial model using pymer4.
    
    RT ~ PrevAcc * PrevTrialType * CurrCong * CurrSw + IncProp + SwProp + (1 | Subject)
    
    If you have block proportion variables, set include_block_covariates=True.
    """
    from pymer4.models import Lmer
    
    # Filter to trials with defined prev_error_type
    model_df = df[df['prev_error_type'].isin(['stability_failure', 'flexibility_failure'])].copy()
    
    model_df['prev_acc'] = model_df['prev_acc'].map({0: 'error', 1: 'correct'})
    model_df['error_type'] = model_df['prev_error_type']
    model_df['cong'] = model_df['curr_cong']
    model_df['sw'] = model_df['curr_sw']
    model_df['subj'] = model_df['subject_ID']
    
    if include_block_covariates:
        # Assuming these columns exist in your data
        formula = 'RT ~ prev_acc * error_type * cong * sw + inc_prop + sw_prop + (1 | subj)'
    else:
        formula = 'RT ~ prev_acc * error_type * cong * sw + (1 | subj)'
    
    model = Lmer(formula, data=model_df)
    model.fit()
    
    return model


# =============================================================================
# DESCRIPTIVE STATISTICS & VISUALIZATION
# =============================================================================

def compute_descriptives(df):
    """
    Compute descriptive statistics for post-error slowing by condition.
    """
    # Filter to defined error types
    analysis_df = df[df['prev_error_type'].isin(['stability_failure', 'flexibility_failure'])].copy()
    
    # Compute means by prev_acc, error_type, curr_cong, curr_sw
    desc = analysis_df.groupby(
        ['prev_acc', 'prev_error_type', 'curr_cong', 'curr_sw']
    )['RT'].agg(['mean', 'std', 'count']).reset_index()
    
    # Pivot to compare post-error vs post-correct
    desc['prev_acc_label'] = desc['prev_acc'].map({0: 'post_error', 1: 'post_correct'})
    
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS: Mean RT by Condition")
    print("="*80)
    print(desc.to_string(index=False))
    
    # Compute slowing
    post_error = desc[desc['prev_acc'] == 0].copy()
    post_correct = desc[desc['prev_acc'] == 1].copy()
    
    merged = pd.merge(
        post_error[['prev_error_type', 'curr_cong', 'curr_sw', 'mean']],
        post_correct[['prev_error_type', 'curr_cong', 'curr_sw', 'mean']],
        on=['prev_error_type', 'curr_cong', 'curr_sw'],
        suffixes=('_error', '_correct')
    )
    merged['slowing'] = merged['mean_error'] - merged['mean_correct']
    
    print("\n" + "="*80)
    print("POST-ERROR SLOWING BY CONDITION")
    print("="*80)
    print(merged.to_string(index=False))
    
    return desc, merged


def plot_slowing_by_condition(slowing_df):
    """
    Plot post-error slowing as a function of previous error type and current trial conditions.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Slowing by error type and congruency
    ax1 = axes[0]
    for error_type in ['stability_failure', 'flexibility_failure']:
        subset = slowing_df[slowing_df['prev_error_type'] == error_type]
        means = subset.groupby('curr_cong')['post_error_slowing'].mean()
        sems = subset.groupby('curr_cong')['post_error_slowing'].sem()
        
        x = [0, 1] if error_type == 'stability_failure' else [0.2, 1.2]
        ax1.bar(x, means.values, width=0.2, yerr=sems.values, 
                label=error_type, alpha=0.7)
    
    ax1.set_xticks([0.1, 1.1])
    ax1.set_xticklabels(['Congruent', 'Incongruent'])
    ax1.set_ylabel('Post-Error Slowing (ms)')
    ax1.set_title('Slowing by Error Type × Current Congruency')
    ax1.legend()
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Slowing by error type and switch type
    ax2 = axes[1]
    for error_type in ['stability_failure', 'flexibility_failure']:
        subset = slowing_df[slowing_df['prev_error_type'] == error_type]
        means = subset.groupby('curr_sw')['post_error_slowing'].mean()
        sems = subset.groupby('curr_sw')['post_error_slowing'].sem()
        
        x = [0, 1] if error_type == 'stability_failure' else [0.2, 1.2]
        ax2.bar(x, means.values, width=0.2, yerr=sems.values,
                label=error_type, alpha=0.7)
    
    ax2.set_xticks([0.1, 1.1])
    ax2.set_xticklabels(['Repeat', 'Switch'])
    ax2.set_ylabel('Post-Error Slowing (ms)')
    ax2.set_title('Slowing by Error Type × Current Switch Type')
    ax2.legend()
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('post_error_slowing_plots.png', dpi=150)
    plt.show()
    
    return fig


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def main(data_path):
    """
    Run the full analysis pipeline.
    """
    # Load data
    print("Loading data...")
    raw_data = pd.read_csv(data_path)
    
    # Prepare data
    print("Preparing data...")
    df = prepare_data(raw_data)
    
    # Descriptive statistics
    desc, slowing_summary = compute_descriptives(df)
    
    # Compute slowing scores for difference model
    print("\nComputing post-error slowing scores...")
    slowing_df = compute_slowing_scores(df)
    
    print(f"\nSlowing data shape: {slowing_df.shape}")
    print(f"Subjects with data: {slowing_df['subject_ID'].nunique()}")
    
    # Run models
    print("\n" + "="*80)
    print("APPROACH 1: DIFFERENCE SCORE MODEL (statsmodels)")
    print("="*80)
    try:
        result1 = run_difference_score_model_statsmodels(slowing_df)
        print(result1.summary())
    except Exception as e:
        print(f"Error running statsmodels: {e}")
    
    print("\n" + "="*80)
    print("APPROACH 2: FULL FACTORIAL MODEL (statsmodels)")
    print("="*80)
    try:
        result2 = run_full_factorial_model_statsmodels(df)
        print(result2.summary())
    except Exception as e:
        print(f"Error running statsmodels: {e}")
    
    # Try pymer4 if available
    print("\n" + "="*80)
    print("PYMER4 MODELS (if available)")
    print("="*80)
    try:
        from pymer4.models import Lmer
        print("\nDifference score model (pymer4):")
        model_pymer = run_difference_score_model_pymer4(slowing_df)
        print(model_pymer.summary())
        
        print("\nFull factorial model (pymer4):")
        model_pymer2 = run_full_factorial_model_pymer4(df)
        print(model_pymer2.summary())
    except ImportError:
        print("pymer4 not installed. Install with: pip install pymer4")
    except Exception as e:
        print(f"Error with pymer4: {e}")
    
    # Plot
    print("\nGenerating plots...")
    try:
        plot_slowing_by_condition(slowing_df)
    except Exception as e:
        print(f"Error plotting: {e}")
    
    return df, slowing_df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Update this path to your data
    data_path = '/Users/erinburns/Library/CloudStorage/Box-Box/CoganLab/D_Data/GlobalLocal/combinedData.csv'
    
    # If running interactively, you can also do:
    # raw_data = pd.read_csv(data_path)
    # df = prepare_data(raw_data)
    # slowing_df = compute_slowing_scores(df)
    # result = run_difference_score_model_statsmodels(slowing_df)
    
    df, slowing_df = main(data_path)
