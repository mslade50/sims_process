"""
Archetype Diagnostics Script
Analyzes finish position distributions by player archetype to validate
whether SG profile characteristics translate to expected distribution shapes.

Expected patterns:
- Bombers (high OTT variance): More kurtotic, bimodal finish distributions
- Scramblers (high ARG/Putt variance): Mean-reverting, platykurtic distributions
- Ball Strikers (high OTT+APP): Right-skewed (upside)
- Putters (high Putt variance): Compressed distributions due to regression coefficients
- Consistent (low variance): More Gaussian finish distributions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Input files
DISTS_FILE = "this_week_dists_adjusted.csv"
RANK_PROBS_FILE = None  # Will search for rank_probs_updated_*.parquet
FINISH_EQUITY_FILE = None  # Will search for finish_equity_*.csv

# Archetype thresholds (percentiles within field)
HIGH_THRESHOLD = 0.70  # Top 30% = "high"
LOW_THRESHOLD = 0.30   # Bottom 30% = "low"

# Output
OUTPUT_DIR = Path(".")
TOURNEY = None  # Will be detected from files


def find_tournament_files():
    """Find the most recent tournament output files."""
    global RANK_PROBS_FILE, FINISH_EQUITY_FILE, TOURNEY

    # Find rank_probs parquet
    rank_files = list(Path(".").glob("rank_probs_updated_*.parquet"))
    if rank_files:
        RANK_PROBS_FILE = str(sorted(rank_files, key=lambda x: x.stat().st_mtime)[-1])
        # Extract tourney name
        TOURNEY = RANK_PROBS_FILE.replace("rank_probs_updated_", "").replace(".parquet", "")
        print(f"[info] Found rank probs: {RANK_PROBS_FILE}")

    # Find finish_equity csv
    equity_files = list(Path(".").glob("finish_equity_*.csv"))
    if equity_files:
        FINISH_EQUITY_FILE = str(sorted(equity_files, key=lambda x: x.stat().st_mtime)[-1])
        print(f"[info] Found finish equity: {FINISH_EQUITY_FILE}")

    return RANK_PROBS_FILE is not None


def load_player_distributions():
    """Load and pivot the player SG category distributions."""
    dists = pd.read_csv(DISTS_FILE)
    dists['player_name'] = dists['player_name'].astype(str).str.lower().str.strip()

    # Pivot to get mean and std per category per player
    means = dists.pivot(index='player_name', columns='category_clean', values='mean_adj')
    stds = dists.pivot(index='player_name', columns='category_clean', values='std_adj')

    # Get df_t for tail analysis (lower df = fatter tails)
    df_t = dists.pivot(index='player_name', columns='category_clean', values='df_t')

    return means, stds, df_t


def load_rank_probabilities():
    """Load the rank probability distributions from simulation."""
    if RANK_PROBS_FILE is None:
        raise FileNotFoundError("No rank_probs file found")

    rank_probs = pd.read_parquet(RANK_PROBS_FILE)
    rank_probs['player_name'] = rank_probs['player_name'].astype(str).str.lower().str.strip()

    # Rename prob column if needed
    if 'prob_u' in rank_probs.columns:
        rank_probs = rank_probs.rename(columns={'prob_u': 'prob'})

    return rank_probs


def load_finish_equity():
    """Load finish equity summary."""
    if FINISH_EQUITY_FILE is None:
        return None

    df = pd.read_csv(FINISH_EQUITY_FILE)
    df['player_name'] = df['player_name'].astype(str).str.lower().str.strip()
    return df


def classify_archetypes(means, stds, df_t):
    """
    Classify players into archetypes based on their SG profiles.

    Returns DataFrame with archetype classifications.
    """
    players = means.index.tolist()

    # Calculate field percentiles for each metric
    ott_mean_pct = means['sg_ott'].rank(pct=True)
    ott_std_pct = stds['sg_ott'].rank(pct=True)
    app_mean_pct = means['sg_app'].rank(pct=True)
    app_std_pct = stds['sg_app'].rank(pct=True)
    arg_mean_pct = means['sg_arg'].rank(pct=True)
    arg_std_pct = stds['sg_arg'].rank(pct=True)
    putt_mean_pct = means['sg_putt'].rank(pct=True)
    putt_std_pct = stds['sg_putt'].rank(pct=True)

    # Overall variance (sum of stds)
    total_std = stds.sum(axis=1)
    total_std_pct = total_std.rank(pct=True)

    # Ball striking vs short game ratio
    ball_striking_mean = means['sg_ott'] + means['sg_app']
    short_game_mean = means['sg_arg'] + means['sg_putt']
    bs_sg_ratio = ball_striking_mean - short_game_mean
    bs_sg_ratio_pct = bs_sg_ratio.rank(pct=True)

    # Variance concentration (where does variance come from?)
    ott_var_share = stds['sg_ott'] / total_std
    putt_var_share = stds['sg_putt'] / total_std
    arg_var_share = stds['sg_arg'] / total_std
    app_var_share = stds['sg_app'] / total_std

    # Build classification DataFrame
    arch = pd.DataFrame(index=players)

    # Raw metrics
    arch['ott_mean'] = means['sg_ott']
    arch['ott_std'] = stds['sg_ott']
    arch['app_mean'] = means['sg_app']
    arch['app_std'] = stds['sg_app']
    arch['arg_mean'] = means['sg_arg']
    arch['arg_std'] = stds['sg_arg']
    arch['putt_mean'] = means['sg_putt']
    arch['putt_std'] = stds['sg_putt']
    arch['total_std'] = total_std
    arch['total_mean'] = means.sum(axis=1)

    # Percentiles
    arch['ott_std_pct'] = ott_std_pct
    arch['putt_std_pct'] = putt_std_pct
    arch['arg_std_pct'] = arg_std_pct
    arch['total_std_pct'] = total_std_pct
    arch['bs_sg_ratio'] = bs_sg_ratio
    arch['bs_sg_ratio_pct'] = bs_sg_ratio_pct

    # Variance shares
    arch['ott_var_share'] = ott_var_share
    arch['putt_var_share'] = putt_var_share
    arch['arg_var_share'] = arg_var_share

    # Primary archetype classification
    def assign_archetype(row):
        archetypes = []

        # Bomber: high OTT variance
        if row['ott_std_pct'] >= HIGH_THRESHOLD:
            archetypes.append('bomber')

        # Scrambler: high ARG + Putt variance relative to ball striking
        if (row['arg_std_pct'] >= HIGH_THRESHOLD or row['putt_std_pct'] >= HIGH_THRESHOLD) and \
           row['ott_std_pct'] < HIGH_THRESHOLD:
            archetypes.append('scrambler')

        # Putter: specifically high putt variance
        if row['putt_std_pct'] >= HIGH_THRESHOLD:
            archetypes.append('high_putt_var')

        # Ball striker: good at OTT+APP, weaker short game
        if row['bs_sg_ratio_pct'] >= HIGH_THRESHOLD:
            archetypes.append('ball_striker')

        # Short game specialist: opposite
        if row['bs_sg_ratio_pct'] <= LOW_THRESHOLD:
            archetypes.append('short_game')

        # High variance overall
        if row['total_std_pct'] >= HIGH_THRESHOLD:
            archetypes.append('high_variance')

        # Low variance (consistent)
        if row['total_std_pct'] <= LOW_THRESHOLD:
            archetypes.append('consistent')

        return '|'.join(archetypes) if archetypes else 'neutral'

    arch['archetypes'] = arch.apply(assign_archetype, axis=1)

    # Simplified primary archetype (for grouping)
    def primary_archetype(row):
        if row['ott_std_pct'] >= HIGH_THRESHOLD:
            return 'bomber'
        elif row['putt_std_pct'] >= HIGH_THRESHOLD:
            return 'high_putt_var'
        elif row['total_std_pct'] <= LOW_THRESHOLD:
            return 'consistent'
        elif row['total_std_pct'] >= HIGH_THRESHOLD:
            return 'high_variance'
        elif row['bs_sg_ratio_pct'] >= HIGH_THRESHOLD:
            return 'ball_striker'
        elif row['bs_sg_ratio_pct'] <= LOW_THRESHOLD:
            return 'short_game'
        else:
            return 'neutral'

    arch['primary_archetype'] = arch.apply(primary_archetype, axis=1)

    return arch


def compute_distribution_metrics(rank_probs, finish_equity):
    """
    Compute distribution shape metrics for each player's finish distribution.

    Returns DataFrame with:
    - kurtosis (excess): positive = fat tails, negative = thin tails
    - skewness: positive = right tail, negative = left tail
    - win_prob, top5, top10, top20 probabilities
    - mc_prob: missed cut probability (approximated as positions > 65)
    - gini: concentration of probability mass
    - bimodality_coefficient: higher = more bimodal
    """
    players = rank_probs['player_name'].unique()
    metrics = []

    for player in players:
        player_dist = rank_probs[rank_probs['player_name'] == player].copy()
        player_dist = player_dist.sort_values('rank')

        # Get probabilities and positions
        positions = player_dist['rank'].values
        probs = player_dist['prob'].values

        # Normalize (should sum to 1, but just in case)
        probs = probs / probs.sum()

        # Weighted statistics treating position as the random variable
        mean_pos = np.sum(positions * probs)
        var_pos = np.sum(((positions - mean_pos) ** 2) * probs)
        std_pos = np.sqrt(var_pos)

        # Skewness
        if std_pos > 0:
            skew_pos = np.sum(((positions - mean_pos) ** 3) * probs) / (std_pos ** 3)
        else:
            skew_pos = 0

        # Kurtosis (excess)
        if std_pos > 0:
            kurt_pos = np.sum(((positions - mean_pos) ** 4) * probs) / (std_pos ** 4) - 3
        else:
            kurt_pos = 0

        # Tail probabilities
        win_prob = probs[positions == 1].sum() if 1 in positions else 0
        top5_prob = probs[positions <= 5].sum()
        top10_prob = probs[positions <= 10].sum()
        top20_prob = probs[positions <= 20].sum()

        # Missed cut approximation (positions beyond ~65)
        n_players = len(positions)
        mc_threshold = min(65, int(n_players * 0.5))  # Roughly half field makes cut
        mc_prob = probs[positions > mc_threshold].sum()

        # Probability in extremes (top 10 + bottom 30%)
        bottom_threshold = int(n_players * 0.7)
        extreme_prob = top10_prob + probs[positions > bottom_threshold].sum()

        # Bimodality coefficient: (skew^2 + 1) / (kurtosis + 3)
        # Higher values suggest bimodality
        bimodality = (skew_pos ** 2 + 1) / (kurt_pos + 3) if (kurt_pos + 3) != 0 else 0

        # Gini coefficient (concentration)
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_probs) / (n * np.sum(sorted_probs))) - (n + 1) / n

        # Win/Top10 ratio (how concentrated is upside at very top?)
        win_top10_ratio = win_prob / top10_prob if top10_prob > 0 else 0

        metrics.append({
            'player_name': player,
            'mean_position': mean_pos,
            'std_position': std_pos,
            'skewness': skew_pos,
            'kurtosis': kurt_pos,
            'win_prob': win_prob,
            'top5_prob': top5_prob,
            'top10_prob': top10_prob,
            'top20_prob': top20_prob,
            'mc_prob': mc_prob,
            'extreme_prob': extreme_prob,
            'bimodality': bimodality,
            'gini': gini,
            'win_top10_ratio': win_top10_ratio
        })

    return pd.DataFrame(metrics)


def analyze_by_archetype(arch_df, metrics_df):
    """
    Combine archetype classifications with distribution metrics
    and compute summary statistics by archetype.
    """
    # Merge
    combined = arch_df.reset_index().merge(
        metrics_df,
        left_on='index',
        right_on='player_name',
        how='inner'
    )

    # Group by primary archetype
    summary = combined.groupby('primary_archetype').agg({
        'kurtosis': ['mean', 'std', 'median'],
        'skewness': ['mean', 'std', 'median'],
        'std_position': ['mean', 'std', 'median'],
        'win_prob': ['mean', 'median'],
        'top10_prob': ['mean', 'median'],
        'mc_prob': ['mean', 'median'],
        'extreme_prob': ['mean', 'median'],
        'bimodality': ['mean', 'median'],
        'total_mean': ['mean'],  # Average skill level
        'player_name': 'count'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'player_name_count': 'n_players'})

    return combined, summary


def plot_archetype_distributions(combined, rank_probs):
    """Create visualizations comparing archetypes."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    archetypes = combined['primary_archetype'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(archetypes)))
    color_map = dict(zip(archetypes, colors))

    # 1. Kurtosis by archetype
    ax = axes[0, 0]
    arch_order = combined.groupby('primary_archetype')['kurtosis'].mean().sort_values().index
    combined.boxplot(column='kurtosis', by='primary_archetype', ax=ax, positions=range(len(arch_order)))
    ax.set_xticklabels(arch_order, rotation=45, ha='right')
    ax.set_title('Kurtosis by Archetype\n(Higher = Fatter Tails)')
    ax.set_xlabel('')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Gaussian')
    plt.suptitle('')

    # 2. Skewness by archetype
    ax = axes[0, 1]
    combined.boxplot(column='skewness', by='primary_archetype', ax=ax)
    ax.set_xticklabels(arch_order, rotation=45, ha='right')
    ax.set_title('Skewness by Archetype\n(Positive = Right Tail/Upside)')
    ax.set_xlabel('')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.suptitle('')

    # 3. Position StdDev by archetype
    ax = axes[0, 2]
    combined.boxplot(column='std_position', by='primary_archetype', ax=ax)
    ax.set_xticklabels(arch_order, rotation=45, ha='right')
    ax.set_title('Finish Position Std Dev by Archetype\n(Higher = More Variable)')
    ax.set_xlabel('')
    plt.suptitle('')

    # 4. Kurtosis vs OTT Variance
    ax = axes[1, 0]
    for arch in archetypes:
        mask = combined['primary_archetype'] == arch
        ax.scatter(combined.loc[mask, 'ott_std'],
                  combined.loc[mask, 'kurtosis'],
                  c=[color_map[arch]], label=arch, alpha=0.6)
    ax.set_xlabel('OTT Std Dev')
    ax.set_ylabel('Finish Distribution Kurtosis')
    ax.set_title('OTT Variance => Kurtosis\n(Expect: Higher OTT Var = Higher Kurtosis)')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    # 5. Kurtosis vs Putt Variance
    ax = axes[1, 1]
    for arch in archetypes:
        mask = combined['primary_archetype'] == arch
        ax.scatter(combined.loc[mask, 'putt_std'],
                  combined.loc[mask, 'kurtosis'],
                  c=[color_map[arch]], label=arch, alpha=0.6)
    ax.set_xlabel('Putt Std Dev')
    ax.set_ylabel('Finish Distribution Kurtosis')
    ax.set_title('Putt Variance => Kurtosis\n(Expect: Weaker relationship due to regression)')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    # 6. Example distributions for extreme archetypes
    ax = axes[1, 2]

    # Find most extreme bomber and most consistent player (with similar skill)
    skill_range = combined['total_mean'].quantile([0.4, 0.6])
    mid_skill = combined[(combined['total_mean'] >= skill_range[0.4]) &
                         (combined['total_mean'] <= skill_range[0.6])]

    if not mid_skill.empty:
        # Highest OTT variance in mid-skill
        if len(mid_skill[mid_skill['primary_archetype'] == 'bomber']) > 0:
            bomber = mid_skill[mid_skill['primary_archetype'] == 'bomber'].iloc[0]['player_name']
        else:
            bomber = mid_skill.nlargest(1, 'ott_std').iloc[0]['player_name']

        # Lowest total variance in mid-skill
        if len(mid_skill[mid_skill['primary_archetype'] == 'consistent']) > 0:
            consistent = mid_skill[mid_skill['primary_archetype'] == 'consistent'].iloc[0]['player_name']
        else:
            consistent = mid_skill.nsmallest(1, 'total_std').iloc[0]['player_name']

        # Plot their distributions
        for player, label, color in [(bomber, 'High OTT Var', 'red'),
                                      (consistent, 'Consistent', 'blue')]:
            player_dist = rank_probs[rank_probs['player_name'] == player]
            ax.plot(player_dist['rank'], player_dist['prob'] * 100,
                   label=f"{player.split(',')[0].title()} ({label})",
                   color=color, alpha=0.7)

        ax.set_xlabel('Finish Position')
        ax.set_ylabel('Probability (%)')
        ax.set_title('Example: High OTT Var vs Consistent\n(Similar Skill Level)')
        ax.legend()
        ax.set_xlim(0, 80)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'archetype_diagnostics_{TOURNEY}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ok] Saved archetype_diagnostics_{TOURNEY}.png")


def plot_correlation_matrix(combined):
    """Plot correlation between SG profile metrics and distribution metrics."""

    profile_cols = ['ott_std', 'app_std', 'arg_std', 'putt_std', 'total_std',
                    'ott_mean', 'total_mean', 'bs_sg_ratio']
    dist_cols = ['kurtosis', 'skewness', 'std_position', 'win_prob',
                 'top10_prob', 'mc_prob', 'bimodality']

    # Compute correlations
    corr_matrix = combined[profile_cols + dist_cols].corr()

    # Extract just profile => distribution correlations
    profile_to_dist = corr_matrix.loc[profile_cols, dist_cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(profile_to_dist.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(dist_cols)))
    ax.set_yticks(range(len(profile_cols)))
    ax.set_xticklabels(dist_cols, rotation=45, ha='right')
    ax.set_yticklabels(profile_cols)

    # Add correlation values
    for i in range(len(profile_cols)):
        for j in range(len(dist_cols)):
            val = profile_to_dist.iloc[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    ax.set_title('Profile Metrics => Distribution Metrics Correlations\n'
                 '(What profile characteristics drive distribution shape?)')
    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'profile_distribution_corr_{TOURNEY}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ok] Saved profile_distribution_corr_{TOURNEY}.png")

    return profile_to_dist


def print_summary_report(combined, summary, corr_matrix):
    """Print a text summary of findings."""

    print("\n" + "="*70)
    print("ARCHETYPE DIAGNOSTICS REPORT")
    print("="*70)

    print(f"\nTournament: {TOURNEY}")
    print(f"Players analyzed: {len(combined)}")

    print("\n" + "-"*70)
    print("ARCHETYPE DISTRIBUTION")
    print("-"*70)
    counts = combined['primary_archetype'].value_counts()
    for arch, count in counts.items():
        print(f"  {arch:20s}: {count:3d} players ({100*count/len(combined):5.1f}%)")

    print("\n" + "-"*70)
    print("KEY FINDINGS: Distribution Metrics by Archetype")
    print("-"*70)

    print("\n1. KURTOSIS (Fat Tails)")
    print("   Expectation: Bombers > Scramblers/Putters")
    kurt_by_arch = summary['kurtosis_mean'].sort_values(ascending=False)
    for arch, val in kurt_by_arch.items():
        indicator = "(fat tails)" if val > 0 else "(thin tails)"
        print(f"   {arch:20s}: {val:+.3f} {indicator}")

    print("\n2. SKEWNESS (Asymmetry)")
    print("   Positive = more upside, Negative = more downside risk")
    skew_by_arch = summary['skewness_mean'].sort_values(ascending=False)
    for arch, val in skew_by_arch.items():
        indicator = "=> More upside" if val > 0.1 else ("=> More downside" if val < -0.1 else "=> Symmetric")
        print(f"   {arch:20s}: {val:+.3f} {indicator}")

    print("\n3. POSITION VOLATILITY (Std Dev of Finish)")
    print("   Expectation: High variance profiles = higher position volatility")
    std_by_arch = summary['std_position_mean'].sort_values(ascending=False)
    for arch, val in std_by_arch.items():
        print(f"   {arch:20s}: {val:.1f} positions")

    print("\n" + "-"*70)
    print("KEY CORRELATIONS: Profile => Distribution")
    print("-"*70)

    print("\nOTT Variance correlations:")
    print(f"  => Kurtosis:  {corr_matrix.loc['ott_std', 'kurtosis']:+.3f} {'[OK] Expected' if corr_matrix.loc['ott_std', 'kurtosis'] > 0.1 else '[??] Unexpected'}")
    print(f"  => Std Pos:   {corr_matrix.loc['ott_std', 'std_position']:+.3f}")

    print("\nPutt Variance correlations:")
    print(f"  => Kurtosis:  {corr_matrix.loc['putt_std', 'kurtosis']:+.3f} {'[OK] Expected (weaker)' if corr_matrix.loc['putt_std', 'kurtosis'] < corr_matrix.loc['ott_std', 'kurtosis'] else '[??] Unexpected'}")
    print(f"  => Std Pos:   {corr_matrix.loc['putt_std', 'std_position']:+.3f}")

    print("\nTotal Skill (mean) correlations:")
    print(f"  => Kurtosis:  {corr_matrix.loc['total_mean', 'kurtosis']:+.3f}")
    print(f"  => Skewness:  {corr_matrix.loc['total_mean', 'skewness']:+.3f} (negative = better players have upside)")

    print("\n" + "-"*70)
    print("INTERPRETATION GUIDE")
    print("-"*70)
    print("""
    If the model is working as theoretically expected:

    1. OTT variance should correlate POSITIVELY with kurtosis
       (Bombers have fat-tailed finish distributions due to compounding)

    2. Putt variance should correlate LESS with kurtosis than OTT
       (Putting regresses due to negative coefficients)

    3. High-skill players should have NEGATIVE skewness
       (More concentrated at top, less downside)

    4. "Consistent" archetype should have LOWER std_position than "high_variance"
       (Profile variance translates to outcome variance)

    If these patterns DON'T appear, it suggests:
    - The category projection may be washing out profile differences
    - The coefficient structure may not be having the intended effect
    - Or the theory is wrong (always possible!)
    """)

    print("="*70)


def export_player_details(combined):
    """Export detailed player-level data for further analysis."""
    output_cols = [
        'player_name', 'primary_archetype', 'archetypes',
        'total_mean', 'total_std',
        'ott_mean', 'ott_std', 'putt_mean', 'putt_std',
        'app_mean', 'app_std', 'arg_mean', 'arg_std',
        'bs_sg_ratio',
        'mean_position', 'std_position', 'kurtosis', 'skewness',
        'win_prob', 'top5_prob', 'top10_prob', 'top20_prob', 'mc_prob',
        'bimodality', 'extreme_prob'
    ]

    export_df = combined[[c for c in output_cols if c in combined.columns]].copy()
    export_df = export_df.sort_values('mean_position')

    filename = OUTPUT_DIR / f'archetype_player_details_{TOURNEY}.csv'
    export_df.to_csv(filename, index=False)
    print(f"[ok] Exported player details to {filename}")

    return export_df


def main():
    print("="*70)
    print("ARCHETYPE DIAGNOSTICS")
    print("Analyzing how SG profiles translate to finish distributions")
    print("="*70)

    # Find files
    if not find_tournament_files():
        print("[error] Could not find tournament output files.")
        print("        Make sure rank_probs_updated_*.parquet exists.")
        return

    # Load data
    print("\n[1/5] Loading player distributions...")
    means, stds, df_t = load_player_distributions()
    print(f"       Loaded {len(means)} players from {DISTS_FILE}")

    print("\n[2/5] Loading rank probabilities...")
    rank_probs = load_rank_probabilities()
    print(f"       Loaded {rank_probs['player_name'].nunique()} players from rank probs")

    print("\n[3/5] Classifying archetypes...")
    arch_df = classify_archetypes(means, stds, df_t)
    print(f"       Archetype breakdown:")
    for arch, count in arch_df['primary_archetype'].value_counts().items():
        print(f"         {arch}: {count}")

    print("\n[4/5] Computing distribution metrics...")
    finish_equity = load_finish_equity()
    metrics_df = compute_distribution_metrics(rank_probs, finish_equity)
    print(f"       Computed metrics for {len(metrics_df)} players")

    print("\n[5/5] Analyzing by archetype...")
    combined, summary = analyze_by_archetype(arch_df, metrics_df)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_archetype_distributions(combined, rank_probs)
    corr_matrix = plot_correlation_matrix(combined)

    # Summary report
    print_summary_report(combined, summary, corr_matrix)

    # Export details
    export_player_details(combined)

    print("\n[done] Diagnostics complete.")
    return combined, summary, corr_matrix


if __name__ == "__main__":
    combined, summary, corr_matrix = main()
