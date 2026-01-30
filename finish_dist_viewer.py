# ============================
# Finish Distribution Viewer
# Load saved distributions and display histograms in Plotly
# With Advanced Distribution Shape Analysis
# NOW: Metrics computed on FULL distribution (including MC)
# NEW: Category contribution analysis
# ============================

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from scipy import stats as scipy_stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Field size / cut line constants
FIELD_SIZE = 120
CUT_LINE = 65


# =============================================================================
# DATA LOADING
# =============================================================================

def load_finish_distributions(npz_path):
    """
    Load finish distributions from saved .npz file
    
    Returns dict with:
        - 'player_names': list of player names
        - 'finish_positions': dict mapping player_name -> np.array of finish positions
        - 'made_cut_mask': dict mapping player_name -> np.array of bools (True = made cut)
        - 'n_simulations': number of simulations
        - 'category_data': dict with category analysis (if available)
    """
    data = np.load(npz_path, allow_pickle=True)
    player_names = data['player_names'].tolist()
    finish_positions = data['finish_positions']
    n_simulations = int(data['n_simulations'])
    
    # Load made_cut_mask if available
    if 'made_cut_mask' in data:
        made_cut_mask = data['made_cut_mask']
        cut_dists = {
            player_names[i]: made_cut_mask[i] 
            for i in range(len(player_names))
        }
    else:
        cut_dists = None
    
    player_dists = {
        player_names[i]: finish_positions[i] 
        for i in range(len(player_names))
    }
    
    # Load category data if available
    category_data = None
    if 'category_names' in data:
        category_names = data['category_names'].tolist()
        category_data = {
            'category_names': category_names,
            'correlations': {player_names[i]: data['category_correlations'][i] for i in range(len(player_names))},
            'mean_t10': {player_names[i]: data['category_mean_t10'][i] for i in range(len(player_names))},
            'mean_bottom': {player_names[i]: data['category_mean_bottom'][i] for i in range(len(player_names))},
            'mean_all': {player_names[i]: data['category_mean_all'][i] for i in range(len(player_names))},
            'std_all': {player_names[i]: data['category_std_all'][i] for i in range(len(player_names))},
            'impact': {player_names[i]: data['category_impact'][i] for i in range(len(player_names))},
        }
    
    return {
        'player_names': player_names,
        'finish_positions': player_dists,
        'made_cut_mask': cut_dists,
        'n_simulations': n_simulations,
        'category_data': category_data
    }


# =============================================================================
# BASIC STATS (computed on FULL distribution)
# =============================================================================

def get_player_stats(player_dists, player_name, made_cut_mask=None):
    """Get summary stats for a player's FULL finish distribution"""
    if player_name not in player_dists:
        return None
    
    positions = player_dists[player_name]
    n = len(positions)
    
    win_pct = np.sum(positions == 1) / n * 100
    t5_pct = np.sum(positions <= 5) / n * 100
    t10_pct = np.sum(positions <= 10) / n * 100
    t20_pct = np.sum(positions <= 20) / n * 100
    
    # MC% calculation
    if made_cut_mask is not None and player_name in made_cut_mask:
        cut_mask = made_cut_mask[player_name]
        mc_pct = (1 - np.mean(cut_mask)) * 100
    else:
        mc_pct = np.sum(positions > CUT_LINE) / n * 100
    
    # Shape stats on FULL distribution
    if len(positions) > 3:
        skewness = scipy_stats.skew(positions)
        kurtosis = scipy_stats.kurtosis(positions)
    else:
        skewness = np.nan
        kurtosis = np.nan
    
    return {
        'player': player_name,
        'win_pct': win_pct,
        't5_pct': t5_pct,
        't10_pct': t10_pct,
        't20_pct': t20_pct,
        'mc_pct': mc_pct,
        'median_finish': np.median(positions),
        'mean_finish': np.mean(positions),
        'std_finish': np.std(positions),
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def get_all_player_stats(player_dists, made_cut_mask=None):
    """Get stats for all players as a list of dicts"""
    all_stats = []
    for player in player_dists.keys():
        stats = get_player_stats(player_dists, player, made_cut_mask)
        if stats:
            all_stats.append(stats)
    return all_stats


# =============================================================================
# ADVANCED DISTRIBUTION SHAPE ANALYSIS (on FULL distribution)
# =============================================================================

def compute_advanced_shape_metrics(player_dists, player_name, made_cut_mask=None):
    """
    Compute advanced distribution shape metrics on FULL distribution.
    """
    if player_name not in player_dists:
        return None
    
    positions = player_dists[player_name]
    n = len(positions)
    
    # MC rate
    if made_cut_mask is not None and player_name in made_cut_mask:
        cut_mask = made_cut_mask[player_name]
        mc_rate = (1 - np.mean(cut_mask)) * 100
    else:
        mc_rate = np.sum(positions > CUT_LINE) / n * 100
    
    if len(positions) < 100:
        return None
    
    # ----- 1. TAIL ASYMMETRY (on full distribution) -----
    left_tail_count = np.sum(positions <= 10)
    right_tail_count = np.sum(positions >= 40)
    
    left_tail_pct = left_tail_count / n * 100
    right_tail_pct = right_tail_count / n * 100
    tail_asymmetry = left_tail_pct - right_tail_pct
    
    # ----- 2. BIMODALITY DETECTION -----
    hist, bin_edges = np.histogram(positions, bins=60, range=(0.5, 120.5), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    smoothed = gaussian_filter1d(hist, sigma=2)
    
    peaks, peak_props = find_peaks(smoothed, height=np.max(smoothed) * 0.1, 
                                    distance=5, prominence=np.max(smoothed) * 0.05)
    
    peak_positions = bin_centers[peaks].tolist() if len(peaks) > 0 else []
    n_peaks = len(peaks)
    
    # Bimodality coefficient
    skew = scipy_stats.skew(positions)
    kurt = scipy_stats.kurtosis(positions)
    
    if n > 3:
        bimodality_coef = (skew**2 + 1) / (kurt + 3)
    else:
        bimodality_coef = np.nan
    
    # ----- 3. COEFFICIENT OF VARIATION (full distribution) -----
    cv_full = np.std(positions) / np.mean(positions) if np.mean(positions) > 0 else np.nan
    
    # ----- 4. QUANTILE SPREAD RATIOS -----
    p10 = np.percentile(positions, 10)
    p25 = np.percentile(positions, 25)
    p50 = np.percentile(positions, 50)
    p75 = np.percentile(positions, 75)
    p90 = np.percentile(positions, 90)
    
    top_spread = p25 - p10
    mid_spread = p50 - p25
    
    if mid_spread > 0:
        quantile_spread_ratio = top_spread / mid_spread
    else:
        quantile_spread_ratio = np.nan
    
    upper_half_spread = p75 - p50
    lower_half_spread = p50 - p25
    
    if lower_half_spread > 0:
        upper_lower_ratio = upper_half_spread / lower_half_spread
    else:
        upper_lower_ratio = np.nan
    
    return {
        'player': player_name,
        'tail_asymmetry': tail_asymmetry,
        'left_tail_pct': left_tail_pct,
        'right_tail_pct': right_tail_pct,
        'bimodality_coef': bimodality_coef,
        'n_peaks': n_peaks,
        'peak_positions': peak_positions,
        'cv_full': cv_full,
        'quantile_spread_ratio': quantile_spread_ratio,
        'upper_lower_ratio': upper_lower_ratio,
        'p10': p10,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'mc_rate': mc_rate,
        'skewness': skew,
        'kurtosis': kurt
    }


def compute_all_advanced_metrics(player_dists, made_cut_mask=None):
    """Compute advanced metrics for all players"""
    all_metrics = []
    for player in player_dists.keys():
        metrics = compute_advanced_shape_metrics(player_dists, player, made_cut_mask)
        if metrics:
            all_metrics.append(metrics)
    return all_metrics


# =============================================================================
# OUTLIER DETECTION BY DECILE
# =============================================================================

def find_interesting_distributions_by_decile(player_dists, made_cut_mask=None):
    """Find players with interesting T20/Win ratios by decile."""
    all_stats = get_all_player_stats(player_dists, made_cut_mask)
    
    all_stats.sort(key=lambda x: x['t20_pct'], reverse=True)
    n = len(all_stats)
    
    for p in all_stats:
        win = p['win_pct']
        p['t20_win_ratio'] = p['t20_pct'] / win if win > 0.1 else np.nan
        p['t10_win_ratio'] = p['t10_pct'] / win if win > 0.1 else np.nan
    
    decile_bounds = [
        (0, int(n * 0.05), "Top 5%"),
        (int(n * 0.05), int(n * 0.10), "5-10%"),
        (int(n * 0.10), int(n * 0.20), "10-20%"),
        (int(n * 0.20), int(n * 0.30), "20-30%"),
        (int(n * 0.30), int(n * 0.40), "30-40%"),
        (int(n * 0.40), int(n * 0.50), "40-50%"),
        (int(n * 0.50), int(n * 0.60), "50-60%"),
        (int(n * 0.60), int(n * 0.70), "60-70%"),
        (int(n * 0.70), int(n * 0.80), "70-80%"),
        (int(n * 0.80), int(n * 0.90), "80-90%"),
        (int(n * 0.90), n, "90-100%"),
    ]
    
    high_outliers = []
    low_outliers = []
    
    for start, end, label in decile_bounds:
        decile_players = all_stats[start:end]
        valid = [p for p in decile_players if not np.isnan(p['t20_win_ratio'])]
        
        if not valid:
            continue
        
        sorted_by_ratio = sorted(valid, key=lambda x: x['t20_win_ratio'], reverse=True)
        
        high_player = sorted_by_ratio[0].copy()
        high_player['decile'] = label
        high_outliers.append(high_player)
        
        low_player = sorted_by_ratio[-1].copy()
        low_player['decile'] = label
        low_outliers.append(low_player)
    
    return {'high_outliers': high_outliers, 'low_outliers': low_outliers}


def find_shape_outliers_by_decile(player_dists, made_cut_mask=None, metric='tail_asymmetry'):
    """Find players with extreme shape metrics within each decile."""
    all_stats = get_all_player_stats(player_dists, made_cut_mask)
    all_stats.sort(key=lambda x: x['t20_pct'], reverse=True)
    n = len(all_stats)
    
    all_advanced = {m['player']: m for m in compute_all_advanced_metrics(player_dists, made_cut_mask)}
    
    decile_bounds = [
        (0, int(n * 0.05), "Top 5%"),
        (int(n * 0.05), int(n * 0.10), "5-10%"),
        (int(n * 0.10), int(n * 0.20), "10-20%"),
        (int(n * 0.20), int(n * 0.30), "20-30%"),
        (int(n * 0.30), int(n * 0.40), "30-40%"),
        (int(n * 0.40), int(n * 0.50), "40-50%"),
        (int(n * 0.50), int(n * 0.60), "50-60%"),
        (int(n * 0.60), int(n * 0.70), "60-70%"),
        (int(n * 0.70), int(n * 0.80), "70-80%"),
        (int(n * 0.80), int(n * 0.90), "80-90%"),
        (int(n * 0.90), n, "90-100%"),
    ]
    
    high_outliers = []
    low_outliers = []
    
    for start, end, label in decile_bounds:
        decile_players = all_stats[start:end]
        
        valid = []
        for p in decile_players:
            if p['player'] in all_advanced:
                adv = all_advanced[p['player']]
                if not np.isnan(adv.get(metric, np.nan)):
                    combined = {**p, **adv}
                    valid.append(combined)
        
        if not valid:
            continue
        
        sorted_by_metric = sorted(valid, key=lambda x: x[metric], reverse=True)
        
        high_player = sorted_by_metric[0].copy()
        high_player['decile'] = label
        high_outliers.append(high_player)
        
        low_player = sorted_by_metric[-1].copy()
        low_player['decile'] = label
        low_outliers.append(low_player)
    
    return {'high_outliers': high_outliers, 'low_outliers': low_outliers, 'metric': metric}


# =============================================================================
# CATEGORY ANALYSIS FUNCTIONS
# =============================================================================

def find_category_outliers_by_decile(player_dists, category_data, made_cut_mask=None, 
                                      category_idx=0, metric='correlation'):
    """
    Find players with extreme category metrics within each decile.
    
    Args:
        category_idx: 0=OTT, 1=APP, 2=ARG, 3=PUTT
        metric: 'correlation', 'impact', 'std'
    """
    if category_data is None:
        return None
    
    all_stats = get_all_player_stats(player_dists, made_cut_mask)
    all_stats.sort(key=lambda x: x['t20_pct'], reverse=True)
    n = len(all_stats)
    
    decile_bounds = [
        (0, int(n * 0.05), "Top 5%"),
        (int(n * 0.05), int(n * 0.10), "5-10%"),
        (int(n * 0.10), int(n * 0.20), "10-20%"),
        (int(n * 0.20), int(n * 0.30), "20-30%"),
        (int(n * 0.30), int(n * 0.40), "30-40%"),
        (int(n * 0.40), int(n * 0.50), "40-50%"),
        (int(n * 0.50), int(n * 0.60), "50-60%"),
        (int(n * 0.60), int(n * 0.70), "60-70%"),
        (int(n * 0.70), int(n * 0.80), "70-80%"),
        (int(n * 0.80), int(n * 0.90), "80-90%"),
        (int(n * 0.90), n, "90-100%"),
    ]
    
    # Get the right data source based on metric
    if metric == 'correlation':
        data_source = category_data['correlations']
    elif metric == 'impact':
        data_source = category_data['impact']
    elif metric == 'std':
        data_source = category_data['std_all']
    else:
        return None
    
    high_outliers = []
    low_outliers = []
    
    for start, end, label in decile_bounds:
        decile_players = all_stats[start:end]
        
        valid = []
        for p in decile_players:
            player = p['player']
            if player in data_source:
                val = data_source[player][category_idx]
                if not np.isnan(val):
                    p_copy = p.copy()
                    p_copy['cat_value'] = val
                    p_copy['cat_corr'] = category_data['correlations'][player][category_idx]
                    p_copy['cat_impact'] = category_data['impact'][player][category_idx]
                    p_copy['cat_std'] = category_data['std_all'][player][category_idx]
                    p_copy['cat_mean_t10'] = category_data['mean_t10'][player][category_idx]
                    p_copy['cat_mean_bottom'] = category_data['mean_bottom'][player][category_idx]
                    valid.append(p_copy)
        
        if not valid:
            continue
        
        sorted_by_val = sorted(valid, key=lambda x: x['cat_value'], reverse=True)
        
        high_player = sorted_by_val[0].copy()
        high_player['decile'] = label
        high_outliers.append(high_player)
        
        low_player = sorted_by_val[-1].copy()
        low_player['decile'] = label
        low_outliers.append(low_player)
    
    return {'high_outliers': high_outliers, 'low_outliers': low_outliers}


# =============================================================================
# PRINTING FUNCTIONS
# =============================================================================

def print_interesting_distributions(player_dists, made_cut_mask=None):
    """Pretty print T20/Win ratio analysis by decile"""
    results = find_interesting_distributions_by_decile(player_dists, made_cut_mask)
    
    print("\n" + "="*80)
    print("DISTRIBUTION SHAPE OUTLIERS BY EQUITY DECILE (T20/Win Ratio)")
    print("="*80)
    
    print("\n📊 HIGH T20/Win Ratio (Wide upside but hard to close within tier)")
    print("-" * 80)
    print(f"{'Decile':<10} {'Player':<22} {'Win%':>7} {'T10%':>7} {'T20%':>7} {'MC%':>7} {'T20/Win':>9}")
    print("-" * 80)
    for p in results['high_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['win_pct']:>6.2f}% {p['t10_pct']:>6.2f}% {p['t20_pct']:>6.2f}% {p['mc_pct']:>6.1f}% {p['t20_win_ratio']:>9.1f}")
    
    print("\n🎯 LOW T20/Win Ratio (Top-heavy / boom or bust within tier)")
    print("-" * 80)
    print(f"{'Decile':<10} {'Player':<22} {'Win%':>7} {'T10%':>7} {'T20%':>7} {'MC%':>7} {'T20/Win':>9}")
    print("-" * 80)
    for p in results['low_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['win_pct']:>6.2f}% {p['t10_pct']:>6.2f}% {p['t20_pct']:>6.2f}% {p['mc_pct']:>6.1f}% {p['t20_win_ratio']:>9.1f}")
    
    return results


def print_tail_asymmetry_analysis(player_dists, made_cut_mask=None):
    """Print tail asymmetry analysis by decile"""
    results = find_shape_outliers_by_decile(player_dists, made_cut_mask, 'tail_asymmetry')
    
    print("\n" + "="*85)
    print("TAIL ASYMMETRY ANALYSIS BY DECILE (Full Distribution)")
    print("(Left Tail = positions 1-10 | Right Tail = positions 40+)")
    print("="*85)
    
    print("\n📈 HIGH Tail Asymmetry (More upside tail than downside)")
    print("-" * 85)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'L.Tail%':>8} {'R.Tail%':>8} {'Asymm':>8} {'MC%':>7}")
    print("-" * 85)
    for p in results['high_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['left_tail_pct']:>7.1f}% {p['right_tail_pct']:>7.1f}% {p['tail_asymmetry']:>+7.1f} {p['mc_rate']:>6.1f}%")
    
    print("\n📉 LOW Tail Asymmetry (More downside tail than upside)")
    print("-" * 85)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'L.Tail%':>8} {'R.Tail%':>8} {'Asymm':>8} {'MC%':>7}")
    print("-" * 85)
    for p in results['low_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['left_tail_pct']:>7.1f}% {p['right_tail_pct']:>7.1f}% {p['tail_asymmetry']:>+7.1f} {p['mc_rate']:>6.1f}%")
    
    return results


def print_bimodality_analysis(player_dists, made_cut_mask=None):
    """Print bimodality analysis by decile"""
    results = find_shape_outliers_by_decile(player_dists, made_cut_mask, 'bimodality_coef')
    
    print("\n" + "="*90)
    print("BIMODALITY / CLUSTERING ANALYSIS BY DECILE (Full Distribution)")
    print("(Higher coef = more bimodal | n_peaks = detected modes)")
    print("="*90)
    
    print("\n🔀 HIGH Bimodality (Two distinct outcome clusters)")
    print("-" * 90)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'BimodalC':>9} {'Peaks':>6} {'Peak Positions':<25}")
    print("-" * 90)
    for p in results['high_outliers']:
        peaks_str = ', '.join([f"{x:.0f}" for x in p.get('peak_positions', [])[:3]])
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['bimodality_coef']:>9.3f} {p['n_peaks']:>6} {peaks_str:<25}")
    
    print("\n🎯 LOW Bimodality (Unimodal / consistent shape)")
    print("-" * 90)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'BimodalC':>9} {'Peaks':>6} {'Peak Positions':<25}")
    print("-" * 90)
    for p in results['low_outliers']:
        peaks_str = ', '.join([f"{x:.0f}" for x in p.get('peak_positions', [])[:3]])
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['bimodality_coef']:>9.3f} {p['n_peaks']:>6} {peaks_str:<25}")
    
    return results


def print_cv_analysis(player_dists, made_cut_mask=None):
    """Print CV analysis by decile"""
    results = find_shape_outliers_by_decile(player_dists, made_cut_mask, 'cv_full')
    
    print("\n" + "="*85)
    print("COEFFICIENT OF VARIATION ANALYSIS BY DECILE (Full Distribution)")
    print("(Higher CV = more volatile | Lower CV = more consistent)")
    print("="*85)
    
    print("\n🎲 HIGH CV (Volatile finish positions)")
    print("-" * 85)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'CV':>8} {'Median':>8} {'StdDev':>8} {'MC%':>7}")
    print("-" * 85)
    for p in results['high_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['cv_full']:>8.3f} {p['p50']:>7.1f} {p['p50']*p['cv_full']:>7.1f} {p['mc_rate']:>6.1f}%")
    
    print("\n🎯 LOW CV (Consistent finish positions)")
    print("-" * 85)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'CV':>8} {'Median':>8} {'StdDev':>8} {'MC%':>7}")
    print("-" * 85)
    for p in results['low_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['cv_full']:>8.3f} {p['p50']:>7.1f} {p['p50']*p['cv_full']:>7.1f} {p['mc_rate']:>6.1f}%")
    
    return results


def print_quantile_spread_analysis(player_dists, made_cut_mask=None):
    """Print quantile spread analysis by decile"""
    results = find_shape_outliers_by_decile(player_dists, made_cut_mask, 'quantile_spread_ratio')
    
    print("\n" + "="*90)
    print("QUANTILE SPREAD RATIO ANALYSIS BY DECILE (Full Distribution)")
    print("(Ratio = P10-P25 / P25-P50 | Low = ceiling effect, High = no ceiling)")
    print("="*90)
    
    print("\n📊 HIGH Spread Ratio (Wide spread in top quartile)")
    print("-" * 90)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'Ratio':>7} {'P10':>6} {'P25':>6} {'P50':>6} {'P75':>6}")
    print("-" * 90)
    for p in results['high_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['quantile_spread_ratio']:>7.2f} {p['p10']:>6.1f} {p['p25']:>6.1f} {p['p50']:>6.1f} {p['p75']:>6.1f}")
    
    print("\n🚧 LOW Spread Ratio (Compressed top quartile)")
    print("-" * 90)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'Ratio':>7} {'P10':>6} {'P25':>6} {'P50':>6} {'P75':>6}")
    print("-" * 90)
    for p in results['low_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['quantile_spread_ratio']:>7.2f} {p['p10']:>6.1f} {p['p25']:>6.1f} {p['p50']:>6.1f} {p['p75']:>6.1f}")
    
    return results


def print_kurtosis_analysis(player_dists, made_cut_mask=None):
    """Print kurtosis analysis by decile"""
    results = find_shape_outliers_by_decile(player_dists, made_cut_mask, 'kurtosis')
    
    print("\n" + "="*85)
    print("KURTOSIS (TAIL WEIGHT) ANALYSIS BY DECILE (Full Distribution)")
    print("(High = fat tails | Low = thin tails)")
    print("="*85)
    
    print("\n🎰 HIGH Kurtosis (Fat tails - more extreme outcomes)")
    print("-" * 85)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'Kurtosis':>9} {'Skewness':>9} {'MC%':>7}")
    print("-" * 85)
    for p in results['high_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['kurtosis']:>+9.2f} {p['skewness']:>+9.2f} {p['mc_rate']:>6.1f}%")
    
    print("\n📍 LOW Kurtosis (Thin tails - clustered outcomes)")
    print("-" * 85)
    print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'Kurtosis':>9} {'Skewness':>9} {'MC%':>7}")
    print("-" * 85)
    for p in results['low_outliers']:
        print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['kurtosis']:>+9.2f} {p['skewness']:>+9.2f} {p['mc_rate']:>6.1f}%")
    
    return results


def print_category_analysis(player_dists, category_data, made_cut_mask=None):
    """Print category contribution analysis by decile"""
    if category_data is None:
        print("\n⚠️  Category data not available. Re-run new_sim_comb.py to generate.")
        return None
    
    cat_names = category_data['category_names']
    
    print("\n" + "="*95)
    print("CATEGORY CONTRIBUTION ANALYSIS BY DECILE")
    print("(Correlation = how much category correlates with finish position, negative = better)")
    print("(Impact = mean_T10 - mean_40+, positive = category helps top finishes)")
    print("="*95)
    
    for cat_idx, cat_name in enumerate(cat_names):
        cat_display = cat_name.upper().replace('SG_', '')
        
        # Correlation analysis (most negative = most helpful)
        results = find_category_outliers_by_decile(player_dists, category_data, made_cut_mask,
                                                    category_idx=cat_idx, metric='correlation')
        if results is None:
            continue
        
        print(f"\n{'='*95}")
        print(f"📊 {cat_display} - CORRELATION WITH FINISH POSITION")
        print(f"{'='*95}")
        
        print(f"\n🔻 Most NEGATIVE Correlation (Category most helps finish position)")
        print("-" * 95)
        print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'Corr':>8} {'Impact':>8} {'μ T10':>8} {'μ 40+':>8} {'σ':>7}")
        print("-" * 95)
        # Low outliers have most negative correlation (good)
        for p in results['low_outliers']:
            print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['cat_corr']:>+7.3f} {p['cat_impact']:>+7.2f} {p['cat_mean_t10']:>7.2f} {p['cat_mean_bottom']:>7.2f} {p['cat_std']:>6.2f}")
        
        print(f"\n🔺 Most POSITIVE Correlation (Category least helps / hurts finish)")
        print("-" * 95)
        print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'Corr':>8} {'Impact':>8} {'μ T10':>8} {'μ 40+':>8} {'σ':>7}")
        print("-" * 95)
        for p in results['high_outliers']:
            print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['cat_corr']:>+7.3f} {p['cat_impact']:>+7.2f} {p['cat_mean_t10']:>7.2f} {p['cat_mean_bottom']:>7.2f} {p['cat_std']:>6.2f}")
    
    # Also show variance outliers
    print(f"\n{'='*95}")
    print("📊 CATEGORY VARIANCE OUTLIERS (Highest variance by category)")
    print("="*95)
    
    for cat_idx, cat_name in enumerate(cat_names):
        cat_display = cat_name.upper().replace('SG_', '')
        results = find_category_outliers_by_decile(player_dists, category_data, made_cut_mask,
                                                    category_idx=cat_idx, metric='std')
        if results is None:
            continue
        
        print(f"\n🎲 {cat_display} - HIGH VARIANCE (Most volatile in this category)")
        print("-" * 95)
        print(f"{'Decile':<10} {'Player':<22} {'T20%':>7} {'σ':>7} {'Corr':>8} {'Impact':>8}")
        print("-" * 95)
        for p in results['high_outliers'][:6]:  # Top 6 only
            print(f"{p['decile']:<10} {p['player'].title():<22} {p['t20_pct']:>6.1f}% {p['cat_std']:>6.2f} {p['cat_corr']:>+7.3f} {p['cat_impact']:>+7.2f}")


def print_full_shape_analysis(player_dists, made_cut_mask=None, category_data=None):
    """Run and print all shape analyses"""
    print("\n" + "="*95)
    print("              COMPREHENSIVE DISTRIBUTION SHAPE ANALYSIS (Full Distribution)")
    print("="*95)
    
    print_interesting_distributions(player_dists, made_cut_mask)
    print_tail_asymmetry_analysis(player_dists, made_cut_mask)
    print_bimodality_analysis(player_dists, made_cut_mask)
    print_cv_analysis(player_dists, made_cut_mask)
    print_quantile_spread_analysis(player_dists, made_cut_mask)
    print_kurtosis_analysis(player_dists, made_cut_mask)
    
    if category_data is not None:
        print_category_analysis(player_dists, category_data, made_cut_mask)
    
    print("\n" + "="*95)
    print("INTERPRETATION GUIDE")
    print("="*95)
    print("""
    T20/Win Ratio:
      HIGH = Wide equity but doesn't close (good T20 value vs win odds)
      LOW  = Top-heavy, wins when contending (good win value vs T20 odds)
    
    Tail Asymmetry:
      HIGH (+) = More upside (T10) than downside (40+) 
      LOW  (-) = More downside than upside - risky
    
    Bimodality:
      HIGH = Two distinct outcome clusters (Jekyll/Hyde)
      LOW  = Single mode, consistent shape
    
    Coefficient of Variation (CV):
      HIGH = Volatile finish positions
      LOW  = Consistent finisher
    
    Quantile Spread Ratio:
      HIGH = Wide spread in top quartile (no ceiling)
      LOW  = Compressed top (ceiling effect)
    
    Kurtosis:
      HIGH = Fat tails, more extremes
      LOW  = Thin tails, clustered outcomes
    
    Category Correlation:
      NEGATIVE = Category helps finish (higher SG -> lower position number)
      POSITIVE = Category doesn't differentiate or hurts
    
    Category Impact:
      POSITIVE = Category is better in T10 finishes than 40+ finishes
      Shows which categories drive a player's best vs worst outcomes
    """)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_interactive_viewer(player_dists, n_simulations=None, default_player=None, made_cut_mask=None):
    """Create interactive Plotly histogram with dropdown menu."""
    sorted_players = sorted(player_dists.keys())
    
    if default_player and default_player in player_dists:
        initial_player = default_player
    else:
        initial_player = sorted_players[0]
    
    all_stats = {p: get_player_stats(player_dists, p, made_cut_mask) for p in sorted_players}
    
    fig = go.Figure()
    
    for i, player in enumerate(sorted_players):
        positions = player_dists[player]
        visible = (player == initial_player)
        
        fig.add_trace(go.Histogram(
            x=positions,
            xbins=dict(start=0.5, end=FIELD_SIZE + 0.5, size=1),
            name='Finish Position',
            marker_color='steelblue',
            opacity=0.75,
            visible=visible,
            hovertemplate="Position: %{x}<br>Count: %{y}<extra></extra>"
        ))
    
    buttons = []
    for i, player in enumerate(sorted_players):
        stats = all_stats[player]
        visibility = [False] * len(sorted_players)
        visibility[i] = True
        
        title_text = (
            f"<b>{player.title()}</b><br>"
            f"<sup>Win: {stats['win_pct']:.2f}% | T5: {stats['t5_pct']:.1f}% | "
            f"T10: {stats['t10_pct']:.1f}% | T20: {stats['t20_pct']:.1f}% | "
            f"MC: {stats['mc_pct']:.1f}%</sup>"
        )
        if n_simulations:
            title_text = title_text.replace("</sup>", f" | {n_simulations:,} sims</sup>")
        
        buttons.append(dict(
            label=player.title(),
            method='update',
            args=[{'visible': visibility}, {'title.text': title_text}]
        ))
    
    init_stats = all_stats[initial_player]
    initial_title = (
        f"<b>{initial_player.title()}</b><br>"
        f"<sup>Win: {init_stats['win_pct']:.2f}% | T5: {init_stats['t5_pct']:.1f}% | "
        f"T10: {init_stats['t10_pct']:.1f}% | T20: {init_stats['t20_pct']:.1f}% | "
        f"MC: {init_stats['mc_pct']:.1f}%</sup>"
    )
    if n_simulations:
        initial_title = initial_title.replace("</sup>", f" | {n_simulations:,} sims</sup>")
    
    for thresh, color, name in [(1, 'gold', 'Win'), (5, 'green', 'T5'), 
                                 (10, 'orange', 'T10'), (20, 'red', 'T20')]:
        fig.add_vline(x=thresh + 0.5, line_dash="dash", line_color=color, 
                     annotation_text=name, annotation_position="top")
    
    fig.add_vline(x=CUT_LINE + 0.5, line_dash="solid", line_color="darkred", 
                 line_width=2, annotation_text="CUT", annotation_position="top")
    
    fig.update_layout(
        title=dict(text=initial_title, x=0.5),
        xaxis_title="Finish Position",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, FIELD_SIZE + 1]),
        bargap=0.1,
        template='plotly_white',
        updatemenus=[dict(
            active=sorted_players.index(initial_player),
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left", y=1.15, yanchor="top",
            bgcolor="white", bordercolor="lightgray", borderwidth=1
        )],
        annotations=[dict(
            text="Select Player:", x=0.0, xref="paper", y=1.22, yref="paper",
            showarrow=False, font=dict(size=12)
        )]
    )
    
    return fig


def plot_finish_histogram(player_dists, player_name, n_simulations=None, made_cut_mask=None):
    """Create histogram for single player."""
    if player_name not in player_dists:
        raise ValueError(f"Player '{player_name}' not found.")
    
    positions = player_dists[player_name]
    
    if made_cut_mask is not None and player_name in made_cut_mask:
        mc_pct = (1 - np.mean(made_cut_mask[player_name])) * 100
    else:
        mc_pct = np.sum(positions > CUT_LINE) / len(positions) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=positions,
        xbins=dict(start=0.5, end=FIELD_SIZE + 0.5, size=1),
        marker_color='steelblue',
        opacity=0.75
    ))
    
    for thresh, color, name in [(1, 'gold', 'Win'), (5, 'green', 'T5'), 
                                 (10, 'orange', 'T10'), (20, 'red', 'T20')]:
        fig.add_vline(x=thresh + 0.5, line_dash="dash", line_color=color, 
                     annotation_text=name, annotation_position="top")
    
    fig.add_vline(x=CUT_LINE + 0.5, line_dash="solid", line_color="darkred", 
                 line_width=2, annotation_text="CUT", annotation_position="top")
    
    subtitle = f"MC Rate: {mc_pct:.1f}%"
    if n_simulations:
        subtitle += f" | {n_simulations:,} sims"
    
    fig.update_layout(
        title=dict(text=f"Finish Distribution: {player_name.title()}<br><sup>{subtitle}</sup>", x=0.5),
        xaxis_title="Finish Position",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, FIELD_SIZE + 1]),
        bargap=0.1,
        template='plotly_white'
    )
    
    return fig


def plot_multiple_players(player_dists, player_names, n_simulations=None, made_cut_mask=None):
    """Create overlay histogram for multiple players."""
    colors = px.colors.qualitative.Set1
    fig = go.Figure()
    
    for i, player_name in enumerate(player_names):
        if player_name not in player_dists:
            print(f"Warning: '{player_name}' not found")
            continue
        
        positions = player_dists[player_name]
        
        fig.add_trace(go.Histogram(
            x=positions,
            xbins=dict(start=0.5, end=FIELD_SIZE + 0.5, size=1),
            name=player_name.title(),
            marker_color=colors[i % len(colors)],
            opacity=0.5
        ))
    
    fig.add_vline(x=CUT_LINE + 0.5, line_dash="solid", line_color="darkred", 
                 line_width=2, annotation_text="CUT", annotation_position="top")
    
    fig.update_layout(
        title="Finish Distribution Comparison",
        xaxis_title="Finish Position",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, FIELD_SIZE + 1]),
        barmode='overlay',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    tourney = "farmers"
    npz_path = f"combined_skill_betting/{tourney}/finish_distributions.npz"
    
    if os.path.exists(npz_path):
        data = load_finish_distributions(npz_path)
        player_dists = data['finish_positions']
        made_cut_mask = data['made_cut_mask']
        n_sims = data['n_simulations']
        category_data = data['category_data']
        
        print(f"Loaded {len(data['player_names'])} players, {n_sims:,} simulations")
        if category_data:
            print(f"Category data available: {category_data['category_names']}")
        
        print_full_shape_analysis(player_dists, made_cut_mask, category_data)
        
        fig = create_interactive_viewer(player_dists, n_sims, made_cut_mask=made_cut_mask)
        fig.show()
        
    else:
        print(f"File not found: {npz_path}")
        print("Run new_sim_comb.py first to generate distributions")