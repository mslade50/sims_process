"""
Correlated Kelly staking optimizer.

Sizes bets on the same golfer jointly, accounting for structural correlation
between nested finish-position markets (win ⊂ top5 ⊂ top10 ⊂ top20) and
matchups sharing the same underlying finishing outcome.

Uses per-sim co-occurrence directly from final_scores — each bet's win/loss
is determined per sim, giving exact joint probabilities with no approximation.

Pure math module — no API calls, no sheets logic, no side effects.
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_correlated_kelly(
    finish_bets,        # list of dicts: player_name, market_type, decimal_odds, sim_prob
    matchup_bets,       # list of dicts: bet_on, opponent, decimal_odds, sim_win_prob
    final_scores,       # (n_players, n_sims) numpy array of 72-hole totals
    player_names,       # list mapping row index -> player name (lowercase)
    finish_probs=None,  # unused, kept for interface compat
    bankroll=30_000.0,
    kelly_fraction=0.60,
    golfer_cap=0.15,    # max fraction on one golfer (pre-fractional)
    portfolio_cap=0.35, # max total deployment (pre-fractional)
    min_bet=0.0,
    book_caps=None,     # dict: book_name -> max dollar risk (e.g. {'pinnacle': 500})
):
    """
    Returns dict with:
        bets     - list of per-bet dicts (label, book, odds, indep_f, corr_f, reduction, dollars)
        golfers  - per-golfer group summaries
        summary  - portfolio-level stats
    """
    name_to_idx = {n: i for i, n in enumerate(player_names)}
    n_sims = final_scores.shape[1]

    # Tag every bet with its primary golfer
    all_bets = []
    for b in finish_bets:
        all_bets.append({
            'type': 'finish',
            'golfer': b['player_name'].lower().strip(),
            'market': b['market_type'],
            'book': b.get('bookmaker', ''),
            'decimal_odds': float(b['decimal_odds']),
            'sim_prob': float(b['sim_prob']),
            'american_odds': b.get('american_odds', ''),
            'label': b['market_type'],
        })
    for b in matchup_bets:
        all_bets.append({
            'type': 'matchup',
            'golfer': b['bet_on'].lower().strip(),
            'opponent': b['opponent'].lower().strip(),
            'book': b.get('bookmaker', ''),
            'decimal_odds': float(b['decimal_odds']),
            'sim_prob': float(b['sim_win_prob']),
            'american_odds': b.get('american_odds', ''),
            'label': f"mu vs {b['opponent'].lower().strip()[:15]}",
        })

    if not all_bets:
        return {'bets': [], 'golfers': {}, 'summary': {}}

    # Group by golfer
    groups = {}
    for bet in all_bets:
        groups.setdefault(bet['golfer'], []).append(bet)

    # Solve each golfer group
    results = []
    for golfer, bets in groups.items():
        g_idx = name_to_idx.get(golfer)
        if g_idx is None:
            for bet in bets:
                f_indep = _independent_kelly_single(bet['sim_prob'], bet['decimal_odds'])
                results.append(_make_result(bet, f_indep, f_indep))
            continue

        if len(bets) == 1:
            bet = bets[0]
            f_indep = _independent_kelly_single(bet['sim_prob'], bet['decimal_odds'])
            results.append(_make_result(bet, f_indep, f_indep))
            continue

        # Multi-bet group: compute per-sim bet outcomes and optimize jointly
        prob_vec, payoff_mat, ordered_bets = _build_joint_from_sims(
            bets, g_idx, name_to_idx, final_scores, n_sims
        )

        indep_fs = [_independent_kelly_single(b['sim_prob'], b['decimal_odds']) for b in ordered_bets]
        corr_fs = _solve_kelly(prob_vec, payoff_mat, golfer_cap, len(ordered_bets))

        for bet, f_i, f_c in zip(ordered_bets, indep_fs, corr_fs):
            results.append(_make_result(bet, f_i, f_c))

    # Post-optimization: apply fractional Kelly
    for r in results:
        r['corr_f_full'] = r['corr_f']
        r['corr_f'] *= kelly_fraction
        r['indep_f_full'] = r['indep_f']
        r['indep_f'] *= kelly_fraction

    # Convert to dollars, apply per-book caps
    _book_caps = book_caps or {}
    for r in results:
        r['dollars'] = round(bankroll * r['corr_f'], 2)
        r['indep_dollars'] = round(bankroll * r['indep_f'], 2)
        # Per-book risk cap
        book_key = r['book'].lower().strip()
        if book_key in _book_caps and r['dollars'] > _book_caps[book_key]:
            r['dollars'] = _book_caps[book_key]
            r['corr_f'] = r['dollars'] / bankroll
            r['book_capped'] = True
        else:
            r['book_capped'] = False
        r['zeroed'] = False
        r['portfolio_scaled'] = False
        r['reduction'] = (1.0 - r['corr_f'] / r['indep_f']) if r['indep_f'] > 0 else 0.0

    # Build summary
    golfer_summaries = {}
    for r in results:
        g = r['golfer']
        if g not in golfer_summaries:
            golfer_summaries[g] = {'bets': 0, 'total_dollars': 0.0, 'total_corr_f': 0.0}
        golfer_summaries[g]['bets'] += 1
        golfer_summaries[g]['total_dollars'] += r['dollars']
        golfer_summaries[g]['total_corr_f'] += r['corr_f']

    multi_groups = [g for g, s in golfer_summaries.items() if s['bets'] > 1]
    avg_reduction = 0.0
    if multi_groups:
        reductions = [r['reduction'] for r in results if r['golfer'] in multi_groups and r['indep_f'] > 0]
        avg_reduction = np.mean(reductions) if reductions else 0.0

    summary = {
        'total_bets': len(results),
        'golfer_groups': len(golfer_summaries),
        'multi_bet_groups': len(multi_groups),
        'avg_reduction': avg_reduction,
        'total_dollars': sum(r['dollars'] for r in results),
        'total_pct': sum(r['corr_f'] for r in results),
        'portfolio_cap_binding': any(r.get('portfolio_scaled') for r in results),
        'bets_zeroed': sum(1 for r in results if r.get('zeroed')),
        'bankroll': bankroll,
        'kelly_fraction': kelly_fraction,
    }

    return {'bets': results, 'golfers': golfer_summaries, 'summary': summary}


# ---------------------------------------------------------------------------
# Joint problem construction — direct per-sim co-occurrence
# ---------------------------------------------------------------------------

_FINISH_THRESHOLDS = {'win': 1, 'winner': 1, 'top_5': 5, 'top_10': 10, 'top_20': 20}


def _bet_outcome_per_sim(bet, golfer_idx, name_to_idx, final_scores, n_sims):
    """
    Compute per-sim win probability for a single bet.
    Returns array of shape (n_sims,) with values in [0, 1].
    For finish bets: uses dead-heat-adjusted rank.
    For matchups: head-to-head with tie splitting.
    """
    golfer_scores = final_scores[golfer_idx]

    if bet['type'] == 'matchup':
        opp_idx = name_to_idx.get(bet['opponent'])
        if opp_idx is None:
            # Opponent not in sim — use sim_prob as fallback
            rng = np.random.default_rng(hash(bet['opponent']) % (2**31))
            return (rng.random(n_sims) < bet['sim_prob']).astype(np.float64)
        opp_scores = final_scores[opp_idx]
        outcomes = np.zeros(n_sims, dtype=np.float64)
        outcomes[golfer_scores < opp_scores] = 1.0   # golfer wins
        outcomes[golfer_scores == opp_scores] = 0.5   # tie: half credit
        return outcomes

    # Finish bet: determine if golfer finishes within threshold
    threshold = _FINISH_THRESHOLDS.get(bet['market'], 20)

    # For each sim, compute rank and dead-heat-adjusted payout
    n_players = final_scores.shape[0]
    outcomes = np.zeros(n_sims, dtype=np.float64)

    # Vectorized: rank = number of players with strictly lower score + 1
    better_count = (final_scores < golfer_scores[np.newaxis, :]).sum(axis=0)
    rank = better_count + 1  # (n_sims,)

    # Count ties at golfer's score (including self)
    tie_count = (final_scores == golfer_scores[np.newaxis, :]).sum(axis=0)

    # Dead-heat factor: overlap of tied positions with threshold
    # If rank=8, tie_count=4 (ranks 8-11), threshold=10: overlap = min(11,10) - max(8,1) + 1 = 3, factor = 3/4
    end_rank = rank + tie_count - 1
    overlap = np.maximum(np.minimum(end_rank, threshold) - np.maximum(rank, 1) + 1, 0)
    outcomes = overlap.astype(np.float64) / tie_count.astype(np.float64)

    return outcomes


def _build_joint_from_sims(bets, golfer_idx, name_to_idx, final_scores, n_sims):
    """
    Build probability vector and payoff matrix directly from per-sim bet outcomes.

    Each bet's win/loss is determined per sim. Joint states are the unique
    combinations of all bet outcomes observed across sims. Fully vectorized.
    """
    n_bets = len(bets)

    # Compute per-sim outcome for each bet: (n_bets, n_sims), values in [0, 1]
    bet_outcomes = np.zeros((n_bets, n_sims), dtype=np.float64)
    for i, bet in enumerate(bets):
        bet_outcomes[i] = _bet_outcome_per_sim(bet, golfer_idx, name_to_idx, final_scores, n_sims)

    profits = np.array([b['decimal_odds'] - 1.0 for b in bets])
    n_states = 2 ** n_bets

    # Separate clean sims (all 0 or 1) from fractional sims (dead heat / ties)
    is_clean = np.all((bet_outcomes == 0.0) | (bet_outcomes == 1.0), axis=0)  # (n_sims,)
    clean_mask = is_clean
    frac_mask = ~is_clean

    prob_vec = np.zeros(n_states)

    # --- Vectorized path for clean sims (vast majority) ---
    if clean_mask.any():
        clean_outcomes = bet_outcomes[:, clean_mask]  # (n_bets, n_clean)
        # Encode each sim as state index: bit i = bet i win (1) or loss (0)
        # MSB = bet 0, LSB = bet n_bets-1
        powers = (2 ** np.arange(n_bets - 1, -1, -1)).reshape(-1, 1)  # (n_bets, 1)
        state_indices = (clean_outcomes.astype(np.int32) * powers).sum(axis=0)  # (n_clean,)
        counts = np.bincount(state_indices, minlength=n_states)
        prob_vec += counts.astype(np.float64) / n_sims

    # --- Fractional sims: split weight across resolved states ---
    if frac_mask.any():
        frac_outcomes = bet_outcomes[:, frac_mask]  # (n_bets, n_frac)
        n_frac_sims = frac_outcomes.shape[1]
        for j in range(n_frac_sims):
            outcomes_j = frac_outcomes[:, j]
            frac_idx = np.where((outcomes_j != 0.0) & (outcomes_j != 1.0))[0]
            n_frac = len(frac_idx)
            for combo in range(2 ** n_frac):
                weight = 1.0 / n_sims
                resolved = outcomes_j.copy()
                for k, fi in enumerate(frac_idx):
                    bit = (combo >> (n_frac - 1 - k)) & 1
                    if bit:
                        weight *= outcomes_j[fi]
                        resolved[fi] = 1.0
                    else:
                        weight *= (1.0 - outcomes_j[fi])
                        resolved[fi] = 0.0
                state = 0
                for i in range(n_bets):
                    state = state * 2 + (1 if resolved[i] > 0.5 else 0)
                prob_vec[state] += weight

    # Build payoff matrix: (n_states, n_bets) — vectorized
    state_bits = ((np.arange(n_states)[:, np.newaxis] >> np.arange(n_bets - 1, -1, -1)) & 1)
    payoff_mat = np.where(state_bits, profits[np.newaxis, :], -1.0)

    return prob_vec, payoff_mat, bets


# ---------------------------------------------------------------------------
# Kelly solver
# ---------------------------------------------------------------------------

def _solve_kelly(prob_vec, payoff_mat, golfer_cap, n_bets):
    """
    Maximize E[log(1 + sum f_i * r_i)] subject to f_i >= 0, sum(f_i) <= golfer_cap.
    Returns array of optimal fractions.
    """
    nonzero = prob_vec > 1e-12
    p = prob_vec[nonzero]
    R = payoff_mat[nonzero]

    def neg_expected_log(f):
        wealth = 1.0 + R @ f
        wealth = np.maximum(wealth, 1e-10)
        return -np.sum(p * np.log(wealth))

    def neg_expected_log_jac(f):
        wealth = 1.0 + R @ f
        wealth = np.maximum(wealth, 1e-10)
        ratio = p / wealth
        return -(R.T @ ratio)

    x0 = np.full(n_bets, 0.001)
    bounds = [(0.0, golfer_cap)] * n_bets
    constraints = [{'type': 'ineq', 'fun': lambda f: golfer_cap - np.sum(f)}]

    result = minimize(
        neg_expected_log,
        x0,
        jac=neg_expected_log_jac,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12},
    )

    return np.maximum(result.x, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _independent_kelly_single(p, decimal_odds):
    """Standard Kelly: f* = (bp - q) / b"""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def _make_result(bet, f_indep, f_corr):
    return {
        'golfer': bet['golfer'],
        'type': bet['type'],
        'label': bet['label'],
        'book': bet['book'],
        'american_odds': bet.get('american_odds', ''),
        'decimal_odds': bet['decimal_odds'],
        'sim_prob': bet['sim_prob'],
        'indep_f': f_indep,
        'corr_f': f_corr,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_correlated_kelly_report(result):
    """Pretty-print the correlated kelly sizing report to console."""
    if not result or not result.get('bets'):
        print("[correlated-kelly] No bets to size.")
        return

    summary = result['summary']
    bk = summary['bankroll']
    kf = summary['kelly_fraction']

    print(f"\n[correlated-kelly] {'='*55}")
    print(f"[correlated-kelly]  Correlated Kelly Staking Report (${bk:,.0f} @ {kf:.2f}x)")
    print(f"[correlated-kelly] {'='*55}")

    golfer_bets = {}
    for r in result['bets']:
        golfer_bets.setdefault(r['golfer'], []).append(r)

    sorted_golfers = sorted(
        golfer_bets.keys(),
        key=lambda g: sum(r['dollars'] for r in golfer_bets[g]),
        reverse=True,
    )

    for golfer in sorted_golfers:
        bets = golfer_bets[golfer]
        n = len(bets)
        corr_note = " -- no correlation" if n == 1 else ""
        print(f"\n[correlated-kelly] {golfer} ({n} bet{'s' if n > 1 else ''}){corr_note}")
        print(f"  {'Bet':<22s} {'Book':<12s} {'Line':>7s}  {'Indep f*':>8s}  {'Corr f*':>8s}  {'Reduct':>8s}  {'$ Size':>8s}")

        for r in sorted(bets, key=lambda x: -x['dollars']):
            odds_str = f"{int(r['american_odds']):+d}" if isinstance(r['american_odds'], (int, float)) and not np.isnan(r['american_odds']) else str(r['american_odds'])
            red_str = f"-{r['reduction']*100:.1f}%" if r['reduction'] > 0 else "0.0%"
            dollar_str = f"${r['dollars']:,.0f}" if r['dollars'] > 0 else "$0"
            zeroed_flag = " (capped)" if r.get('book_capped') else ""
            print(f"  {r['label']:<22s} {r['book']:<12s} {odds_str:>7s}  "
                  f"{r['indep_f']*100:>7.2f}%  {r['corr_f']*100:>7.2f}%  "
                  f"{red_str:>8s}  {dollar_str:>8s}{zeroed_flag}")

        total_d = sum(r['dollars'] for r in bets)
        total_f = sum(r['corr_f'] for r in bets)
        print(f"  Total exposure: {total_f*100:.2f}% (${total_d:,.0f})")

    print(f"\n[correlated-kelly] {'='*20} Portfolio Summary {'='*20}")
    print(f"  Total bets: {summary['total_bets']} | Golfer groups: {summary['golfer_groups']}")
    print(f"  Multi-bet groups: {summary['multi_bet_groups']} (avg reduction: {summary['avg_reduction']*100:.1f}%)")
    print(f"  Total deployment: ${summary['total_dollars']:,.0f} ({summary['total_pct']*100:.1f}% of ${bk:,.0f})")
    cap_str = "BINDING" if summary['portfolio_cap_binding'] else "not binding"
    print(f"  Portfolio cap ({kf*100:.0f}%): {cap_str}")
    if summary['bets_zeroed'] > 0:
        print(f"  Bets zeroed (< min): {summary['bets_zeroed']}")
    print()


# ---------------------------------------------------------------------------
# HTML email report
# ---------------------------------------------------------------------------

def build_correlated_kelly_email_html(result, tourney=""):
    """Build HTML email body for correlated Kelly staking report."""
    if not result or not result.get('bets'):
        return "<p>No bets qualified for correlated Kelly sizing.</p>"

    summary = result['summary']
    bk = summary['bankroll']
    kf = summary['kelly_fraction']

    golfer_bets = {}
    for r in result['bets']:
        golfer_bets.setdefault(r['golfer'], []).append(r)

    sorted_golfers = sorted(
        golfer_bets.keys(),
        key=lambda g: sum(r['dollars'] for r in golfer_bets[g]),
        reverse=True,
    )

    hdr_style = "background:#343a40; color:white; padding:6px 8px; text-align:center; font-size:11px;"
    cell = "padding:5px 8px; font-size:11px; border-bottom:1px solid #dee2e6;"
    golfer_hdr = "background:#495057; color:white; padding:6px 8px; font-size:12px; font-weight:600;"

    rows_html = ""
    for golfer in sorted_golfers:
        bets = golfer_bets[golfer]
        n = len(bets)
        total_d = sum(r['dollars'] for r in bets)
        total_f = sum(r['corr_f'] for r in bets)
        corr_note = " (single bet)" if n == 1 else ""

        rows_html += f"""<tr><td colspan="8" style="{golfer_hdr}">
            {golfer.title()} &mdash; {n} bet{'s' if n > 1 else ''}{corr_note}
            &nbsp;&nbsp;|&nbsp;&nbsp; Exposure: {total_f*100:.2f}% (${total_d:,.0f})
        </td></tr>"""

        for r in sorted(bets, key=lambda x: -x['dollars']):
            odds_str = ""
            try:
                am = r['american_odds']
                if isinstance(am, (int, float)) and not np.isnan(am):
                    odds_str = f"{int(am):+d}"
                else:
                    odds_str = str(am)
            except (TypeError, ValueError):
                odds_str = str(r.get('american_odds', ''))

            red_val = r.get('reduction', 0)
            red_str = f"-{red_val*100:.1f}%" if red_val > 0 else "0.0%"

            dollar_str = f"${r['dollars']:,.0f}" if r['dollars'] > 0 else "$0"
            if r.get('zeroed'):
                dollar_str += " <small style='color:#999;'>(min)</small>"

            if r.get('zeroed'):
                bg = "#f8d7da"
            elif r['dollars'] >= 500:
                bg = "#d4edda"
            elif r['dollars'] >= 200:
                bg = "#fff3cd"
            else:
                bg = "#ffffff"

            rows_html += f"""<tr style="background:{bg};">
                <td style="{cell} padding-left:20px;">{r['label']}</td>
                <td style="{cell} text-align:center;">{r['book']}</td>
                <td style="{cell} text-align:center;">{odds_str}</td>
                <td style="{cell} text-align:center;">{r['sim_prob']*100:.1f}%</td>
                <td style="{cell} text-align:center; font-weight:600;">{r['indep_f']*100:.2f}%</td>
                <td style="{cell} text-align:center; font-weight:600;">{r['corr_f']*100:.2f}%</td>
                <td style="{cell} text-align:center;">{red_str}</td>
                <td style="{cell} text-align:right; font-weight:700;">{dollar_str}</td>
            </tr>"""

    multi = summary['multi_bet_groups']
    avg_red = summary['avg_reduction'] * 100
    cap_str = "BINDING" if summary['portfolio_cap_binding'] else "not binding"

    html = f"""
    <html><body style="font-family:Arial,sans-serif; background:#f8f9fa; padding:20px;">
    <div style="max-width:750px; margin:0 auto; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1);">

    <div style="background:#212529; color:white; padding:16px 20px;">
        <h2 style="margin:0; font-size:18px;">Correlated Kelly Staking</h2>
        <p style="margin:4px 0 0; font-size:13px; color:#adb5bd;">
            {tourney.replace('_', ' ').title()} &mdash; ${bk:,.0f} bankroll @ {kf:.0%} Kelly
        </p>
    </div>

    <table style="border-collapse:collapse; width:100%;">
        <tr>
            <th style="{hdr_style} text-align:left;">Bet</th>
            <th style="{hdr_style}">Book</th>
            <th style="{hdr_style}">Line</th>
            <th style="{hdr_style}">Sim Prob</th>
            <th style="{hdr_style}">Indep f*</th>
            <th style="{hdr_style}">Corr f*</th>
            <th style="{hdr_style}">Reduction</th>
            <th style="{hdr_style} text-align:right;">$ Size</th>
        </tr>
        {rows_html}
    </table>

    <div style="background:#f1f3f5; padding:12px 20px; font-size:12px; color:#495057; border-top:2px solid #dee2e6;">
        <strong>Portfolio:</strong> {summary['total_bets']} bets across {summary['golfer_groups']} golfers
        &nbsp;|&nbsp; {multi} multi-bet groups (avg reduction: {avg_red:.1f}%)
        &nbsp;|&nbsp; Total: ${summary['total_dollars']:,.0f} ({summary['total_pct']*100:.1f}%)
        &nbsp;|&nbsp; Cap: {cap_str}
        {f"&nbsp;|&nbsp; {summary['bets_zeroed']} zeroed (< min)" if summary['bets_zeroed'] else ""}
    </div>

    </div></body></html>"""

    return html
