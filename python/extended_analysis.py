"""
Extended PCMCI+ Analysis - Deeper Exploration

Run this after the main notebook to explore:
1. Different alpha levels (catch weaker signals)
2. Shorter/longer tau_max
3. CMI for nonlinear relationships
4. Rolling window analysis (regime changes)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pcmci

# =============================================================================
# 1. Fetch Data (same as notebook)
# =============================================================================

ASSETS = {
    'SPY': 'US Equities',
    'TLT': 'Treasuries',
    'GLD': 'Gold',
    'UUP': 'US Dollar',
    'EEM': 'EM Equities',
}

end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)

print("Fetching data...")
data = yf.download(list(ASSETS.keys()), start=start_date, end=end_date, progress=False)

# Compute Parkinson volatility
def parkinson_volatility(high, low, window=20):
    log_hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2))
    return np.sqrt(factor * log_hl.rolling(window).mean() * 252)

volatility = pd.DataFrame()
for symbol in ASSETS.keys():
    volatility[symbol] = parkinson_volatility(data['High'][symbol], data['Low'][symbol], 20)
volatility = volatility.dropna()

var_names = list(ASSETS.keys())
vol_matrix = volatility[var_names].values.T

print(f"Data: {vol_matrix.shape[1]} days, {vol_matrix.shape[0]} assets\n")

# =============================================================================
# 2. Compare Different Alpha Levels
# =============================================================================

print("=" * 70)
print("ANALYSIS 1: Different Significance Levels")
print("=" * 70)

for alpha in [0.01, 0.05, 0.10, 0.20]:
    result = pcmci.run_pcmci(vol_matrix, tau_max=5, alpha=alpha, var_names=var_names)
    
    # Count cross-asset links (excluding self-loops)
    cross_links = [l for l in result.significant_links 
                   if l.source_var != l.target_var or l.tau > 0]
    cross_spillovers = [l for l in cross_links if l.source_var != l.target_var]
    
    print(f"\nalpha={alpha:.2f}: {result.n_significant} total links, {len(cross_spillovers)} cross-asset spillovers")
    
    for link in cross_spillovers:
        src = var_names[link.source_var]
        tgt = var_names[link.target_var]
        print(f"  {src}(t-{link.tau}) → {tgt}(t): r={link.val:+.3f}, p={link.pval:.4f}")

# =============================================================================
# 3. Different Lag Windows
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: Different Maximum Lags")
print("=" * 70)

for tau_max in [3, 5, 10, 15]:
    result = pcmci.run_pcmci(vol_matrix, tau_max=tau_max, alpha=0.10, var_names=var_names)
    
    cross_spillovers = [l for l in result.significant_links 
                        if l.source_var != l.target_var]
    
    print(f"\ntau_max={tau_max:2d}: {len(cross_spillovers)} cross-asset links ({result.runtime*1000:.1f}ms)")
    
    for link in sorted(cross_spillovers, key=lambda x: abs(x.val), reverse=True)[:5]:
        src = var_names[link.source_var]
        tgt = var_names[link.target_var]
        print(f"  {src}(t-{link.tau}) → {tgt}(t): r={link.val:+.3f}")

# =============================================================================
# 4. CMI vs Partial Correlation (Nonlinear Detection)
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: Linear (ParCorr) vs Nonlinear (CMI) Dependencies")
print("=" * 70)

vol_arr = volatility.values

# Test all pairs
print("\nPairwise comparisons (contemporaneous):")
print("-" * 60)
print(f"{'Pair':<15} {'ParCorr':>10} {'p-value':>10} {'CMI':>10} {'p-value':>10}")
print("-" * 60)

for i in range(len(var_names)):
    for j in range(i+1, len(var_names)):
        X = vol_arr[:, i]
        Y = vol_arr[:, j]
        
        r, p_parcorr = pcmci.parcorr_test(X, Y)
        cmi_result = pcmci.cmi_test(X, Y, n_perm=100)
        
        pair = f"{var_names[i]}-{var_names[j]}"
        
        # Highlight if CMI detects something parcorr misses
        flag = ""
        if p_parcorr > 0.05 and cmi_result.pvalue < 0.05:
            flag = " *** NONLINEAR!"
        
        print(f"{pair:<15} {r:>+10.3f} {p_parcorr:>10.4f} {cmi_result.cmi:>10.3f} {cmi_result.pvalue:>10.4f}{flag}")

# =============================================================================
# 5. Lead-Lag with CMI (Nonlinear)
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: Lead-Lag Analysis (Linear vs Nonlinear)")
print("=" * 70)

def lead_lag_comparison(source_idx, target_idx, max_lag=10):
    """Compare parcorr and CMI at different lags"""
    src_name = var_names[source_idx]
    tgt_name = var_names[target_idx]
    
    print(f"\n{src_name} → {tgt_name}:")
    print(f"{'Lag':>4} {'ParCorr':>10} {'p':>8} {'CMI':>10} {'p':>8}")
    print("-" * 45)
    
    best_parcorr_lag = 0
    best_parcorr_val = 0
    best_cmi_lag = 0
    best_cmi_val = 0
    
    for lag in range(1, max_lag + 1):
        source = vol_arr[:-lag, source_idx]
        target = vol_arr[lag:, target_idx]
        
        r, p_r = pcmci.parcorr_test(source, target)
        cmi_val = pcmci.mi(source, target)  # Fast, no permutation
        
        if abs(r) > abs(best_parcorr_val):
            best_parcorr_val = r
            best_parcorr_lag = lag
        if cmi_val > best_cmi_val:
            best_cmi_val = cmi_val
            best_cmi_lag = lag
        
        sig = "*" if p_r < 0.05 else " "
        print(f"{lag:>4} {r:>+10.3f} {p_r:>8.4f}{sig} {cmi_val:>10.3f}")
    
    print(f"\nBest ParCorr: lag={best_parcorr_lag}, r={best_parcorr_val:+.3f}")
    print(f"Best CMI:     lag={best_cmi_lag}, MI={best_cmi_val:.3f}")

# Analyze key pairs
lead_lag_comparison(1, 0)  # TLT → SPY
lead_lag_comparison(0, 4)  # SPY → EEM
lead_lag_comparison(1, 3)  # TLT → UUP
lead_lag_comparison(2, 0)  # GLD → SPY

# =============================================================================
# 6. Rolling Window Analysis (Regime Detection)
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 5: Rolling Window - Time-Varying Causality")
print("=" * 70)

window_size = 120  # ~6 months
step = 20  # ~1 month

# Track SPY → EEM relationship over time
spy_eem_strength = []
tlt_spy_strength = []
dates = []

print(f"\nRolling PCMCI+ (window={window_size} days, step={step} days)...")

for start in range(0, vol_matrix.shape[1] - window_size, step):
    end = start + window_size
    window_data = vol_matrix[:, start:end]
    window_date = volatility.index[end - 1]
    
    result = pcmci.run_pcmci(window_data, tau_max=3, alpha=0.20, var_names=var_names)
    
    # Find SPY → EEM link
    spy_eem = 0
    tlt_spy = 0
    for link in result.significant_links:
        if var_names[link.source_var] == 'SPY' and var_names[link.target_var] == 'EEM':
            spy_eem = link.val
        if var_names[link.source_var] == 'TLT' and var_names[link.target_var] == 'SPY':
            tlt_spy = link.val
    
    spy_eem_strength.append(spy_eem)
    tlt_spy_strength.append(tlt_spy)
    dates.append(window_date)

# Plot time-varying causality
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(dates, spy_eem_strength, 'b-', linewidth=2, label='SPY → EEM')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].fill_between(dates, spy_eem_strength, 0, alpha=0.3)
axes[0].set_ylabel('Causal Strength')
axes[0].set_title('Time-Varying Volatility Spillover (Rolling 6-month PCMCI+)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(dates, tlt_spy_strength, 'r-', linewidth=2, label='TLT → SPY')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].fill_between(dates, tlt_spy_strength, 0, alpha=0.3, color='red')
axes[1].set_ylabel('Causal Strength')
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_causality.png', dpi=150)
plt.show()

print("\nSaved: rolling_causality.png")

# =============================================================================
# 7. Focus on the April 2025 Crisis
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 6: April 2025 Volatility Spike - Who Moved First?")
print("=" * 70)

# Find the crisis period (when SPY vol > 25%)
crisis_mask = volatility['SPY'] > 0.25
if crisis_mask.any():
    crisis_start = volatility.index[crisis_mask].min()
    crisis_end = volatility.index[crisis_mask].max()
    
    print(f"\nCrisis period: {crisis_start.date()} to {crisis_end.date()}")
    
    # Get pre-crisis window (30 days before)
    pre_crisis_start = crisis_start - timedelta(days=60)
    pre_crisis_data = volatility.loc[pre_crisis_start:crisis_end]
    
    print(f"Analysis window: {pre_crisis_data.index[0].date()} to {pre_crisis_data.index[-1].date()}")
    print(f"Observations: {len(pre_crisis_data)}")
    
    # Run PCMCI+ on crisis period
    crisis_matrix = pre_crisis_data[var_names].values.T
    
    result = pcmci.run_pcmci(crisis_matrix, tau_max=5, alpha=0.10, var_names=var_names)
    
    print(f"\nCrisis-period causal links:")
    for link in result.significant_links:
        if link.source_var != link.target_var:
            src = var_names[link.source_var]
            tgt = var_names[link.target_var]
            print(f"  {src}(t-{link.tau}) → {tgt}(t): r={link.val:+.3f}, p={link.pval:.4f}")
    
    # Day-by-day analysis around the spike
    print("\n\nDay-by-day volatility around the peak:")
    print("-" * 60)
    
    peak_date = volatility['SPY'].idxmax()
    window_around_peak = volatility.loc[peak_date - timedelta(days=10):peak_date + timedelta(days=5)]
    
    print(window_around_peak.round(3).to_string())
    
else:
    print("No crisis period detected with SPY vol > 25%")

# =============================================================================
# 8. Distance Correlation Matrix
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 7: Distance Correlation Matrix (Detects Any Dependence)")
print("=" * 70)

n_assets = len(var_names)
dcor_matrix = np.zeros((n_assets, n_assets))

for i in range(n_assets):
    for j in range(n_assets):
        if i == j:
            dcor_matrix[i, j] = 1.0
        elif i < j:
            dc = pcmci.dcor(vol_arr[:, i], vol_arr[:, j])
            dcor_matrix[i, j] = dc
            dcor_matrix[j, i] = dc

# Display as heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(dcor_matrix, cmap='RdYlGn', vmin=0, vmax=1)

ax.set_xticks(range(n_assets))
ax.set_yticks(range(n_assets))
ax.set_xticklabels(var_names)
ax.set_yticklabels(var_names)

# Add values
for i in range(n_assets):
    for j in range(n_assets):
        ax.text(j, i, f'{dcor_matrix[i,j]:.2f}', ha='center', va='center', fontsize=12)

ax.set_title('Distance Correlation Matrix (Volatility)')
plt.colorbar(im, label='dCor')
plt.tight_layout()
plt.savefig('dcor_matrix.png', dpi=150)
plt.show()

print("\nSaved: dcor_matrix.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Findings:

1. SPY ↔ EEM: Strong contemporaneous link (r~0.67), no lead-lag
   → They move together, likely common factor (global risk)

2. TLT → UUP at lag 5: Weak but significant
   → Treasury vol predicts dollar vol 5 days later

3. All assets show strong AR(1): ~0.75-0.81
   → Volatility clustering (expected)

4. GLD appears independent from other assets
   → Safe haven behavior confirmed

5. Crisis periods may show different causal structure
   → Consider regime-switching models

Actionable Signals:
- When TLT vol spikes, watch for UUP vol increase in ~5 days
- SPY and EEM are redundant for diversification during stress
- GLD provides true diversification benefit
""")
