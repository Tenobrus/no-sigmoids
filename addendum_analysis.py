"""
Addendum Analysis: Updated METR Time Horizon 1.1 Data
Refits sigmoid and exponential curves to the new METR TH 1.1 dataset.
Uses the same date convention as the original paper (days since 1970-01-01 / 10000).
"""

import numpy as np
import torch
import torch.optim as optim
from datetime import datetime, date, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
from tqdm import tqdm

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def convert_to_continuous_time(date_string, divisor=1e4):
    """Match original paper's date conversion."""
    if isinstance(date_string, str):
        dt = datetime.strptime(date_string, "%Y-%m-%d").date()
    else:
        dt = date_string
    days_since_epoch = int((dt - date(1970, 1, 1)).days)
    return days_since_epoch / divisor


def continuous_time_to_date(ct, divisor=1e4):
    """Convert continuous time back to date."""
    days = int(ct * divisor)
    return date(1970, 1, 1) + timedelta(days=days)


# ============================================================
# Data: TH 1.1 SOTA Models (from METR website)
# ============================================================
th11_sota = [
    ("GPT-2",                   "2019-02-14",   0.039762,   0),
    ("Davinci-002",             "2020-05-28",   0.148793,   0),
    ("GPT-3.5 Turbo",          "2022-03-15",   0.604245,   0),
    ("GPT-4",                   "2023-03-14",   3.524514,   0),
    ("GPT-4 (1106)",            "2023-11-06",   3.609578,   0),
    ("GPT-4o",                  "2024-05-13",   6.404471,   0),
    ("Claude 3.5 Sonnet (Old)", "2024-06-20",  10.771603,   1),
    ("o1-preview",              "2024-09-12",  19.395096,   1),
    ("Claude 3.5 Sonnet (New)", "2024-10-22",  19.776241,   1),
    ("o1",                      "2024-12-05",  37.937543,   1),
    ("Claude 3.7 Sonnet",       "2025-02-24",  59.763762,   1),
    ("o3",                      "2025-04-16", 120.730463,   1),
    ("GPT-5",                   "2025-08-07", 213.954459,   1),
    ("Gemini 3 Pro",            "2025-11-18", 236.654674,   1),
    ("Claude Opus 4.5",         "2025-11-24", 320.422486,   1),
    ("GPT-5.2 (high)",          "2025-12-11", 394.383957,   1),
]

# TH 1.0 SOTA (what original paper used)
th10_sota = [
    ("GPT-2",                   "2019-02-14",   0.039744,   0),
    ("Davinci-002",             "2020-05-28",   0.148821,   0),
    ("GPT-3.5 Turbo",          "2022-03-15",   0.604384,   0),
    ("GPT-4",                   "2023-03-14",   5.364045,   0),
    ("GPT-4 (1106)",            "2023-11-06",   8.557433,   0),
    ("GPT-4o",                  "2024-05-13",   9.170450,   0),
    ("Claude 3.5 Sonnet (Old)", "2024-06-20",  18.216831,   1),
    ("o1-preview",              "2024-09-12",  22.095267,   1),
    ("Claude 3.5 Sonnet (New)", "2024-10-22",  28.983512,   1),
    ("o1-elicited",             "2024-12-05",  39.206576,   1),
    ("Claude 3.7 Sonnet",       "2025-02-24",  54.226342,   1),
    ("o3",                      "2025-04-16",  92.179215,   1),
    ("Grok-4",                  "2025-07-09", 110.075251,   1),
    ("GPT-5",                   "2025-08-07", 137.318539,   1),
    ("GPT-5.1 Codex Max",      "2025-11-19", 161.753349,   1),
]


def fit_sigmoid_torch(t, y, max_iter=500000, lr=1e-3, seed=44):
    """Fit sigmoid: h = b1 * sigmoid(b2 * t + b3). Matches original paper."""
    rng = np.random.default_rng(seed)
    init_params = rng.normal(loc=0.0, scale=1.0, size=3).astype(float)

    t_tensor = torch.tensor(t, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    b1 = torch.nn.Parameter(torch.tensor(float(init_params[0])))
    b2 = torch.nn.Parameter(torch.tensor(float(init_params[1])))
    b3 = torch.nn.Parameter(torch.tensor(float(init_params[2])))
    params = [b1, b2, b3]

    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    for i in tqdm(range(max_iter), desc="Sigmoid fit"):
        optimizer.zero_grad()
        z = torch.clamp(b2 * t_tensor + b3, -50.0, 50.0)
        y_hat = b1 * torch.sigmoid(z)
        loss = loss_fn(y_hat, y_tensor)
        loss.backward()
        optimizer.step()

        grad_norm = torch.norm(
            torch.stack([p.grad.detach().abs().max() for p in params])
        ).item()
        if grad_norm < 1e-4:
            break

    result = {
        'b1': b1.item(),
        'b2': b2.item(),
        'b3': b3.item(),
        'mse': loss.item()
    }

    # Inflection point: where b2*t + b3 = 0, i.e., t = -b3/b2
    if b2.item() != 0:
        inflection_ct = -b3.item() / b2.item()
        try:
            inflection_date = continuous_time_to_date(inflection_ct)
            result['inflection_ct'] = inflection_ct
            result['inflection_date'] = inflection_date.isoformat()
        except (ValueError, OverflowError):
            result['inflection_ct'] = inflection_ct
            result['inflection_date'] = f"ct={inflection_ct:.4f}"

    # Asymptote
    result['asymptote'] = b1.item()

    return result


def sigmoid_predict(t, b1, b2, b3):
    """Predict using sigmoid model."""
    z = np.clip(b2 * t + b3, -50, 50)
    return b1 / (1.0 + np.exp(-z))


def fit_exponential(dates_ct, horizons):
    """Fit log(h) = beta0 + beta1 * d."""
    log_h = np.log(horizons)
    A = np.vstack([dates_ct, np.ones(len(dates_ct))]).T
    result = np.linalg.lstsq(A, log_h, rcond=None)
    beta1, beta0 = result[0]

    pred_log = beta0 + beta1 * dates_ct
    pred = np.exp(pred_log)
    mse = np.mean((pred - horizons) ** 2)
    r2_log = 1 - np.sum((log_h - pred_log)**2) / np.sum((log_h - np.mean(log_h))**2)
    doubling_days = np.log(2) / beta1 * 1e4  # convert back from ct units

    return {
        'beta0': beta0, 'beta1': beta1,
        'mse': mse, 'r2_log': r2_log,
        'doubling_days': doubling_days,
        'doubling_months': doubling_days / 30.44
    }


def run_analysis():
    print("=" * 70)
    print("ADDENDUM ANALYSIS: METR Time Horizon 1.1 Update")
    print("=" * 70)

    # Prepare data in continuous time (matching original paper)
    dates_11_ct = np.array([convert_to_continuous_time(d[1]) for d in th11_sota])
    horizons_11 = np.array([d[2] for d in th11_sota])
    names_11 = [d[0] for d in th11_sota]
    reasoning_11 = [d[3] for d in th11_sota]

    dates_10_ct = np.array([convert_to_continuous_time(d[1]) for d in th10_sota])
    horizons_10 = np.array([d[2] for d in th10_sota])
    names_10 = [d[0] for d in th10_sota]

    print(f"\nTH 1.1: {len(th11_sota)} SOTA models, dates ct: [{dates_11_ct.min():.4f}, {dates_11_ct.max():.4f}]")
    print(f"TH 1.0: {len(th10_sota)} SOTA models")

    # ============================================================
    # 1. Sigmoid Curve Fit
    # ============================================================
    print("\n--- Fitting Sigmoid (TH 1.1) ---")
    sig11 = fit_sigmoid_torch(dates_11_ct.tolist(), horizons_11.tolist(), seed=44)
    print(f"  b1 (asymptote) = {sig11['b1']:.2f} min ({sig11['b1']/60:.1f} hrs)")
    print(f"  b2 (steepness) = {sig11['b2']:.4f}")
    print(f"  b3 (shift)     = {sig11['b3']:.4f}")
    print(f"  MSE = {sig11['mse']:.2f}")
    print(f"  Inflection: {sig11.get('inflection_date', 'N/A')}")

    print("\n--- Fitting Sigmoid (TH 1.0) ---")
    sig10 = fit_sigmoid_torch(dates_10_ct.tolist(), horizons_10.tolist(), seed=44)
    print(f"  b1 (asymptote) = {sig10['b1']:.2f} min ({sig10['b1']/60:.1f} hrs)")
    print(f"  b2 (steepness) = {sig10['b2']:.4f}")
    print(f"  b3 (shift)     = {sig10['b3']:.4f}")
    print(f"  MSE = {sig10['mse']:.2f}")
    print(f"  Inflection: {sig10.get('inflection_date', 'N/A')}")

    # ============================================================
    # 2. Exponential Fit
    # ============================================================
    print("\n--- Fitting Exponential (TH 1.1) ---")
    exp11 = fit_exponential(dates_11_ct, horizons_11)
    print(f"  R² (log): {exp11['r2_log']:.4f}")
    print(f"  Doubling: {exp11['doubling_days']:.0f} days ({exp11['doubling_months']:.1f} months)")
    print(f"  MSE: {exp11['mse']:.2f}")

    print("\n--- Fitting Exponential (TH 1.0) ---")
    exp10 = fit_exponential(dates_10_ct, horizons_10)
    print(f"  R² (log): {exp10['r2_log']:.4f}")
    print(f"  Doubling: {exp10['doubling_days']:.0f} days ({exp10['doubling_months']:.1f} months)")

    # 2023+ only
    mask = dates_11_ct >= convert_to_continuous_time("2023-01-01")
    exp11_2023 = fit_exponential(dates_11_ct[mask], horizons_11[mask])
    print(f"\n--- Exponential (2023+ TH 1.1) ---")
    print(f"  R² (log): {exp11_2023['r2_log']:.4f}")
    print(f"  Doubling: {exp11_2023['doubling_days']:.0f} days ({exp11_2023['doubling_months']:.1f} months)")

    # ============================================================
    # 3. How well do original predictions hold?
    # ============================================================
    print("\n--- Original Paper Predictions vs New Data ---")
    new_models = [d for d in th11_sota if d[0] in ["Gemini 3 Pro", "Claude Opus 4.5", "GPT-5.2 (high)"]]
    for name, dstr, actual, _ in new_models:
        ct = convert_to_continuous_time(dstr)
        sig_pred = sigmoid_predict(ct, sig10['b1'], sig10['b2'], sig10['b3'])
        exp_pred = np.exp(exp10['beta0'] + exp10['beta1'] * ct)
        print(f"  {name} ({dstr}):")
        print(f"    Actual TH 1.1:  {actual:.1f} min ({actual/60:.1f} hrs)")
        print(f"    Sigmoid (TH 1.0 fit): {sig_pred:.1f} min ({sig_pred/60:.1f} hrs)")
        print(f"    Exponential (TH 1.0): {exp_pred:.1f} min ({exp_pred/60:.1f} hrs)")

    # ============================================================
    # 4. Generate Figures
    # ============================================================
    print("\n--- Generating Figures ---")

    # Prediction range
    ct_start = convert_to_continuous_time("2019-01-01")
    ct_end_short = convert_to_continuous_time("2027-06-01")
    ct_end_long = convert_to_continuous_time("2029-01-01")
    ct_pred = np.linspace(ct_start, ct_end_short, 500)
    ct_long = np.linspace(ct_start, ct_end_long, 500)

    def ct_to_datetime(ct_arr):
        return [datetime(1970, 1, 1) + timedelta(days=int(c * 1e4)) for c in ct_arr]

    dates_pred_dt = ct_to_datetime(ct_pred)
    dates_long_dt = ct_to_datetime(ct_long)
    dates_11_dt = [datetime.strptime(d[1], "%Y-%m-%d") for d in th11_sota]
    dates_10_dt = [datetime.strptime(d[1], "%Y-%m-%d") for d in th10_sota]

    # --- Figure 1: Sigmoid comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: TH 1.0 sigmoid + new data
    ax = axes[0]
    y_sig10 = sigmoid_predict(ct_pred, sig10['b1'], sig10['b2'], sig10['b3'])
    y_exp10 = np.exp(exp10['beta0'] + exp10['beta1'] * ct_pred)

    ax.plot(dates_pred_dt, y_sig10, 'g-', lw=2, label=f"Sigmoid (TH 1.0)\ninflection: {sig10.get('inflection_date', 'N/A')}")
    ax.plot(dates_pred_dt, y_exp10, 'r--', lw=1.5, alpha=0.7, label=f"Exponential (TH 1.0)")
    ax.scatter(dates_10_dt, horizons_10, c='black', s=40, zorder=5, label="TH 1.0 data points")

    new_dt = [datetime.strptime(d[1], "%Y-%m-%d") for d in new_models]
    new_h = [d[2] for d in new_models]
    ax.scatter(new_dt, new_h, c='red', s=120, marker='*', zorder=6, label="New TH 1.1 models")
    for m, dt, h in zip(new_models, new_dt, new_h):
        ax.annotate(m[0], (dt, h), textcoords="offset points", xytext=(5, 10), fontsize=7)

    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("50% Time Horizon (minutes)")
    ax.set_title("(a) Original TH 1.0 Fits with New Data Overlaid")
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(500, max(new_h) * 1.2))

    # Panel B: TH 1.1 updated fit
    ax = axes[1]
    y_sig11 = sigmoid_predict(ct_pred, sig11['b1'], sig11['b2'], sig11['b3'])
    y_exp11 = np.exp(exp11['beta0'] + exp11['beta1'] * ct_pred)

    ax.plot(dates_pred_dt, y_sig11, 'g-', lw=2, label=f"Sigmoid (TH 1.1)\ninflection: {sig11.get('inflection_date', 'N/A')}")
    ax.plot(dates_pred_dt, y_exp11, 'r--', lw=1.5, alpha=0.7, label=f"Exponential (TH 1.1)")
    colors = ['blue' if r == 0 else 'darkgreen' for r in reasoning_11]
    ax.scatter(dates_11_dt, horizons_11, c=colors, s=40, zorder=5)
    ax.scatter([], [], c='blue', s=40, label="Base models")
    ax.scatter([], [], c='darkgreen', s=40, label="Reasoning models")

    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("50% Time Horizon (minutes)")
    ax.set_title("(b) Updated TH 1.1 Sigmoid Fit")
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(700, sig11['b1'] * 1.1))

    plt.tight_layout()
    plt.savefig("figures/addendum_fig1_sigmoid.png", dpi=300, bbox_inches='tight')
    print("  Saved: figures/addendum_fig1_sigmoid.png")

    # --- Figure 2: Log-scale and long-term projections ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Log-scale
    ax = axes[0]
    y_sig11_log = sigmoid_predict(ct_pred, sig11['b1'], sig11['b2'], sig11['b3'])
    y_exp11_log = np.exp(exp11['beta0'] + exp11['beta1'] * ct_pred)
    y_sig11_log[y_sig11_log <= 0] = 1e-6

    ax.semilogy(dates_pred_dt, y_sig11_log, 'g-', lw=2, label="Sigmoid (TH 1.1)")
    ax.semilogy(dates_pred_dt, y_exp11_log, 'r--', lw=1.5, label=f"Exponential (R²={exp11['r2_log']:.3f})")
    ax.scatter(dates_11_dt, horizons_11, c=colors, s=40, zorder=5)
    for name, dt, h in zip(names_11, dates_11_dt, horizons_11):
        if h > 50 or h < 0.1:
            ax.annotate(name, (dt, h), textcoords="offset points", xytext=(5, 5), fontsize=6)

    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("50% Time Horizon (minutes, log scale)")
    ax.set_title("(a) Log-Scale: TH 1.1 Data")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # Panel B: Long-term
    ax = axes[1]
    y_sig_long = sigmoid_predict(ct_long, sig11['b1'], sig11['b2'], sig11['b3'])
    y_exp_long = np.exp(exp11['beta0'] + exp11['beta1'] * ct_long)
    y_sig_long[y_sig_long <= 0] = 1e-6

    ax.semilogy(dates_long_dt, y_sig_long, 'g-', lw=2, label="Sigmoid (TH 1.1)")
    ax.semilogy(dates_long_dt, y_exp_long, 'r--', lw=1.5, label="Exponential (TH 1.1)")
    ax.scatter(dates_11_dt, horizons_11, c='black', s=30, zorder=5, label="TH 1.1 data")

    ax.axhline(y=60*24, color='orange', ls=':', alpha=0.7, label="1 day")
    ax.axhline(y=60*24*7, color='purple', ls=':', alpha=0.7, label="1 week")
    ax.axhline(y=60*24*30, color='brown', ls=':', alpha=0.7, label="1 month")

    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("50% Time Horizon (minutes, log scale)")
    ax.set_title("(b) Long-Term Projections (2019-2029)")
    ax.legend(fontsize=7, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.01, 200000)

    plt.tight_layout()
    plt.savefig("figures/addendum_fig2_log.png", dpi=300, bbox_inches='tight')
    print("  Saved: figures/addendum_fig2_log.png")

    # --- Figure 3: Direct data comparison TH 1.0 vs TH 1.1 ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Exponential fit lines
    ax.semilogy(dates_pred_dt, np.exp(exp10['beta0'] + exp10['beta1'] * ct_pred),
                'b--', lw=1.5, alpha=0.6, label=f"Exp (TH 1.0): doubling={exp10['doubling_days']:.0f}d")
    ax.semilogy(dates_pred_dt, np.exp(exp11['beta0'] + exp11['beta1'] * ct_pred),
                'r-', lw=2, alpha=0.8, label=f"Exp (TH 1.1): doubling={exp11['doubling_days']:.0f}d")

    ax.scatter(dates_10_dt, horizons_10, c='blue', s=50, marker='o', zorder=5, label="TH 1.0 SOTA", alpha=0.7)
    ax.scatter(dates_11_dt, horizons_11, c='red', s=50, marker='^', zorder=5, label="TH 1.1 SOTA")

    for name, dt, h in zip(names_11, dates_11_dt, horizons_11):
        if h > 100:
            ax.annotate(name, (dt, h), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Model Release Date", fontsize=12)
    ax.set_ylabel("50% Time Horizon (minutes, log scale)", fontsize=12)
    ax.set_title("TH 1.0 vs TH 1.1: Exponential Trend Comparison", fontsize=14)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/addendum_fig3_comparison.png", dpi=300, bbox_inches='tight')
    print("  Saved: figures/addendum_fig3_comparison.png")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nSigmoid Inflection Points:")
    print(f"  Original paper (TH 1.0): 2025-06-06")
    print(f"  Our TH 1.0 refit:        {sig10.get('inflection_date', 'N/A')}")
    print(f"  New TH 1.1 fit:          {sig11.get('inflection_date', 'N/A')}")

    print(f"\nSigmoid Asymptote:")
    print(f"  TH 1.0: {sig10['b1']:.1f} min ({sig10['b1']/60:.1f} hrs)")
    print(f"  TH 1.1: {sig11['b1']:.1f} min ({sig11['b1']/60:.1f} hrs)")

    print(f"\nExponential Doubling Times:")
    print(f"  TH 1.0 (all):   {exp10['doubling_days']:.0f} days ({exp10['doubling_months']:.1f} months)")
    print(f"  TH 1.1 (all):   {exp11['doubling_days']:.0f} days ({exp11['doubling_months']:.1f} months)")
    print(f"  TH 1.1 (2023+): {exp11_2023['doubling_days']:.0f} days ({exp11_2023['doubling_months']:.1f} months)")

    print(f"\nGoodness of Fit:")
    print(f"  Sigmoid MSE  (TH 1.0): {sig10['mse']:.2f}")
    print(f"  Sigmoid MSE  (TH 1.1): {sig11['mse']:.2f}")
    print(f"  Exp MSE      (TH 1.0): {exp10['mse']:.2f}")
    print(f"  Exp MSE      (TH 1.1): {exp11['mse']:.2f}")
    print(f"  Exp R² log   (TH 1.0): {exp10['r2_log']:.4f}")
    print(f"  Exp R² log   (TH 1.1): {exp11['r2_log']:.4f}")

    # Save
    results = {
        'sigmoid_th11': {k: v for k, v in sig11.items()},
        'sigmoid_th10': {k: v for k, v in sig10.items()},
        'exponential_th11': exp11,
        'exponential_th10': exp10,
        'exponential_2023_th11': exp11_2023,
    }
    with open("results/addendum_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: results/addendum_results.json")

    return results


if __name__ == "__main__":
    results = run_analysis()
