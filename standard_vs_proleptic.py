import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Setup directory
folder_path = os.path.join("data", "standard_vs_proleptic")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 2. Data Preparation
data = {
    "standard": [
        "X",
        "X",
        40.13,
        33.19,
        "X",
        "X",
        "X",
        46.54,
        "X",
        "X",
        "X",
        "X",
        "X",
        28.58,
        43.33,
        "X",
        "X",
        "X",
        "X",
        36.05,
        "X",
        "X",
        29.67,
        35.76,
        "X",
        "X",
        "X",
        "X",
        "X",
        43.91,
    ],
    "proleptic": [
        23.17,
        "X",
        22.64,
        "X",
        19.49,
        33.78,
        39.08,
        "X",
        "X",
        19.65,
        34.91,
        "X",
        30.16,
        26.52,
        23.36,
        30.98,
        "X",
        34.62,
        25.96,
        "X",
        "X",
        30.16,
        "X",
        28.63,
        33.51,
        23.38,
        19.78,
        "X",
        27.6,
        "X",
    ],
}


def get_metrics(times):
    successes = [t for t in times if isinstance(t, (int, float))]
    failures = times.count("X")
    return successes, failures


std_times, std_fail = get_metrics(data["standard"])
pro_times, pro_fail = get_metrics(data["proleptic"])

# 3. Print Descriptive Statistics
stats_df = pd.DataFrame(
    {
        "Policy": ["Standard", "Proleptic"],
        "Successes": [len(std_times), len(pro_times)],
        "Failures": [std_fail, pro_fail],
        "Success Rate (%)": [
            len(std_times) / len(data["standard"]) * 100,
            len(pro_times) / len(data["proleptic"]) * 100,
        ],
        "Mean Time (s)": [np.mean(std_times), np.mean(pro_times)],
        "Median Time (s)": [np.median(std_times), np.median(pro_times)],
        "Std Dev (s)": [np.std(std_times), np.std(pro_times)],
    }
)
print("--- Summary Statistics ---")
print(stats_df.to_string(index=False))

# Calculate improvements
success_rate_improvement = (
    (len(pro_times) / len(data["proleptic"]) - len(std_times) / len(data["standard"]))
    / (len(std_times) / len(data["standard"]))
    * 100
)
time_improvement = (np.mean(std_times) - np.mean(pro_times)) / np.mean(std_times) * 100

print(f"\n--- Performance Improvements (Proleptic vs Standard) ---")
print(
    f"Success Rate Improvement: +{success_rate_improvement:.2f}% (relative improvement)"
)
print(f"Execution Time Improvement: -{time_improvement:.2f}% (faster)")

# 4. Statistical Tests
_, p_success = stats.fisher_exact(
    [[len(std_times), std_fail], [len(pro_times), pro_fail]]
)
_, p_time = stats.mannwhitneyu(std_times, pro_times, alternative="two-sided")

print(f"\n--- Statistical Significance ---")
print(f"Success Rate P-value (Fisher): {p_success:.4f}")
print(f"Execution Time P-value (Mann-Whitney U): {p_time:.4f}")

# 5. Create Comprehensive Visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Success Rate Comparison (Percentage)
ax1 = fig.add_subplot(gs[0, :2])
success_rates = [
    len(std_times) / len(data["standard"]) * 100,
    len(pro_times) / len(data["proleptic"]) * 100,
]
bars = ax1.bar(
    ["Standard", "Proleptic"],
    success_rates,
    color=["#3498db", "#e67e22"],
    alpha=0.8,
    edgecolor="black",
    linewidth=2,
)
ax1.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
ax1.set_title("Success Rate Comparison", fontsize=14, fontweight="bold")
ax1.set_ylim(0, 100)
ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
ax1.grid(axis="y", alpha=0.3)

# Add percentage labels on bars
for i, (bar, rate) in enumerate(zip(bars, success_rates)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 2,
        f'{rate:.1f}%\n({[len(std_times), len(pro_times)][i]}/{len(data["standard"])})',
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

# Plot 2: Stacked Count Visualization
ax2 = fig.add_subplot(gs[0, 2])
plot_data = pd.DataFrame(
    {"Success": [len(std_times), len(pro_times)], "Failure": [std_fail, pro_fail]},
    index=["Standard", "Proleptic"],
)
plot_data.plot(
    kind="bar",
    stacked=True,
    ax=ax2,
    color=["#2ecc71", "#e74c3c"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)
ax2.set_title("Trial Outcomes\n(n=30 each)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Count", fontsize=11)
ax2.set_xlabel("")
ax2.tick_params(axis="x", rotation=0)
ax2.legend(title="Outcome", loc="upper right")

# Add count labels
for i, (s, f) in enumerate(zip(plot_data["Success"], plot_data["Failure"])):
    ax2.text(
        i,
        s / 2,
        str(s),
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11,
    )
    ax2.text(
        i,
        s + f / 2,
        str(f),
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11,
    )

# Plot 3: Box Plot with Individual Points
ax3 = fig.add_subplot(gs[1, :])
time_data = pd.DataFrame(
    {
        "Time": std_times + pro_times,
        "Policy": ["Standard"] * len(std_times) + ["Proleptic"] * len(pro_times),
    }
)
box_parts = ax3.boxplot(
    [std_times, pro_times],
    tick_labels=["Standard", "Proleptic"],
    patch_artist=True,
    widths=0.6,
    boxprops=dict(facecolor="lightblue", alpha=0.7, linewidth=2),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)

# Color boxes differently
box_parts["boxes"][0].set_facecolor("#3498db")
box_parts["boxes"][1].set_facecolor("#e67e22")

# Overlay strip plot
for i, times in enumerate([std_times, pro_times]):
    x = np.random.normal(i + 1, 0.04, size=len(times))
    ax3.scatter(
        x, times, alpha=0.6, s=80, color="black", edgecolors="white", linewidths=1
    )

ax3.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight="bold")
ax3.set_title(
    "Execution Time Distribution (Successful Trials Only)",
    fontsize=14,
    fontweight="bold",
)
ax3.grid(axis="y", alpha=0.3)

# Add mean markers
means = [np.mean(std_times), np.mean(pro_times)]
ax3.scatter(
    [1, 2],
    means,
    color="red",
    s=200,
    marker="D",
    edgecolors="black",
    linewidths=2,
    label="Mean",
    zorder=5,
)
ax3.legend()

# Plot 4: Kernel Density Estimate
ax4 = fig.add_subplot(gs[2, :2])
sns.kdeplot(
    std_times,
    label="Standard",
    fill=True,
    color="#3498db",
    alpha=0.5,
    ax=ax4,
    linewidth=2,
)
sns.kdeplot(
    pro_times,
    label="Proleptic",
    fill=True,
    color="#e67e22",
    alpha=0.5,
    ax=ax4,
    linewidth=2,
)
ax4.axvline(
    np.mean(std_times),
    color="#3498db",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Std Mean: {np.mean(std_times):.1f}s",
)
ax4.axvline(
    np.mean(pro_times),
    color="#e67e22",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Pro Mean: {np.mean(pro_times):.1f}s",
)
ax4.set_title("Execution Time Probability Density", fontsize=14, fontweight="bold")
ax4.set_xlabel("Time (seconds)", fontsize=12)
ax4.set_ylabel("Density", fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Summary Statistics Table
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis("tight")
ax5.axis("off")

table_data = [
    ["Metric", "Standard", "Proleptic"],
    ["Success Rate", f"{success_rates[0]:.1f}%", f"{success_rates[1]:.1f}%"],
    ["Mean Time", f"{np.mean(std_times):.2f}s", f"{np.mean(pro_times):.2f}s"],
    ["Median Time", f"{np.median(std_times):.2f}s", f"{np.median(pro_times):.2f}s"],
    ["Std Dev", f"{np.std(std_times):.2f}s", f"{np.std(pro_times):.2f}s"],
    ["Min Time", f"{np.min(std_times):.2f}s", f"{np.min(pro_times):.2f}s"],
    ["Max Time", f"{np.max(std_times):.2f}s", f"{np.max(pro_times):.2f}s"],
]

table = ax5.table(
    cellText=table_data, cellLoc="center", loc="center", colWidths=[0.35, 0.325, 0.325]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor("#34495e")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#ecf0f1")

ax5.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

plt.savefig(
    os.path.join(folder_path, "comprehensive_policy_comparison.png"),
    dpi=300,
    bbox_inches="tight",
)
print(f"\nComprehensive comparison plot saved to {folder_path}")

# Additional focused comparison plot
fig2, (ax6, ax7) = plt.subplots(1, 2, figsize=(14, 6))

# Success vs Failure counts side by side
categories = ["Standard", "Proleptic"]
success_counts = [len(std_times), len(pro_times)]
failure_counts = [std_fail, pro_fail]

x = np.arange(len(categories))
width = 0.35

bars1 = ax6.bar(
    x - width / 2,
    success_counts,
    width,
    label="Success",
    color="#2ecc71",
    alpha=0.8,
    edgecolor="black",
    linewidth=2,
)
bars2 = ax6.bar(
    x + width / 2,
    failure_counts,
    width,
    label="Failure",
    color="#e74c3c",
    alpha=0.8,
    edgecolor="black",
    linewidth=2,
)

ax6.set_ylabel("Number of Trials", fontsize=12, fontweight="bold")
ax6.set_title("Success vs Failure Counts", fontsize=14, fontweight="bold")
ax6.set_xticks(x)
ax6.set_xticklabels(categories)
ax6.legend()
ax6.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

# Mean execution time comparison
means = [np.mean(std_times), np.mean(pro_times)]
bars = ax7.bar(
    categories,
    means,
    color=["#3498db", "#e67e22"],
    alpha=0.8,
    edgecolor="black",
    linewidth=2,
)
ax7.set_ylabel("Mean Execution Time (seconds)", fontsize=12, fontweight="bold")
ax7.set_title(
    "Average Execution Time\n(Successful Trials Only)", fontsize=14, fontweight="bold"
)
ax7.grid(axis="y", alpha=0.3)

# Add error bars (standard deviation)
ax7.errorbar(
    categories,
    means,
    yerr=[np.std(std_times), np.std(pro_times)],
    fmt="none",
    ecolor="black",
    capsize=10,
    capthick=2,
    alpha=0.7,
)

# Add value labels
for i, (bar, mean) in enumerate(zip(bars, means)):
    height = bar.get_height()
    ax7.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{mean:.2f}s",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

plt.tight_layout()
plt.savefig(
    os.path.join(folder_path, "focused_comparison.png"), dpi=300, bbox_inches="tight"
)

print(f"Focused comparison plot saved to {folder_path}")
print("\n=== Analysis Complete ===")
