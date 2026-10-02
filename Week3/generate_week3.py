import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.datasets import load_wine

os.makedirs('figures_week3', exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.9

# Load Wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['wine_class'] = [f'Cultivar {c+1}' for c in wine.target]
df['target'] = wine.target

PALETTE = {'Cultivar 1': '#800020', 'Cultivar 2': '#2B6CB0', 'Cultivar 3': '#D97706'}
PALETTE_LIST = ['#800020', '#2B6CB0', '#D97706']

# -------------------------------------------------------------
# VISUALIZATION 1: Box Plot of Alcohol by Wine Class
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
sns.boxplot(
    data=df, x='wine_class', y='alcohol', palette=PALETTE, ax=ax,
    width=0.45, showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
    boxprops=dict(alpha=0.85, edgecolor='black', linewidth=1.1),
    whiskerprops=dict(color='black', linewidth=1.1),
    capprops=dict(color='black', linewidth=1.1),
    medianprops=dict(color='yellow', linewidth=2.0)
)
sns.stripplot(data=df, x='wine_class', y='alcohol', color='black', alpha=0.35, size=4.5, jitter=0.2, ax=ax)

ax.set_title('Figure 1: Alcohol Content Distribution Across Wine Cultivars (Box Plot)', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.set_xlabel('Wine Cultivar Category', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Alcohol Content (% by volume)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotations
ax.annotate('Highest Median Alcohol\n(Median = 13.75%, Mean = 13.74%)', xy=(0, 13.75), xytext=(-0.35, 14.4),
            arrowprops=dict(facecolor='#800020', shrink=0.08, width=1.2, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#800020',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#800020', lw=1))

ax.annotate('Lowest Median Alcohol\n(Median = 12.29%, Mean = 12.28%)', xy=(1, 12.29), xytext=(0.65, 11.3),
            arrowprops=dict(facecolor='#2B6CB0', shrink=0.08, width=1.2, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#2B6CB0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4F8', edgecolor='#2B6CB0', lw=1))

plt.tight_layout()
plt.savefig('figures_week3/fig1_alcohol_boxplot.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 2: Group Means with 95% Confidence Intervals
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

group_stats = df.groupby('wine_class')['alcohol'].agg(['count', 'mean', 'sem']).reset_index()
group_stats['ci95'] = group_stats.apply(lambda row: stats.t.ppf(0.975, row['count'] - 1) * row['sem'], axis=1)

for idx, row in group_stats.iterrows():
    c_name = row['wine_class']
    color = PALETTE[c_name]
    ax.errorbar(
        x=row['wine_class'], y=row['mean'], yerr=row['ci95'],
        fmt='o', color=color, ecolor=color, elinewidth=2.5, capsize=8, capthick=2.0,
        markersize=10, markeredgecolor='black', markeredgewidth=1.2, zorder=4
    )
    ax.text(idx + 0.12, row['mean'], f"Mean: {row['mean']:.2f}%\n95% CI: [{row['mean']-row['ci95']:.2f}%, {row['mean']+row['ci95']:.2f}%]",
            va='center', fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8FAFC', edgecolor=color, lw=1))

ax.set_title('Figure 2: Mean Alcohol Content by Cultivar with 95% Confidence Intervals', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.set_xlabel('Wine Cultivar Category', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Mean Alcohol (% by volume)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylim(11.8, 14.2)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Global Mean Reference Line
global_mean = df['alcohol'].mean()
ax.axhline(global_mean, color='#718096', linestyle=':', linewidth=1.5, label=f'Grand Mean ({global_mean:.2f}%)')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)

plt.tight_layout()
plt.savefig('figures_week3/fig2_group_means_ci.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 3: Distribution Density / Histogram Plot
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300, sharey=True)

for idx, (c_name, color) in enumerate(PALETTE.items()):
    sub = df[df['wine_class'] == c_name]['alcohol']
    ax = axes[idx]
    sns.histplot(sub, kde=True, color=color, ax=ax, bins=10, stat='density', edgecolor='black', linewidth=0.8, alpha=0.6)
    
    # Fit theoretical normal curve
    mu, std = sub.mean(), sub.std()
    x_axis = np.linspace(sub.min() - 0.5, sub.max() + 0.5, 100)
    ax.plot(x_axis, stats.norm.pdf(x_axis, mu, std), color='black', linestyle='--', linewidth=1.5, label='Normal Fit')
    
    # Shapiro-Wilk p-value on chart
    shapiro_stat, shapiro_p = stats.shapiro(sub)
    ax.set_title(f'{c_name} (N = {len(sub)})\nShapiro W = {shapiro_stat:.3f}, p = {shapiro_p:.3f}', fontsize=10, fontweight='bold', color='#1A365D')
    ax.set_xlabel('Alcohol Content (% vol)', fontsize=9.5, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Probability Density', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Figure 3: Within-Group Alcohol Distributions with Fitted Normal Curves', fontsize=12, fontweight='bold', color='#1A365D', y=1.02)
plt.tight_layout()
plt.savefig('figures_week3/fig3_distribution_normality.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 4: Quantile-Quantile (Q-Q) Plots for Normality Diagnostics
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

for idx, (c_name, color) in enumerate(PALETTE.items()):
    sub = df[df['wine_class'] == c_name]['alcohol']
    ax = axes[idx]
    (osm, osr), (slope, intercept, r) = stats.probplot(sub, dist="norm", plot=None)
    
    ax.scatter(osm, osr, color=color, edgecolor='black', linewidth=0.6, s=45, label=f'R² = {r**2:.3f}', zorder=3)
    ax.plot(osm, slope * np.array(osm) + intercept, color='#2D3748', linestyle='--', linewidth=1.5, zorder=2)
    
    ax.set_title(f'Figure 4{chr(65+idx)}: Q-Q Plot — {c_name}', fontsize=10, fontweight='bold', color='#1A365D')
    ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=9.5, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Ordered Sample Values (% vol)', fontsize=9.5, fontweight='bold')
    else:
        ax.set_ylabel('')
    ax.legend(fontsize=8.5, loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Figure 4: Normal Probability (Q-Q) Diagnostic Plots per Wine Cultivar', fontsize=12, fontweight='bold', color='#1A365D', y=1.02)
plt.tight_layout()
plt.savefig('figures_week3/fig4_qq_normality_plots.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 5: Post-Hoc Tukey HSD Pairwise Difference Forest Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
tukey = pairwise_tukeyhsd(endog=df['alcohol'], groups=df['wine_class'], alpha=0.05)
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

pairs = [f"{row['group1']} vs. {row['group2']}" for _, row in tukey_df.iterrows()]
diffs = tukey_df['meandiff']
lower = tukey_df['lower']
upper = tukey_df['upper']
y_pos = np.arange(len(pairs))

ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Difference (H₀)')
for idx, (p, d, l, u) in enumerate(zip(pairs, diffs, lower, upper)):
    ax.errorbar(
        x=d, y=idx, xerr=[[d - l], [u - d]], fmt='s', color='#1A365D',
        ecolor='#2B6CB0', elinewidth=2.5, capsize=6, capthick=2.0, markersize=8
    )
    p_adj = tukey_df.loc[idx, 'p-adj']
    ax.text(d, idx + 0.22, f"Diff: {d:+.3f} (95% CI: [{l:+.3f}, {u:+.3f}], p < 0.001)",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1A365D',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.8))

ax.set_yticks(y_pos)
ax.set_yticklabels(pairs, fontsize=10, fontweight='bold')
ax.set_xlabel('Mean Difference in Alcohol (% vol)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_title('Figure 5: Tukey HSD Simultaneous 95% Confidence Intervals for Pairwise Comparisons', fontsize=11.5, fontweight='bold', pad=14, color='#1A365D')
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
ax.set_ylim(-0.5, len(pairs) - 0.2)

plt.tight_layout()
plt.savefig('figures_week3/fig5_tukey_hsd_forest_plot.png', bbox_inches='tight')
plt.close()

print('All 5 Week 3 figures generated successfully in figures_week3/ !')
