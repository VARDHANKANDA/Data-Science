import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

os.makedirs('figures_week2', exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.9

# Load Wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['wine_class'] = [f'Cultivar {c+1}' for c in wine.target]
df['cultivar_name'] = df['wine_class'].map({
    'Cultivar 1': 'Cultivar 1 (Barolo style)',
    'Cultivar 2': 'Cultivar 2 (Grignolino style)',
    'Cultivar 3': 'Cultivar 3 (Barbera style)'
})
df['target'] = wine.target

# Color palette: Burgundy, Slate Blue, Amber
PALETTE = {'Cultivar 1': '#800020', 'Cultivar 2': '#2B6CB0', 'Cultivar 3': '#D97706'}
PALETTE_LIST = ['#800020', '#2B6CB0', '#D97706']

# -------------------------------------------------------------
# VISUALIZATION 1: Distribution Overview of Cultivars
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
counts = df['wine_class'].value_counts().sort_index()
percentages = counts / len(df) * 100
bars = ax.bar(counts.index, counts.values, color=PALETTE_LIST, width=0.45, edgecolor='black', linewidth=1.0, zorder=3)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

ax.set_title('Figure 1: Distribution of Wine Cultivars in the Benchmark Dataset', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.set_xlabel('Wine Cultivar Category', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Sample Count (Total N = 178)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylim(0, 85)

for bar, pct, (c_name, cnt) in zip(bars, percentages, counts.items()):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + 2.0, f'{cnt} samples\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2D3748')

# Callout annotation
ax.annotate('Cultivar 2 forms largest share (39.9%)\nBalanced overall sampling',
            xy=(1, counts['Cultivar 2']), xytext=(1.35, 72),
            arrowprops=dict(facecolor='#2B6CB0', shrink=0.08, width=1.5, headwidth=7),
            fontsize=9, fontweight='bold', color='#2B6CB0',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#2B6CB0', lw=1))

plt.tight_layout()
plt.savefig('figures_week2/fig1_cultivar_distribution.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 2: Feature Distribution (Violin + Boxplot 4 Key Features)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), dpi=300)
features = [
    ('alcohol', 'Alcohol Content (% vol)', 'Figure 2A: Alcohol Concentration'),
    ('flavanoids', 'Flavanoids (mg/L)', 'Figure 2B: Flavanoids (Antioxidant Phenols)'),
    ('color_intensity', 'Color Intensity (Abs units)', 'Figure 2C: Color Pigmentation Intensity'),
    ('proline', 'Proline (mg/L Amino Acid)', 'Figure 2D: Proline Concentration')
]

for idx, (feat, label, title) in enumerate(features):
    r, c = idx // 2, idx % 2
    ax = axes[r, c]
    sns.violinplot(data=df, x='wine_class', y=feat, palette=PALETTE, ax=ax, inner='quartile', cut=0, linewidth=1.2)
    sns.stripplot(data=df, x='wine_class', y=feat, color='black', alpha=0.35, size=4, jitter=0.2, ax=ax)
    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#1A365D', pad=8)
    ax.set_xlabel('', fontsize=9)
    ax.set_ylabel(label, fontsize=9.5, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# Specific annotation in 2B for Cultivar 3 collapse
axes[0, 1].annotate('Severe Flavanoid Depletion\n(Mean: 0.78 mg/L)', xy=(2, 0.8), xytext=(1.35, 3.5),
                    arrowprops=dict(facecolor='#D97706', shrink=0.08, width=1.2, headwidth=6),
                    fontsize=8.5, fontweight='bold', color='#D97706',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7', edgecolor='#D97706', lw=1))

# Specific annotation in 2D for Cultivar 1 proline dominance
axes[1, 1].annotate('High Proline Signature\n(Median: 1,095 mg/L)', xy=(0, 1100), xytext=(0.25, 1420),
                    arrowprops=dict(facecolor='#800020', shrink=0.08, width=1.2, headwidth=6),
                    fontsize=8.5, fontweight='bold', color='#800020',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#800020', lw=1))

plt.suptitle('Figure 2: Distribution and Spread of Critical Chemical Markers Across Cultivars', fontsize=12.5, fontweight='bold', color='#1A365D', y=0.99)
plt.tight_layout()
plt.savefig('figures_week2/fig2_feature_distributions.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 3: Bivariate Relationships (Flavanoids vs Total Phenols & Alcohol vs Proline)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)

# 3A: Flavanoids vs Total Phenols
for c_name, color in PALETTE.items():
    sub = df[df['wine_class'] == c_name]
    axes[0].scatter(sub['total_phenols'], sub['flavanoids'], label=c_name, color=color, alpha=0.85, edgecolors='black', linewidth=0.6, s=55)
sns.regplot(data=df, x='total_phenols', y='flavanoids', scatter=False, ax=axes[0], color='#4A5568', line_kws={'linestyle': '--', 'linewidth': 1.5, 'label': 'Global Trend (r = 0.865)'})
axes[0].set_title('Figure 3A: Flavanoids vs. Total Phenols (Antioxidants)', fontsize=10.5, fontweight='bold', color='#1A365D')
axes[0].set_xlabel('Total Phenols (g/L)', fontsize=9.5, fontweight='bold')
axes[0].set_ylabel('Flavanoids (g/L)', fontsize=9.5, fontweight='bold')
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)

axes[0].annotate('Strong Positive Linear\nRelationship (r = 0.865)', xy=(2.8, 3.2), xytext=(1.4, 4.2),
                 arrowprops=dict(facecolor='#2D3748', shrink=0.08, width=1.2, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#2D3748',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7FAFC', edgecolor='#A0AEC0', lw=1))

# 3B: Alcohol vs Proline
for c_name, color in PALETTE.items():
    sub = df[df['wine_class'] == c_name]
    axes[1].scatter(sub['alcohol'], sub['proline'], label=c_name, color=color, alpha=0.85, edgecolors='black', linewidth=0.6, s=55)
axes[1].set_title('Figure 3B: Alcohol vs. Proline Amino Acid Concentration', fontsize=10.5, fontweight='bold', color='#1A365D')
axes[1].set_xlabel('Alcohol Content (% vol)', fontsize=9.5, fontweight='bold')
axes[1].set_ylabel('Proline (mg/L)', fontsize=9.5, fontweight='bold')
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)

axes[1].annotate('Cultivar 1 Cluster:\nHigh Alcohol & High Proline', xy=(13.8, 1150), xytext=(11.5, 1380),
                 arrowprops=dict(facecolor='#800020', shrink=0.08, width=1.2, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#800020',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#800020', lw=1))

plt.suptitle('Figure 3: Bivariate Scatter Plots Revealing Chemical Co-Dependence and Cultivar Clusters', fontsize=12.5, fontweight='bold', color='#1A365D', y=0.99)
plt.tight_layout()
plt.savefig('figures_week2/fig3_bivariate_relationships.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 4: Correlation Heatmap
# -------------------------------------------------------------
plt.figure(figsize=(10.5, 8.5), dpi=300)
corr_cols = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]
corr_clean_names = [
    'Alcohol', 'Malic Acid', 'Ash', 'Ash Alcalinity', 'Magnesium',
    'Total Phenols', 'Flavanoids', 'Nonflav Phenols', 'Proanthocyanins',
    'Color Intensity', 'Hue', 'OD280/OD315', 'Proline'
]
corr_matrix = df[corr_cols].corr()
corr_matrix.columns = corr_clean_names
corr_matrix.index = corr_clean_names

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(220, 15, as_cmap=True)

ax = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='vlag', vmin=-0.8, vmax=1.0,
            square=True, linewidths=0.8, linecolor='white',
            cbar_kws={'shrink': 0.75, 'label': 'Pearson Correlation Coefficient (r)'},
            annot_kws={'size': 8.5, 'weight': 'normal'})
plt.title('Figure 4: Correlation Heatmap of 13 Chemical Constituents', fontsize=12.5, fontweight='bold', pad=15, color='#1A365D')

plt.tight_layout()
plt.savefig('figures_week2/fig4_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 5: PCA 2D Biplot with 95% Confidence Ellipses
# -------------------------------------------------------------
X = df[corr_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['wine_class'] = df['wine_class']

fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

for c_name, color in PALETTE.items():
    sub = df_pca[df_pca['wine_class'] == c_name]
    ax.scatter(sub['PC1'], sub['PC2'], label=f'{c_name}', color=color, alpha=0.85, edgecolors='black', linewidth=0.7, s=65, zorder=4)
    confidence_ellipse(sub['PC1'], sub['PC2'], ax, n_std=2.0, facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5, linestyle='--')

# Plot top loadings
top_features = ['flavanoids', 'total_phenols', 'od280/od315_of_diluted_wines', 'color_intensity', 'alcohol', 'malic_acid']
scale_arrow = 2.8

for idx, feat in enumerate(corr_cols):
    if feat in top_features:
        l_x = pca.components_[0, idx] * scale_arrow
        l_y = pca.components_[1, idx] * scale_arrow
        ax.arrow(0, 0, l_x, l_y, color='#1A365D', alpha=0.85, width=0.03, head_width=0.15, zorder=5)
        clean_f = corr_clean_names[idx]
        ax.text(l_x * 1.15, l_y * 1.15, clean_f, color='#1A365D', fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=6)

ax.set_title('Figure 5: Principal Component Analysis (PCA) 2D Projection with 95% Confidence Ellipses', fontsize=12, fontweight='bold', pad=15, color='#1A365D')
ax.set_xlabel(f'Principal Component 1 ({var_exp[0]:.1f}% Variance Explained - Phenolic & Dilution Profile)', fontsize=10, fontweight='bold')
ax.set_ylabel(f'Principal Component 2 ({var_exp[1]:.1f}% Variance Explained - Alcohol & Pigmentation)', fontsize=10, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
ax.legend(title='Wine Cultivars', frameon=True, facecolor='white', framealpha=0.95, loc='upper left')

# Annotations for clusters
ax.annotate('Cultivar 1:\nHigh Phenolics & Alcohol', xy=(-2.5, -1.5), xytext=(-4.5, -3.2),
            arrowprops=dict(facecolor='#800020', shrink=0.08, width=1.2, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#800020',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F2', edgecolor='#800020', lw=1))

ax.annotate('Cultivar 3:\nHigh Malic Acid & Color\nDepleted Phenolics', xy=(2.8, -1.8), xytext=(3.2, -3.5),
            arrowprops=dict(facecolor='#D97706', shrink=0.08, width=1.2, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#D97706',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7', edgecolor='#D97706', lw=1))

plt.tight_layout()
plt.savefig('figures_week2/fig5_pca_multivariate_separation.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# VISUALIZATION 6: Standardized Chemical Fingerprint Profile
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=300)
sel_features = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'total_phenols', 'flavanoids', 'color_intensity', 'od280/od315_of_diluted_wines', 'proline']
sel_labels = ['Alcohol', 'Malic Acid', 'Alcalinity', 'Phenols', 'Flavanoids', 'Color Int.', 'OD280/OD315', 'Proline']

means_by_class = df.groupby('wine_class')[sel_features].mean()
std_means = (means_by_class - df[sel_features].mean()) / df[sel_features].std()

x_indices = np.arange(len(sel_features))
width = 0.25

for i, (c_name, color) in enumerate(PALETTE.items()):
    ax.bar(x_indices + (i - 1)*width, std_means.loc[c_name], width=width, label=c_name, color=color, edgecolor='black', linewidth=0.7)

ax.set_xticks(x_indices)
ax.set_xticklabels(sel_labels, fontsize=9.5, fontweight='bold')
ax.set_ylabel('Standardized Deviation from Global Mean (Z-Score)', fontsize=10, fontweight='bold')
ax.set_title('Figure 6: Standardized Chemical Profile "Fingerprints" Across Cultivars', fontsize=12, fontweight='bold', pad=15, color='#1A365D')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.9, linestyle='-')
ax.legend(title='Wine Cultivars', frameon=True, facecolor='white', framealpha=0.95)

plt.tight_layout()
plt.savefig('figures_week2/fig6_standardized_fingerprint.png', bbox_inches='tight')
plt.close()

print('All 6 figures generated successfully in figures_week2/ !')
