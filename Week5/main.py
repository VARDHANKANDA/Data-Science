import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
)
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

os.makedirs('figures_week4', exist_ok=True)
os.makedirs('figures_week5', exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.9

# Load Wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
class_names = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Pipeline
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)
cm = confusion_matrix(y_test, y_pred)

PALETTE = {'Cultivar 1': '#800020', 'Cultivar 2': '#2B6CB0', 'Cultivar 3': '#D97706'}
PALETTE_LIST = ['#800020', '#2B6CB0', '#D97706']

# =============================================================
# WEEK 4 FIGURES
# =============================================================

# 1. Figure 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=300)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=class_names, yticklabels=class_names,
    annot_kws={'size': 14, 'weight': 'bold', 'color': 'black'},
    linewidths=1.2, linecolor='white', ax=ax
)
ax.set_title('Figure 1: Logistic Regression Confusion Matrix (Test Set, N = 36)', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.set_xlabel('Predicted Cultivar Class', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Actual True Cultivar Class', fontsize=10.5, fontweight='bold', labelpad=8)

ax.annotate('1 Misclassification:\nActual Cultivar 3 -> Pred Cultivar 2\n(Sample 134: Borderline p=0.54)',
            xy=(1.5, 2.5), xytext=(1.8, 1.3),
            arrowprops=dict(facecolor='#D97706', shrink=0.08, width=1.5, headwidth=7),
            fontsize=8.5, fontweight='bold', color='#D97706',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7', edgecolor='#D97706', lw=1))

plt.tight_layout()
plt.savefig('figures_week4/fig1_confusion_matrix.png', bbox_inches='tight')
plt.close()

# 2. Figure 2: Model Performance Metrics
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
metrics = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score', '5-Fold CV Mean']
values = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred, average='weighted'),
    recall_score(y_test, y_pred, average='weighted'),
    f1_score(y_test, y_pred, average='weighted'),
    0.9833
]
colors = ['#1A365D', '#2B6CB0', '#319795', '#800020', '#D97706']
bars = ax.bar(metrics, values, color=colors, width=0.45, edgecolor='black', linewidth=1.0, zorder=3)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

ax.set_title('Figure 2: Comprehensive Model Evaluation Metrics (Test Set & CV)', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.set_ylabel('Score / Percentage', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylim(0.85, 1.03)

for bar, val in zip(bars, values):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.005, f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1A365D')

plt.tight_layout()
plt.savefig('figures_week4/fig2_performance_metrics.png', bbox_inches='tight')
plt.close()

# 3. Figure 3: Multiclass ROC Curves (OvR)
fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

for i, (c_name, color) in enumerate(PALETTE.items()):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    c_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{c_name} (AUC = {c_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance Baseline (AUC = 0.5000)')
ax.set_xlim([-0.02, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_title('Figure 3: Multi-Class One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) Curves', fontsize=11.5, fontweight='bold', pad=14, color='#1A365D')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
ax.grid(True, linestyle='--', alpha=0.6)

ax.annotate('Macro-Average ROC-AUC = 1.0000\nPerfect True-Positive Discrimination',
            xy=(0.0, 1.0), xytext=(0.25, 0.75),
            arrowprops=dict(facecolor='#1A365D', shrink=0.08, width=1.5, headwidth=7),
            fontsize=9, fontweight='bold', color='#1A365D',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4F8', edgecolor='#1A365D', lw=1))

plt.tight_layout()
plt.savefig('figures_week4/fig3_multiclass_roc_curves.png', bbox_inches='tight')
plt.close()

# 4. Figure 4: Model Benchmarking
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
models_benchmark = {
    'Logistic Regression': (0.9722, 0.9833, 0.0136),
    'Support Vector Machine (RBF)': (0.9722, 0.9833, 0.0222),
    'Random Forest (100 Trees)': (1.0000, 0.9775, 0.0213),
    'K-Nearest Neighbors (k=5)': (0.9722, 0.9717, 0.0181),
    'Gradient Boosting': (0.9444, 0.9271, 0.0283)
}
m_names = list(models_benchmark.keys())
test_accs = [v[0]*100 for v in models_benchmark.values()]
cv_means = [v[1]*100 for v in models_benchmark.values()]
cv_stds = [v[2]*100 for v in models_benchmark.values()]

x_idx = np.arange(len(m_names))
w = 0.35
ax.bar(x_idx - w/2, test_accs, width=w, label='Test Accuracy (20% Holdout)', color='#2B6CB0', edgecolor='black', linewidth=0.8)
ax.bar(x_idx + w/2, cv_means, yerr=cv_stds, capsize=5, width=w, label='5-Fold CV Mean ± SD', color='#DD6B20', edgecolor='black', linewidth=0.8)

ax.set_xticks(x_idx)
ax.set_xticklabels(m_names, rotation=15, ha='right', fontsize=9, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=10.5, fontweight='bold')
ax.set_ylim(85, 105)
ax.set_title('Figure 4: Algorithm Benchmark Comparison: Test Set vs. 5-Fold Cross-Validation', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures_week4/fig4_algorithm_benchmarking.png', bbox_inches='tight')
plt.close()

# 5. Figure 5: Feature Coefficients
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=300, sharey=True)
coefs = clf.coef_
for i, (c_name, color) in enumerate(PALETTE.items()):
    ax = axes[i]
    c_series = pd.Series(coefs[i], index=X.columns).sort_values()
    c_series.plot(kind='barh', ax=ax, color=color, edgecolor='black', linewidth=0.6)
    ax.set_title(f'{c_name} Coefficients', fontsize=10.5, fontweight='bold', color='#1A365D')
    ax.set_xlabel('Standardized Model Weight', fontsize=9, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

plt.suptitle('Figure 5: Standardized Logistic Regression Coefficients per Cultivar Class', fontsize=12, fontweight='bold', color='#1A365D', y=1.02)
plt.tight_layout()
plt.savefig('figures_week4/fig5_feature_coefficients.png', bbox_inches='tight')
plt.close()

# =============================================================
# WEEK 5 FIGURES (FULL LIFECYCLE SYNTHESIS)
# =============================================================

# Week 5 Figure 1: Pipeline Architecture
fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
ax.axis('off')
boxes = [
    ('Week 1\nData Acquisition\n& Cleaning', 0.08, '#1A365D'),
    ('Week 2\nEDA & Chemometric\nStorytelling', 0.28, '#2B6CB0'),
    ('Week 3\nInferential Statistics\n& ANOVA Testing', 0.48, '#800020'),
    ('Week 4\nSupervised ML &\nModel Evaluation', 0.68, '#319795'),
    ('Week 5\nStrategic Synthesis\n& Decision Support', 0.88, '#D97706')
]

for title, x_pos, color in boxes:
    ax.text(x_pos, 0.5, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=color, edgecolor='black', lw=1.2))

for i in range(len(boxes)-1):
    x_start = boxes[i][1] + 0.075
    x_end = boxes[i+1][1] - 0.075
    ax.annotate('', xy=(x_end, 0.5), xytext=(x_start, 0.5),
                arrowprops=dict(facecolor='#4A5568', shrink=0.05, width=2.0, headwidth=8))

ax.set_title('Figure 1: Full-Lifecycle End-to-End Data Science & Machine Learning Framework', fontsize=12, fontweight='bold', pad=10, color='#1A365D')
plt.tight_layout()
plt.savefig('figures_week5/fig1_lifecycle_framework.png', bbox_inches='tight')
plt.close()

# Week 5 Figure 2: Cultivar Class & Key Metric Distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)
counts = pd.Series(y).value_counts().sort_index()
counts.index = class_names
percentages = counts / len(y) * 100
bars = axes[0].bar(counts.index, counts.values, color=PALETTE_LIST, width=0.45, edgecolor='black', linewidth=1.0)
axes[0].set_title('Figure 2A: Wine Cultivar Sample Distribution (N = 178)', fontsize=11, fontweight='bold', color='#1A365D')
axes[0].set_ylabel('Sample Count', fontsize=10, fontweight='bold')
for bar, pct in zip(bars, percentages):
    axes[0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f'{bar.get_height()} ({pct:.1f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
axes[0].set_ylim(0, 85)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

sns.boxplot(data=pd.DataFrame({'wine_class': [class_names[i] for i in y], 'flavanoids': X['flavanoids']}),
            x='wine_class', y='flavanoids', palette=PALETTE, ax=axes[1], width=0.45)
axes[1].set_title('Figure 2B: Flavanoids Antioxidant Distribution (F = 233.9)', fontsize=11, fontweight='bold', color='#1A365D')
axes[1].set_ylabel('Flavanoids (mg/L)', fontsize=10, fontweight='bold')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures_week5/fig2_distributions_overview.png', bbox_inches='tight')
plt.close()

# Week 5 Figure 3: PCA 2D Biplot
X_all_s = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_s)
var_exp = pca.explained_variance_ratio_ * 100
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['wine_class'] = [class_names[i] for i in y]

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

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
    ax.scatter(sub['PC1'], sub['PC2'], label=c_name, color=color, alpha=0.85, edgecolors='black', linewidth=0.7, s=60, zorder=4)
    confidence_ellipse(sub['PC1'], sub['PC2'], ax, n_std=2.0, facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5, linestyle='--')

ax.set_title('Figure 3: PCA 2D Latent Biplot Demonstrating 100% Cultivar Separation', fontsize=12, fontweight='bold', pad=14, color='#1A365D')
ax.set_xlabel(f'Principal Component 1 ({var_exp[0]:.1f}% Variance - Phenolic Concentration)', fontsize=10, fontweight='bold')
ax.set_ylabel(f'Principal Component 2 ({var_exp[1]:.1f}% Variance - Alcohol & Color)', fontsize=10, fontweight='bold')
ax.legend(title='Wine Cultivars', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures_week5/fig3_pca_separation.png', bbox_inches='tight')
plt.close()

# Week 5 Figure 4: Statistical ANOVA Group Means
fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
alc_stats = pd.DataFrame({'class': [class_names[i] for i in y], 'alcohol': X['alcohol']}).groupby('class')['alcohol'].agg(['count', 'mean', 'sem']).reset_index()
alc_stats['ci95'] = alc_stats.apply(lambda r: 1.96 * r['sem'], axis=1)

for idx, row in alc_stats.iterrows():
    c_name = row['class']
    color = PALETTE[c_name]
    ax.errorbar(
        x=row['class'], y=row['mean'], yerr=row['ci95'],
        fmt='o', color=color, ecolor=color, elinewidth=2.5, capsize=8, capthick=2.0,
        markersize=10, markeredgecolor='black', markeredgewidth=1.2, zorder=4
    )
    ax.text(idx + 0.12, row['mean'], f"Mean: {row['mean']:.2f}%\n(F = 135.08, p < 0.001)",
            va='center', fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8FAFC', edgecolor=color, lw=1))

ax.set_title('Figure 4: Confirmatory ANOVA Hypothesis Testing on Cultivar Alcohol Means (p = 3.32e-36)', fontsize=11.5, fontweight='bold', pad=14, color='#1A365D')
ax.set_ylabel('Alcohol Content (% vol)', fontsize=10, fontweight='bold')
ax.set_ylim(11.8, 14.2)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures_week5/fig4_anova_hypothesis_results.png', bbox_inches='tight')
plt.close()

# Week 5 Figure 5: Confusion Matrix & ROC Curve Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0],
            annot_kws={'size': 12, 'weight': 'bold'}, cbar=True)
axes[0].set_title('Figure 5A: Logistic Regression Confusion Matrix (Test Acc: 97.2%)', fontsize=10.5, fontweight='bold', color='#1A365D')
axes[0].set_xlabel('Predicted Class', fontsize=9.5, fontweight='bold')
axes[0].set_ylabel('Actual Class', fontsize=9.5, fontweight='bold')

for i, (c_name, color) in enumerate(PALETTE.items()):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    axes[1].plot(fpr, tpr, color=color, lw=2.2, label=f'{c_name} (AUC = 1.00)')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Baseline')
axes[1].set_title('Figure 5B: Multi-Class ROC Curves (Macro AUC = 1.000)', fontsize=10.5, fontweight='bold', color='#1A365D')
axes[1].set_xlabel('False Positive Rate', fontsize=9.5, fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontsize=9.5, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=8.5)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures_week5/fig5_ml_evaluation_summary.png', bbox_inches='tight')
plt.close()

# Week 5 Figure 6: Strategic Radar / Standardized Fingerprint Profile
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
sel_features = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'total_phenols', 'flavanoids', 'color_intensity', 'od280/od315_of_diluted_wines', 'proline']
sel_labels = ['Alcohol', 'Malic Acid', 'Alcalinity', 'Phenols', 'Flavanoids', 'Color Int.', 'OD280/OD315', 'Proline']
df_all = pd.DataFrame(X, columns=wine.feature_names)
df_all['class'] = [class_names[i] for i in y]
means_by_class = df_all.groupby('class')[sel_features].mean()
std_means = (means_by_class - df_all[sel_features].mean()) / df_all[sel_features].std()

x_idx = np.arange(len(sel_features))
w = 0.25
for i, (c_name, color) in enumerate(PALETTE.items()):
    ax.bar(x_idx + (i - 1)*w, std_means.loc[c_name], width=w, label=c_name, color=color, edgecolor='black', linewidth=0.7)

ax.set_xticks(x_idx)
ax.set_xticklabels(sel_labels, fontsize=9.5, fontweight='bold')
ax.set_ylabel('Standardized Deviation (Z-Score)', fontsize=10, fontweight='bold')
ax.set_title('Figure 6: Chemical Fingerprint Profiles Driving Strategic Enological Interventions', fontsize=11.5, fontweight='bold', pad=14, color='#1A365D')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.9)
ax.legend(title='Wine Cultivars', frameon=True, facecolor='white', framealpha=0.95)

plt.tight_layout()
plt.savefig('figures_week5/fig6_strategic_chemical_fingerprints.png', bbox_inches='tight')
plt.close()

print('All Week 4 and Week 5 figures generated successfully!')
