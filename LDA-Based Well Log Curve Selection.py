import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ======================== User configuration area =========================
# File path
file_path = 'LDA.xlsx'              # Please modify this to your Excel file path
sheet_name = 0                          # Worksheet name or index

# Column name configuration
depth_col = 'Depth'                     # Depth column name
dolo_col = 'DOLO'                       # Dolomite content column name (%)

# Logging curve column names: including resistivity
feature_cols = ['Pe', 'DEN', 'CNL', 'AC', 'GR', 'RLLD', 'RLLS']

# DOLO class thresholds (classified using all data)
dolo_bins = [0, 30, 70, 100]            # Low: <30, Medium: 30-70, High: >70
dolo_labels = [0, 1, 2]                 # Class codes
class_names = ['Low dolomite (<30%)', 'Medium dolomite (30-70%)', 'High dolomite (>70%)']

# Figure saving parameters
fig_dpi = 300
fig_format = 'jpg'
output_prefix = 'LDA_all_data'

# ======================== Unified visualization settings =========================
# Only affects figure output, without affecting data processing, model training, or Excel results
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'savefig.dpi': fig_dpi,
    'figure.dpi': fig_dpi,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False
})

sns.set_theme(
    style='ticks',
    font='Times New Roman',
    rc={
        'font.family': 'Times New Roman',
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black'
    }
)

# ======================== Dedicated legend settings for LDA Projection =========================
# For a 3.5 × 2.8 inch single-column paper figure, a 6.5 pt legend is more appropriate
LDA_LEGEND_FONTSIZE = 6.5

# English class names for figures, used only for visualization to avoid Chinese character garbling under Times New Roman
class_names_plot = [
    'Low (<30%)',
    'Medium (30-70%)',
    'High (>70%)'
]

def apply_pub_style(ax, grid=False):
    """
    Apply publication-style axis formatting.
    """
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=9,
        width=0.8,
        length=3,
        direction='out'
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    if grid:
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.35)
    else:
        ax.grid(False)

    return ax

def format_legend(legend, fontsize=9):
    """
    Apply publication-style legend formatting.
    """
    if legend is not None:
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.90)

        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
            text.set_fontsize(fontsize)

        if legend.get_title() is not None:
            legend.get_title().set_fontname('Times New Roman')
            legend.get_title().set_fontsize(fontsize)

def save_figure(fig, filename):
    """
    Save figure with unified publication settings.
    """
    fig.savefig(
        filename,
        dpi=fig_dpi,
        bbox_inches='tight',
        pad_inches=0.05
    )
    plt.close(fig)

# ======================== Data loading and preprocessing ==================
print("Reading data...")
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Check required columns
required_cols = [depth_col, dolo_col] + feature_cols
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Remove rows containing NaN values
df_clean = df[required_cols].dropna().copy()
print(f"Original data rows: {len(df)}, Rows after cleaning: {len(df_clean)}")

# ======================== Generate class labels ======================
df_clean['Class'] = pd.cut(
    df_clean[dolo_col],
    bins=dolo_bins,
    labels=dolo_labels,
    right=False
)

# Remove rows with NaN classes (handling DOLO values outside the threshold range)
df_clean = df_clean.dropna(subset=['Class']).copy()
if len(df_clean) == 0:
    raise ValueError("All DOLO values are outside the threshold range. Please adjust bins.")

y = df_clean['Class'].values.astype(int)
print(f"Sample count for each class: {pd.Series(y).value_counts().to_dict()}")

# Feature matrix
X = df_clean[feature_cols].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix shape: {X_scaled.shape}")

# ======================== LDA model training ======================
lda = LDA(n_components=min(2, len(class_names)-1))
X_lda = lda.fit_transform(X_scaled, y)
y_pred = lda.predict(X_scaled)
y_prob = lda.predict_proba(X_scaled)

# Model evaluation
acc = accuracy_score(y, y_pred)
print(f"LDA classification accuracy: {acc:.4f}")
print("Classification report:")
print(classification_report(y, y_pred, target_names=class_names))

# Obtain discriminant coefficients (first discriminant axis)
coef = lda.coef_
if coef.ndim == 2:
    coef = coef[0]   # Use the first discriminant axis

importance = np.abs(coef)
importance_norm = importance / importance.sum()

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': coef,
    'Abs_Coefficient': importance,
    'Importance (%)': importance_norm * 100
}).sort_values('Importance (%)', ascending=False)

print("Feature importance:")
print(feature_importance_df)

# ======================== Result summary =========================
results_df = df_clean[[depth_col, dolo_col]].copy()
results_df['True_Class'] = y
results_df['True_Class_Name'] = [class_names[i] for i in y]
results_df['Pred_Class'] = y_pred
results_df['Pred_Class_Name'] = [class_names[i] for i in y_pred]

# Add prediction probabilities
for i, name in enumerate(class_names):
    results_df[f'Prob_{name}'] = y_prob[:, i]

# Save Excel
excel_out = f'{output_prefix}_classified.xlsx'
with pd.ExcelWriter(excel_out) as writer:
    results_df.to_excel(writer, sheet_name='Classification', index=False)
    feature_importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

print(f"Classification results have been saved to: {excel_out}")

# ======================== Visualization ===========================
colors = sns.color_palette('Set1', n_colors=len(class_names))

# ======================== Figure 1: LDA projection =========================
if X_lda.shape[1] >= 2:

    fig1, ax1 = plt.subplots(figsize=(3.5, 2.8))

    for i, class_name in enumerate(class_names_plot):
        idx = y == i
        ax1.scatter(
            X_lda[idx, 0],
            X_lda[idx, 1],
            color=colors[i],
            label=class_name,
            s=16,
            alpha=0.75,
            edgecolors='black',
            linewidth=0.3
        )

    ax1.set_xlabel('LD1')
    ax1.set_ylabel('LD2')
    ax1.set_title('LDA Projection')

    # Key modification here: reduce the LDA Projection legend separately to 6.5 pt
    legend = ax1.legend(
        loc='best',
        frameon=True,
        fancybox=False,
        edgecolor='black',
        prop={
            'family': 'Times New Roman',
            'size': LDA_LEGEND_FONTSIZE
        },
        markerscale=0.65,
        handlelength=0.75,
        handletextpad=0.25,
        borderpad=0.18,
        labelspacing=0.18,
        borderaxespad=0.25,
        columnspacing=0.4
    )
    format_legend(legend, fontsize=LDA_LEGEND_FONTSIZE)

    apply_pub_style(ax1, grid=False)
    fig1.tight_layout()

    save_figure(fig1, f'{output_prefix}_LDA_projection.{fig_format}')

else:

    fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))

    for i, class_name in enumerate(class_names_plot):
        ax1.hist(
            X_lda[y == i],
            bins=30,
            alpha=0.55,
            label=class_name,
            color=colors[i],
            edgecolor='black',
            linewidth=0.3
        )

    ax1.set_xlabel('LD1')
    ax1.set_ylabel('Frequency')
    ax1.set_title('LDA Histogram')

    # The one-dimensional LDA histogram also belongs to Figure 1 and uses the same smaller legend
    legend = ax1.legend(
        loc='best',
        frameon=True,
        fancybox=False,
        edgecolor='black',
        prop={
            'family': 'Times New Roman',
            'size': LDA_LEGEND_FONTSIZE
        },
        markerscale=0.65,
        handlelength=0.75,
        handletextpad=0.25,
        borderpad=0.18,
        labelspacing=0.18,
        borderaxespad=0.25,
        columnspacing=0.4
    )
    format_legend(legend, fontsize=LDA_LEGEND_FONTSIZE)

    apply_pub_style(ax1, grid=False)
    fig1.tight_layout()

    save_figure(fig1, f'{output_prefix}_LDA_hist.{fig_format}')

# ======================== Figure 2: Feature importance =========================
fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))

bars = ax2.barh(
    feature_importance_df['Feature'],
    feature_importance_df['Importance (%)'],
    color=plt.cm.viridis(np.linspace(0.25, 0.75, len(feature_cols))),
    edgecolor='black',
    linewidth=0.5,
    height=0.65
)

ax2.set_xlabel('Importance (%)')
ax2.set_ylabel('Logging curve')
ax2.set_title('Feature Importance')
ax2.invert_yaxis()

for bar in bars:
    width = bar.get_width()
    ax2.text(
        width + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.1f}%',
        ha='left',
        va='center',
        fontsize=9,
        fontname='Times New Roman'
    )

xmax = feature_importance_df['Importance (%)'].max()
ax2.set_xlim(0, xmax * 1.18)

apply_pub_style(ax2, grid=False)
fig2.tight_layout()

save_figure(fig2, f'{output_prefix}_feature_importance.{fig_format}')

# ======================== Figure 3: Depth profile =========================
fig3, axes = plt.subplots(
    1,
    2,
    figsize=(6.8, 4.2),
    sharey=True
)

depth = results_df[depth_col].values

# ------------------------ Measured class ------------------------
ax = axes[0]

for i, class_name in enumerate(class_names_plot):
    mask = results_df['True_Class'] == i
    ax.scatter(
        results_df.loc[mask, dolo_col],
        depth[mask],
        color=colors[i],
        label=class_name,
        s=10,
        alpha=0.70,
        edgecolors='none'
    )

ax.set_xlabel(f'{dolo_col} (%)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
ax.set_title('Measured Class')

legend = ax.legend(
    loc='lower right',
    frameon=True,
    fancybox=False,
    edgecolor='black',
    handlelength=1.2,
    handletextpad=0.4,
    borderpad=0.35
)
format_legend(legend, fontsize=9)

apply_pub_style(ax, grid=False)

# ------------------------ Predicted class ------------------------
ax = axes[1]

for i, class_name in enumerate(class_names_plot):
    mask = results_df['Pred_Class'] == i
    ax.scatter(
        results_df.loc[mask, dolo_col],
        depth[mask],
        color=colors[i],
        label=class_name,
        s=10,
        alpha=0.70,
        edgecolors='none'
    )

ax.set_xlabel(f'{dolo_col} (%)')
ax.invert_yaxis()
ax.set_title(f'LDA Prediction (Accuracy = {acc:.2f})')

legend = ax.legend(
    loc='lower right',
    frameon=True,
    fancybox=False,
    edgecolor='black',
    handlelength=1.2,
    handletextpad=0.4,
    borderpad=0.35
)
format_legend(legend, fontsize=9)

apply_pub_style(ax, grid=False)

fig3.suptitle(
    'Depth Profile Analysis',
    fontsize=9,
    fontname='Times New Roman',
    y=0.98
)

fig3.tight_layout(rect=[0, 0, 1, 0.96])

save_figure(fig3, f'{output_prefix}_depth_profile.{fig_format}')

print("All figures have been saved.")
print("Processing completed.")