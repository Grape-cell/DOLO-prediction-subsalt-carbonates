# ======================== Required Libraries =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import shap
import warnings
warnings.filterwarnings("ignore")

# ======================== User Configuration =========================
all_file = 'training_set.xlsx'
predict_file = 'prediction_set.xlsx'

well_col = 'Well'
depth_col = 'Depth'
dolo_col = 'DOLO'
feature_cols = ['CNL', 'DEN', 'AC', 'PE', 'GR']

output_prefix = 'SCI_DOLO'
dpi = 300

# ======================== Output Folder Configuration =========================
# All Excel files and figures will be output to this folder
# You only need to modify this section, for example:
# output_dir = r"D:\Geo\DOLO_Result\SCI_Figures"
output_dir = r"D:\Geo\paper\Z-多测井结合的预测白云岩含量\投稿\一改\代码"

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

def out_path(filename):
    """Unified output path"""
    return output_dir / filename

# ======================== Unified Publication Figure Style Settings =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": dpi,
    "figure.dpi": dpi,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False
})

sns.set_theme(
    style="ticks",
    font="Times New Roman",
    rc={
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black"
    }
)

# ======================== Publication Figure Color Scheme =========================
# Low-saturation, color-blind-friendly colors suitable for journal figures
MODEL_COLORS = {
    "XGB": "#0072B2",       # blue
    "RF": "#009E73",        # green
    "LGB": "#D55E00",       # vermillion
    "Stacking": "#6A3D9A"   # purple
}

CLASS_COLORS = {
    "Low": "#0072B2",
    "Medium": "#E69F00",
    "High": "#D55E00"
}

ROC_COLORS = ["#0072B2", "#E69F00", "#D55E00"]

CM_CMAP = LinearSegmentedColormap.from_list(
    "academic_blue",
    ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#3182BD", "#08519C"]
)

RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "academic_residual",
    ["#2166AC", "#F7F7F7", "#B2182B"]
)

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "academic_shap",
    ["#2166AC", "#F7F7F7", "#B2182B"]
)

# ======================== Figure Panel Label Settings =========================
# Fixed panel labels as required by the user: RF=(a), XGB=(b), LGB=(c), Stacking=(d)
PANEL_LABELS = {
    "RF": "(a)",
    "XGB": "(b)",
    "LGB": "(c)",
    "Stacking": "(d)"
}

def add_bottom_title(fig, title, y=0.065):
    """
    Place the figure title directly below the figure and move it slightly upward to ensure balanced journal figure layout.
    """
    fig.text(
        0.5,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=9,
        fontname="Times New Roman"
    )

def sci_style_axes(ax, grid=False):
    """Unified journal figure axis style"""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9,
        width=0.8,
        length=3,
        direction="out"
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    if grid:
        ax.grid(True, linestyle="-", linewidth=0.35, alpha=0.28)
    else:
        ax.grid(False)

    return ax

def sci_style_legend(legend, fontsize=8):
    """Unified legend style"""
    if legend is not None:
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.92)

        for text in legend.get_texts():
            text.set_fontname("Times New Roman")
            text.set_fontsize(fontsize)

        if legend.get_title() is not None:
            legend.get_title().set_fontname("Times New Roman")
            legend.get_title().set_fontsize(fontsize)

def sci_savefig(filename):
    """Unified save format"""
    plt.savefig(out_path(filename), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def set_all_fonts(fig, fontsize=9):
    """Force all text in the figure to use a unified font"""
    for ax in fig.axes:
        ax.title.set_fontname("Times New Roman")
        ax.title.set_fontsize(fontsize)
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontsize(fontsize)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(fontsize)

# ======================== 1. Data =========================
df = pd.read_excel(all_file)
df.columns = df.columns.str.strip()
df = df[[well_col, depth_col, dolo_col] + feature_cols].dropna()

groups = df[well_col]

# ======================== 2. Leave-One-Well-Out (GroupKFold) =========================
print("\n===== Leave-one-well-out validation (strict version) =====")

gkf = GroupKFold(n_splits=len(groups.unique()))
results = []

for train_idx, test_idx in gkf.split(df, groups=groups):

    train = df.iloc[train_idx].copy()
    test  = df.iloc[test_idx].copy()

    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    X_train = train[feature_cols]
    y_train = train[dolo_col]
    X_test  = test[feature_cols]
    y_test  = test[dolo_col]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    well_name = test[well_col].iloc[0]
    results.append([well_name, r2, rmse])

results_df = pd.DataFrame(results, columns=["Well","R2","RMSE"])
print(results_df)

print("\nMean R²:", results_df["R2"].mean())

# ======================== 3. Prediction Well =========================
print("\n===== Prediction well evaluation (true generalization ability) =====")

df_pred = pd.read_excel(predict_file)
df_pred.columns = df_pred.columns.str.strip()
df_pred = df_pred[[depth_col, dolo_col] + feature_cols].dropna()

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
df_pred[feature_cols] = scaler.transform(df_pred[feature_cols])

X_all = df[feature_cols]
y_all = df[dolo_col]

X_pred = df_pred[feature_cols]
y_true = df_pred[dolo_col]

# ======================== 4. Three Models =========================
models = {
    "XGB": XGBRegressor(n_estimators=300, max_depth=4),
    "RF": RandomForestRegressor(n_estimators=300),
    "LGB": LGBMRegressor(n_estimators=300)
}

pred_df = df_pred.copy()
metrics = []

for name, m in models.items():
    m.fit(X_all, y_all)
    pred = m.predict(X_pred)

    pred_df[f"DOLO_{name}"] = pred

    r2 = r2_score(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)

    metrics.append([name, r2, rmse, mae])

    print(f"{name} prediction well R² = {r2:.3f}")

metrics_df = pd.DataFrame(metrics, columns=["Model","R2","RMSE","MAE"])

best_model = metrics_df.loc[metrics_df["R2"].idxmax(),"Model"]
pred_df["Best"] = pred_df[f"DOLO_{best_model}"]

print(f"\n🔥 Best model: {best_model}")
print(f"🔥 Best model prediction well R²: {metrics_df['R2'].max():.3f}")

# ======================== 5. Leakage-Free Stacking (Fixed Version) =========================
print("\n===== Stacking fusion (strict version) =====")

n_splits = min(5, len(groups.unique()))
gkf = GroupKFold(n_splits=n_splits)

stack_train = np.zeros((X_all.shape[0], 3))

for i,(name,m) in enumerate(models.items()):
    tmp = np.zeros(X_all.shape[0])

    for tr,va in gkf.split(X_all, groups=groups):
        m.fit(X_all.iloc[tr], y_all.iloc[tr])
        tmp[va] = m.predict(X_all.iloc[va])

    stack_train[:,i] = tmp

meta = LinearRegression()
meta.fit(stack_train, y_all)

stack_pred = np.column_stack([
    m.fit(X_all, y_all).predict(X_pred) for m in models.values()
])

pred_df["Stacking"] = meta.predict(stack_pred)

r2_stack = r2_score(y_true, pred_df["Stacking"])
print(f"🔥 Stacking prediction well R² = {r2_stack:.3f}")

# ======================== 6. Classification =========================
bins = [0,20,80,100]
labels = ["Low","Medium","High"]

y_all_cls = LabelEncoder().fit_transform(pd.cut(y_all, bins=bins, labels=labels).astype(str))
y_true_cls = LabelEncoder().fit_transform(pd.cut(y_true, bins=bins, labels=labels).astype(str))

clf = XGBClassifier(eval_metric='mlogloss')
clf.fit(X_all, y_all_cls)

prob = clf.predict_proba(X_pred)

# ======================== Basic ROC Plot =========================
y_bin = label_binarize(y_true_cls, classes=[0,1,2])

fig, ax = plt.subplots(figsize=(3.35, 3.35))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        linewidth=1.35,
        color=ROC_COLORS[i],
        label=f"{labels[i]} AUC = {roc_auc:.2f}"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="#4D4D4D",
    linewidth=0.8,
    alpha=0.85
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

legend = ax.legend(
    frameon=True,
    loc="lower right",
    handlelength=1.5,
    handletextpad=0.45,
    borderpad=0.35,
    labelspacing=0.28
)
sci_style_legend(legend, fontsize=8)

sci_style_axes(ax, grid=True)
fig.tight_layout(rect=[0, 0.08, 1, 1])
add_bottom_title(fig, "(b) ROC Curve")
sci_savefig("ROC.jpg")

# ======================== 7. SHAP =========================
print("\n===== SHAP interpretation (best model) =====")

best_model_name = best_model

if best_model_name == "XGB":
    model_final = XGBRegressor(n_estimators=300, max_depth=4)
elif best_model_name == "RF":
    model_final = RandomForestRegressor(n_estimators=300)
elif best_model_name == "LGB":
    model_final = LGBMRegressor(n_estimators=300)

model_final.fit(X_all, y_all)

explainer = shap.TreeExplainer(model_final)
shap_values = explainer.shap_values(X_all)

# ======================== SHAP Summary Plot =========================
plt.figure(figsize=(3.75, 3.05))

shap.summary_plot(
    shap_values,
    X_all,
    show=False,
    cmap=SHAP_CMAP,
    max_display=len(feature_cols)
)

fig = plt.gcf()
fig.set_size_inches(3.75, 3.05)
ax = plt.gca()

ax.set_title(f"SHAP Summary ({best_model_name})", fontsize=9, pad=6)
ax.set_xlabel("SHAP value", fontsize=9)

# SHAP plots often automatically generate a colorbar; font and line width are unified here
for current_ax in fig.axes:
    current_ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    for spine in current_ax.spines.values():
        spine.set_linewidth(0.8)

    current_ax.title.set_fontname("Times New Roman")
    current_ax.xaxis.label.set_fontname("Times New Roman")
    current_ax.yaxis.label.set_fontname("Times New Roman")

    for label in current_ax.get_xticklabels() + current_ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(9)

set_all_fonts(fig, fontsize=9)

fig.tight_layout()
sci_savefig("SCI_SHAP_summary.jpg")

# ======================== SHAP Bar Plot =========================
plt.figure(figsize=(3.45, 2.85))

shap.summary_plot(
    shap_values,
    X_all,
    plot_type="bar",
    show=False,
    max_display=len(feature_cols)
)

fig = plt.gcf()
fig.set_size_inches(3.45, 2.85)
ax = plt.gca()

# Reset the SHAP bar colors to better match the journal figure style
for patch in ax.patches:
    patch.set_facecolor("#4C78A8")
    patch.set_edgecolor("black")
    patch.set_linewidth(0.45)
    patch.set_alpha(0.92)

ax.set_title(f"SHAP Importance ({best_model_name})", fontsize=9, pad=6)
ax.set_xlabel("Mean |SHAP value|", fontsize=9)

sci_style_axes(ax, grid=False)
set_all_fonts(fig, fontsize=9)

fig.tight_layout()
sci_savefig("SCI_SHAP_bar.jpg")

print("🔥 SHAP completed (best model)")

# ======================== 8. SCI Enhancement =========================
std = np.std(stack_pred, axis=1)
pred_df["Uncertainty"] = std

residual = y_true - pred_df["Best"]

# ======================== Residual-Depth Plot =========================
fig, ax = plt.subplots(figsize=(3.35, 2.8))

vmax = np.nanmax(np.abs(residual))
vmin = -vmax

sc = ax.scatter(
    pred_df[depth_col],
    residual,
    c=residual,
    cmap=RESIDUAL_CMAP,
    vmin=vmin,
    vmax=vmax,
    s=18,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.25
)

ax.axhline(
    0,
    color="#4D4D4D",
    linestyle="--",
    linewidth=0.8,
    alpha=0.85
)

ax.invert_yaxis()
ax.set_xlabel("Depth")
ax.set_ylabel("Residual")
ax.set_title("")

cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.025)
cbar.set_label("Residual", fontsize=9, fontname="Times New Roman")
cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(9)

sci_style_axes(ax, grid=True)
fig.tight_layout(rect=[0, 0.08, 1, 1])
add_bottom_title(fig, f"({PANEL_LABELS.get(best_model, '(b)')[1]}) Residual vs. Depth")
sci_savefig("Error_Depth.jpg")

# ======================== Model R² Comparison Plot =========================
fig, ax = plt.subplots(figsize=(3.35, 2.55))

bar_colors = [MODEL_COLORS.get(m, "#4C78A8") for m in metrics_df["Model"]]

bars = ax.bar(
    metrics_df["Model"],
    metrics_df["R2"],
    width=0.55,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.55,
    alpha=0.92
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontname="Times New Roman"
    )

ax.set_xlabel("Model")
ax.set_ylabel("R²")
ax.set_title("Model Performance")
ax.set_ylim(0, max(metrics_df["R2"].max() * 1.18, 1.0))

sci_style_axes(ax, grid=True)
fig.tight_layout()
sci_savefig("Model_Compare.jpg")

# ======================== 9. Output =========================
with pd.ExcelWriter(out_path(f"{output_prefix}_NB.xlsx")) as writer:
    pred_df.to_excel(writer, "Prediction", index=False)
    metrics_df.to_excel(writer, "Metrics", index=False)

print("\n🔥🔥🔥 Full workflow completed (Q1 submission-ready version)")

# ======================== 10. SCI-Level Visualization Enhancement (Prediction Well Only) =========================
print("\n===== SCI-level visualization output (prediction well) =====")

# ======================== 10.1 Three-Class Classification (for ROC & Confusion Matrix) =========================
print("\n===== 🔥 Q1-level final visualization (multi-model comparison) =====")

bins = [0,30,70,100]
labels = ["Low","Medium","High"]

model_preds = {
    "XGB": pred_df["DOLO_XGB"],
    "RF": pred_df["DOLO_RF"],
    "LGB": pred_df["DOLO_LGB"],
    "Stacking": pred_df["Stacking"]
}

y_true_cls = pd.cut(y_true, bins=bins, labels=labels).astype(str)
le = LabelEncoder()
y_true_enc = le.fit_transform(y_true_cls)

# =========================================================
# Loop through each model to output SCI-level figures
# =========================================================
for name, pred in model_preds.items():

    print(f"\n👉 Processing model: {name}")

    pred = np.clip(pred, 0, 100)

    panel = PANEL_LABELS.get(name, "")
    title_prefix = f"{panel} " if panel else ""

    # ================= Classification =================
    y_pred_cls = pd.cut(pred, bins=bins, labels=labels).astype(str)
    y_pred_cls[y_pred_cls == "nan"] = "Medium"
    y_pred_enc = le.transform(y_pred_cls)

    # ================= ROC (separate classifier for each base model; ROC is not plotted for Stacking) =================
    clf_models = {
        "XGB": XGBClassifier(eval_metric="mlogloss"),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42),
        "LGB": LGBMClassifier(n_estimators=300, verbose=-1)
    }

    if name in clf_models:

        y_all_cls_roc = pd.cut(y_all, bins=bins, labels=labels).astype(str)
        y_all_cls_roc[y_all_cls_roc == "nan"] = "Medium"

        le_roc = LabelEncoder()
        y_all_enc_roc = le_roc.fit_transform(y_all_cls_roc)

        clf_roc = clf_models[name]
        clf_roc.fit(X_all, y_all_enc_roc)

        prob = clf_roc.predict_proba(X_pred)

        y_true_cls_roc = pd.cut(y_true, bins=bins, labels=labels).astype(str)
        y_true_cls_roc[y_true_cls_roc == "nan"] = "Medium"
        y_true_enc_roc = le_roc.transform(y_true_cls_roc)

        y_bin = label_binarize(y_true_enc_roc, classes=[0, 1, 2])

        fig, ax = plt.subplots(figsize=(3.35, 3.35))

        for i, cls_name in enumerate(le_roc.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                linewidth=1.35,
                color=ROC_COLORS[i],
                label=f"{cls_name} AUC = {roc_auc:.2f}"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#4D4D4D",
            linewidth=0.8,
            alpha=0.85
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        legend = ax.legend(
            frameon=True,
            loc="lower right",
            handlelength=1.5,
            handletextpad=0.45,
            borderpad=0.35,
            labelspacing=0.28
        )
        sci_style_legend(legend, fontsize=8)

        sci_style_axes(ax, grid=True)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        add_bottom_title(fig, f"{title_prefix}ROC Curve ({name})")
        sci_savefig(f"SCI_ROC_{name}.jpg")

    else:
        print(f"⚠️ {name} is a regression stacking output; ROC is skipped because it has no class probabilities.")

    # ================= Confusion Matrix =================
    cm = confusion_matrix(y_true_enc, y_pred_enc)

    fig, ax = plt.subplots(figsize=(3.35, 2.85))

    # Keep consistency with the R² crossplot: use the corresponding primary model color for each model
    model_color = MODEL_COLORS.get(name, "#4C78A8")

    cm_model_cmap = LinearSegmentedColormap.from_list(
        f"cm_{name}",
        ["#FFFFFF", "#F7F7F7", model_color],
        N=256
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cm_model_cmap,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={
            "fontsize": 9,
            "fontname": "Times New Roman",
            "color": "black"
        },
        cbar_kws={
            "shrink": 0.82,
            "pad": 0.025
        },
        linewidths=0.55,
        linecolor="white",
        square=True,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    for label_tick in cbar.ax.get_yticklabels():
        label_tick.set_fontname("Times New Roman")
        label_tick.set_fontsize(9)

    sci_style_axes(ax, grid=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    add_bottom_title(fig, f"{title_prefix}Confusion Matrix ({name})")
    sci_savefig(f"SCI_CM_{name}.jpg")

    # ================= CrossPlot: R² Crossplot =================
    fig, ax = plt.subplots(figsize=(3.35, 3.35))

    model_color = MODEL_COLORS.get(name, "#4C78A8")

    ax.scatter(
        y_true,
        pred,
        alpha=0.78,
        s=20,
        color=model_color,
        edgecolors="black",
        linewidths=0.25
    )

    min_val = min(y_true.min(), pred.min())
    max_val = max(y_true.max(), pred.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="#333333",
        linewidth=0.9,
        label="1:1 line"
    )

    r2 = r2_score(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)

    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        fontname="Times New Roman",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.88
        )
    )

    ax.set_xlabel("Measured DOLO (%)")
    ax.set_ylabel("Predicted DOLO (%)")
    ax.set_title("")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        handlelength=1.5,
        borderpad=0.3
    )
    sci_style_legend(legend, fontsize=8)

    sci_style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    add_bottom_title(fig, f"{title_prefix}Prediction vs. Measured ({name})")
    sci_savefig(f"SCI_CrossPlot_{name}.jpg")

    # ================= Residual Distribution =================
    residual = y_true - pred

    fig, ax = plt.subplots(figsize=(3.35, 2.55))

    sns.histplot(
        residual,
        bins=35,
        kde=True,
        color=model_color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        line_kws={
            "linewidth": 1.15,
            "color": "#222222"
        },
        ax=ax
    )

    ax.axvline(
        0,
        color="#4D4D4D",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85
    )

    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("")

    sci_style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    add_bottom_title(fig, f"{title_prefix}Residual Distribution ({name})")
    sci_savefig(f"SCI_Residual_{name}.jpg")

# =========================================================
# Advanced Correlation Plot (Prediction Well)
# =========================================================
pairplot_df = df_pred.copy()
pairplot_df["Class"] = y_true_cls

pairplot_palette = {
    "Low": CLASS_COLORS["Low"],
    "Medium": CLASS_COLORS["Medium"],
    "High": CLASS_COLORS["High"]
}

g = sns.pairplot(
    pairplot_df,
    vars=feature_cols,
    hue="Class",
    palette=pairplot_palette,
    diag_kind="kde",
    height=1.35,
    aspect=1.0,
    plot_kws={
        "alpha": 0.65,
        "s": 12,
        "edgecolor": "black",
        "linewidth": 0.25
    },
    diag_kws={
        "linewidth": 1.0,
        "alpha": 0.65
    }
)

g.fig.set_size_inches(6.2, 6.2)
g.fig.suptitle(
    "Feature Correlation of Prediction Well",
    y=1.02,
    fontsize=9,
    fontname="Times New Roman"
)

for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=9, fontname="Times New Roman")
        ax.set_ylabel(ax.get_ylabel(), fontsize=9, fontname="Times New Roman")
        ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)

        for label_tick in ax.get_xticklabels() + ax.get_yticklabels():
            label_tick.set_fontname("Times New Roman")
            label_tick.set_fontsize(9)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

if g._legend is not None:
    g._legend.set_title("Class")
    g._legend.get_title().set_fontsize(9)
    g._legend.get_title().set_fontname("Times New Roman")
    for text in g._legend.texts:
        text.set_fontsize(9)
        text.set_fontname("Times New Roman")

plt.savefig(
    out_path("SCI_Pairplot.jpg"),
    dpi=dpi,
    bbox_inches="tight",
    pad_inches=0.05
)
plt.close()

print(f"\n🔥🔥🔥 Q1-level final visualization completed. All results have been output to: {output_dir}")