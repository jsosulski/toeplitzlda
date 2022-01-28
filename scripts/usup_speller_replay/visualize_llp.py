# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["figure.dpi"] = 200

sns.set_context("paper")
sns.set_style("whitegrid")
figure_path = Path.home() / "results_usup" / "plots"
ds = "both"  # LLP, Mix or both
# nlet = 63 if ds == "LLP" else 35
nlet = 35
max_let = 189 if ds == "LLP" else 105
nsub = 13 if ds == "LLP" else 12
if ds == "both":
    nsub = 25
lowpass_freq_str = "8hz"

toep_td = 27
slda_td = 6

# set this to False for ICML
plot_titles = False
colored_labels = True

figure_path.mkdir(exist_ok=True)


def strikethrough(text):
    return "\u0336".join(text) + "\u0336"


def savefig(fig, name, prefix=None, format="pdf", dpi=200):
    if prefix is None:
        prefix = f"{ds}_{lowpass_freq_str}"
    figfile = figure_path / f"{prefix}_{name}.{format}"
    fig.savefig(figfile, dpi=dpi)


paths = list(
    (Path.home() / "results_usup").glob(f"{lowpass_freq_str}*")
)
fs_onecol = (4.5, 4.2)
fs_twocol = (9, 4.2)


dfs = []
if ds == "both":
    dss = ["LLP", "Mix"]
else:
    dss = [ds]
for datset in dss:
    for p in paths:
        try:
            df = pd.read_csv(p / f"{datset}_usup_toeplitz.csv")
        except:
            print(f"Could not read {p}")
            continue
        df["use_jump"] = "_jump" in str(p)
        df["use_base"] = "_base" in str(p)
        df["use_chdrop"] = "_chdrop" in str(p)
        if ds == "both":
            df["subject"] = ("A" if datset == "LLP" else "B") + df.subject.astype(
                str
            ).str.zfill(2)
            df["max_let"] = 189 if datset == "LLP" else 105
        else:
            df["max_let"] = max_let
        dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)
all_df["Classifier"] = all_df.clf.replace({"slda": "sLDA", "toep_lda": "ToeplitzLDA"})
all_df["Correct"] = all_df.correct
all_df["$N_t$"] = all_df.ntime_features
all_df["Feature dimension"] = all_df.ntime_features * 31
all_df["AUC"] = all_df.auc
# Original LLP Code removed Fp1 and Fp2
all_df.loc[all_df["ntime_features"] == 6, "Feature dimension"] -= 2 * 6
all_df = all_df.infer_objects()

num_evaluated_features = len(all_df["ntime_features"].unique())
per_block = (
    all_df.groupby(["block", "subject", "Classifier", "lowpass", "$N_t$"])
    .mean()
    .reset_index()
)
summary = (
    per_block.groupby(["subject", "Classifier", "lowpass", "$N_t$"])
    .mean()
    .reset_index()
)
summary["$N_t$ / Feature dimension"] = summary["$N_t$"].astype(int).astype(str)
summary["$N_t$ / Feature dimension"] += " / " + summary["Feature dimension"].astype(
    int
).astype(str)

cp = sns.color_palette("viridis", num_evaluated_features - 1)
cp.insert(0, (1, 0, 0))

# %% Correct ratio
fig, ax = plt.subplots(1, 1, figsize=fs_onecol)
# plot_matched(ax=ax, data=summary, x="ntime_features", y="correct")
g = sns.boxplot(
    ax=ax,
    data=summary,
    x="$N_t$ / Feature dimension",
    y="correct",
    hue="Classifier",
    # whis=1.5,
    fliersize=4,
    saturation=1,
)

ax.legend(loc="best")
ax.set_ylim(None, 1)
ax.annotate(
    strikethrough("A1, A2"),
    (-0.2, 1.005),
    size=7.9,
    color="red",
    fontweight="bold",
    ha="center",
)  # , arrowprops=dict(facecolor='black'))
# ax.set_xlabel("$N_t$ / Feature dimension")
ax.set_ylabel("Ratio of correct letters")
if plot_titles:
    ax.set_title("Speller performance, based on dimensions")
ax.set_ylim((0.4, 1.03))
xtl = ax.get_xticklabels()
if colored_labels:
    for l, c in zip(xtl, cp):
        l.set_color(c)
fig.tight_layout()
savefig(fig, "average_for_dimensions")
plt.show()

# %% Learning curves
fig, ax = plt.subplots(1, 1, figsize=fs_onecol)
summary = (
    all_df.groupby(["Classifier", "nth_letter", "lowpass", "$N_t$"])
    .mean()
    .reset_index()
)
sns.lineplot(
    ax=ax,
    data=summary,
    x="nth_letter",
    y="correct",
    style="Classifier",
    hue="$N_t$",
    palette=cp,
)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
ax.set_xlim(1, nlet)
# ax.set_ylim(0, 1.05)
ax.set_xlabel("Nth letter")
ax.set_ylabel("Correctly classified")
if plot_titles:
    ax.set_title("Correct letter ratio for different time dimensions")
ax.get_xticks()

fig.tight_layout()
savefig(fig, "learning_curve_per_timedimension")
plt.show()

# %% Learning curves
fig, ax = plt.subplots(1, 1, figsize=fs_onecol)
summary = (
    all_df.groupby(["Classifier", "nth_letter", "lowpass", "$N_t$"])
    .mean()
    .reset_index()
)
# cp = sns.color_palette("viridis", 4)
# cp[-1] = (1, 0, 0)
sns.lineplot(
    ax=ax,
    data=summary,
    x="nth_letter",
    y="auc",
    style="Classifier",
    hue="$N_t$",
    palette=cp,
    # legend=False,
)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
ax.set_xlim(1, nlet)
# ax.set_ylim(0, 1.05)
ax.set_xlabel("Number of letters for training")
ax.set_ylabel("AUC")
if plot_titles:
    ax.set_title("AUC learning curve for different time dimensions")
ax.get_xticks()
lh, ll = ax.get_legend_handles_labels()
lh.extend(2 * [lh[0]])
ll.extend(2 * [""])
# ax.set_ylim((0.68, None))
ax.legend(lh, ll, loc="lower right", ncol=2)

fig.tight_layout()
savefig(fig, "learning_curve_per_timedimension_auc")
plt.show()

# %% DF SELECT
df_toep = all_df.loc[(all_df.clf == "toep_lda") & (all_df.ntime_features == toep_td)]
df_slda = all_df.loc[(all_df.clf == "slda") & (all_df.ntime_features == slda_td)]
df = pd.concat([df_toep, df_slda])

# %%
# rdf = df.loc[df.subject != 10]
# rdf['sb'] = rdf.subject.astype(str) + rdf.block.astype(str)
rdf = df

fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax, data=rdf, x="nth_letter", y="correct", hue="Classifier")
ax.legend(loc="lower right")
ax.set_xlabel("Nth letter")
ax.set_ylabel("Correctly classified")
if plot_titles:
    ax.set_title("Classification performance on letter-level")
fig.tight_layout()
savefig(fig, "binary")
plt.show()

fig, ax = plt.subplots(1, 1)
sns.lineplot(ax=ax, data=rdf, x="nth_letter", y="AUC", hue="Classifier")
ax.legend(loc="lower right")
ax.set_xlabel("Nth letter")
if plot_titles:
    ax.set_title("Classification performance on epoch-level")
fig.tight_layout()
savefig(fig, "auc")
plt.show()
# %%
# plt.figure()
# g = sns.FacetGrid(rdf, col="subject", row="block", hue="clf")
# g.map_dataframe(sns.lineplot, x="nth_letter", y="auc")
# plt.show()
# %%
# g = sns.FacetGrid(rdf, col="subject", row="block", hue="clf")
# g.map_dataframe(sns.lineplot, x="nth_letter", y="correct", markers="o")
# plt.show()
hline_pos = [3] * nsub
if ds in ["LLP", "both"]:
    hline_pos[
        5
    ] -= 1  # LLP dataset is missing block 2 of sub 6 due to optical marker issues
hline_pos = np.cumsum(hline_pos)
vline_pos = [7, 14, 21, 28, 35, 42, 49, 56]
if ds == "Mix":
    vline_pos = vline_pos[:4]

yticklabel_pos_llp = [
    1.5,
    4.5,
    7.5,
    10.5,
    13.5,
    16,
    18.5,
    21.5,
    24.5,
    27.5,
    30.5,
    33.5,
    36.5,
]
yticklabel_pos_mix = [
    1.5,
    4.5,
    7.5,
    10.5,
    13.5,
    16.5,
    19.5,
    22.5,
    25.5,
    28.5,
    31.5,
    34.5,
]
if ds == "LLP":
    yticklabel_pos = yticklabel_pos_llp
elif ds == "Mix":
    yticklabel_pos = yticklabel_pos_mix
else:
    yticklabel_pos = [*yticklabel_pos_llp, *[y + 38 for y in yticklabel_pos_mix]]


yticklabels = df.subject.unique()  # list(range(1, nsub + 1))


for letter_limit in [nlet]:
    rdf = df.loc[df.nth_letter <= letter_limit]
    cm = sns.color_palette("viridis_r", as_cmap=True)
    hm_df = rdf[["subject", "block", "nth_letter", "correct", "clf"]]
    slda_df = hm_df.loc[hm_df["clf"] == "slda"]
    slda_df = slda_df[["subject", "block", "nth_letter", "correct"]]
    slda_df = slda_df.pivot(
        index=["subject", "block"], columns="nth_letter", values="correct"
    )
    toep_lda_df = hm_df.loc[hm_df["clf"] == "toep_lda"]
    toep_lda_df = toep_lda_df[["subject", "block", "nth_letter", "correct"]]
    toep_lda_df = toep_lda_df.pivot(
        index=["subject", "block"], columns="nth_letter", values="correct"
    )
    # fig, axes = plt.subplots(
    #     1, 2, figsize=(1 + 11 * (letter_limit / nlet), 4), sharex="all", sharey="all"
    # )
    fig, axes = plt.subplots(1, 2, figsize=fs_onecol, sharex="all", sharey="all")
    axes = axes.ravel()
    g = sns.heatmap(
        slda_df,
        ax=axes[0],
        square=True,
        vmin=0,
        vmax=1,
        cbar=None,
        cmap=cm,
        linewidths=0.05,
        linecolor="black",
    )
    [axes[0].axvline(l - 0.1, color="white", linewidth=1) for l in vline_pos]
    [axes[0].axhline(l - 0.1, color="white", linewidth=1) for l in hline_pos]
    # axes[0].set_title("Means: LLP, Covariance: global LW-Shrinkage Cov")
    # axes[0].set_title(f"sLDA for $N_t={slda_td}$")
    axes[0].set_title(f"sLDA")
    axes[0].set_xlabel("Nth letter")
    # axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_ylabel("Subject")
    gs = sns.heatmap(
        toep_lda_df,
        ax=axes[1],
        square=True,
        vmin=0,
        vmax=1,
        cbar=None,
        cmap=cm,
        linewidths=0.05,
        linecolor="black",
    )
    [axes[1].axvline(l - 0.1, color="white", linewidth=1) for l in vline_pos]
    [axes[1].axhline(l - 0.1, color="white", linewidth=1) for l in hline_pos]
    # axes[1].set_title("Means: LLP, Covariance: global Toeplitz-Tapered")
    axes[1].set_title(f"ToeplitzLDA")
    axes[1].set_xlabel("Nth letter")
    # axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[0].set_yticks(yticklabel_pos)
    axes[0].set_yticklabels(yticklabels)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.1)
    # fig.text(0.5, 0.01, "Nth letter")
    # fig.suptitle("Correct classified letter, no post-fix")
    # fig.suptitle("")
    savefig(fig, f"heatmap_letters_{letter_limit}", format="png", dpi=300)
    savefig(fig, f"heatmap_letters_{letter_limit}", format="pdf")
    plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=fs_twocol)
diff_df = toep_lda_df.astype(int) - slda_df.astype(int)
g = sns.heatmap(
    diff_df,
    ax=ax,
    square=True,
    vmin=-1,
    vmax=1,
    cbar=None,
    cmap=sns.diverging_palette(370, 125, s=60, as_cmap=True, center="dark"),
    linewidths=0.0,
    linecolor="black",
)
fig.tight_layout()
savefig(fig, "diff_map", format="png")
plt.show()

# %%
# Increasing vs plot
# %%
rdf = df

gdf = rdf.groupby(["subject", "block", "clf"]).max().reset_index()
gdf["Subject - Block"] = gdf.subject.astype(str) + "-" + gdf.block.astype(str)
max_letter = gdf.nth_letter.unique().max()
# gdf["Incorrectly classified letters"] = max_letter - gdf.correct_sofar
gdf["Incorrectly classified letters"] = gdf.max_let / 3 - gdf.correct_sofar
gdf["wrong_letters"] = gdf["Incorrectly classified letters"]

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
sns.barplot(
    ax=ax,
    data=gdf,
    x="Subject - Block",
    y="Incorrectly classified letters",
    hue="Classifier",
)
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=60)
fig.tight_layout()
plt.show()

reductions = []

for s in gdf.subject.unique():
    cursub_df = gdf.loc[(gdf.subject == s)]
    normal_lda_mistakes = (
        cursub_df[(cursub_df.clf == "slda")].wrong_letters.sum().astype(int)
    )
    toep_lda_mistakes = (
        cursub_df[(cursub_df.clf == "toep_lda")].wrong_letters.sum().astype(int)
    )
    max_letters = int((len(cursub_df) / 2) * (cursub_df.max_let.unique()[0] / 3))
    red = 100 - 100 * toep_lda_mistakes / normal_lda_mistakes
    print(f"Subject {s}")
    print(f" Shrinkage LDA: {normal_lda_mistakes}/{max_letters}")
    print(f" Toeplitz LDA:  {toep_lda_mistakes}/{max_letters}")
    print(f" Reduction of errors: {red}%")
    reductions.append(red)

print(f"\nAverage across Subjects")
print(f" Reduction of errors: {np.mean(reductions)}%\n")
# %%
fig, ax = plt.subplots(1, 1, figsize=fs_onecol)
xt = gdf.subject.unique()  # list(range(1, nsub+1))
ax.bar(xt, reductions, color="gray")
ax.set_xticks(xt)
ax.set_xticklabels(xt, rotation=45, size=7)
ax.set_xlabel("Subject")
ax.set_ylabel("Reduction of incorrectly spelled letters [%]")
ax.axhline(np.mean(reductions), label="Average error reduction", color="k")
ax.annotate(f"{np.mean(reductions):1.1f}", (20.9, np.mean(reductions) + 1), color="k")

ax.legend(loc="lower center")
if plot_titles:
    ax.set_title("Reduction of errors by using Toeplitz LDA instead of sLDA")
fig.tight_layout()
savefig(fig, "error_rate_reduction")
plt.show()

# %%

asdf = gdf.groupby(["subject", "clf"]).sum().reset_index()
asdf["correct_ratio"] = (asdf.nth_letter - asdf.wrong_letters) / asdf.nth_letter
asdf["error_rate"] = 1 - asdf.correct_ratio
met = "error_rate"
slda_cr = asdf.loc[asdf.clf == "slda"][met].reset_index()
tlda_cr = asdf.loc[asdf.clf == "toep_lda"][met].reset_index()
plt.scatter(tlda_cr[met], slda_cr[met], clip_on=False, zorder=20)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1])
plt.gca().set_box_aspect(1)
plt.title(met)
plt.xlabel("ToeplitzLDA")
plt.ylabel("sLDA")
plt.show()


print(asdf.groupby("clf").median()[["wrong_letters", "correct_ratio", "error_rate"]])
