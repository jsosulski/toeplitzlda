from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")
FIG_PATH = Path.home() / "Desktop" / "ablage"

for dcode in ["Mix", "LLP"]:
    # for dcode in ["LLP", "Mix"]:
    # df = pd.read_csv(f"/home/jan/res_archive/results_em_{dcode.lower()}.csv")
    all_df = pd.read_csv(f"/home/jan/results_em_{dcode.lower()}.csv")
    # %%
    for nch in all_df.n_channels.unique():
        for nquarts in all_df.data_amount_in_quarters.unique():
            df = all_df.loc[all_df.n_channels == nch]
            df = df.loc[df.data_amount_in_quarters == nquarts]
            descr = f"{dcode.lower()}_{nch}_channels_{int(df.num_epos.median())}_epochs"
            print(nch)
            print(df.groupby(["aggregated_mean", "toeplitz_covariance"]).mean().correct)
            f, ax = plt.subplots(4, 1, figsize=(10, 13), sharex="all")
            sns.lineplot(
                data=df,
                x="nth_letter",
                y="correct",
                hue="toeplitz_covariance",
                style="aggregated_mean",
                ax=ax[0],
            )
            ax[0].set_ylim(0, 1.1)
            sns.lineplot(
                data=df,
                x="nth_letter",
                y="softmax_logratio_to_second",
                hue="toeplitz_covariance",
                style="aggregated_mean",
                ax=ax[1],
                legend=False,
            )
            sns.lineplot(
                data=df,
                x="nth_letter",
                y="distance_to_true_letter",
                hue="toeplitz_covariance",
                style="aggregated_mean",
                ax=ax[2],
                legend=False,
            )
            sns.lineplot(data=df, x="nth_letter", y="evaluation_time", ax=ax[3], color="k")
            f.suptitle(descr)
            f.savefig(FIG_PATH / f"stats_{descr}.png", dpi=150)
            plt.show()
            # %%
            # cm = sns.color_palette("viridis_r", as_cmap=True)
            cm = sns.color_palette(["#FFD700", "#0057B8"], as_cmap=True)

            n_letters = df.nth_letter.max()
            vline_pos = range(7, n_letters, 7)
            hline_pos = list()
            for sub in df["subject"].unique():
                hline_pos.append(len(df.loc[df["subject"] == sub].block.unique()))
            hline_pos = np.cumsum(hline_pos[:-1])

            # %%
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="all", sharey="all")
            fig.suptitle(descr)
            for ti, toep in enumerate([True, False]):
                for ai, agg_mean in enumerate([True, False]):

                    hyp_df = df.loc[
                        (df["toeplitz_covariance"] == toep) & (df["aggregated_mean"] == agg_mean)
                    ]
                    hm_df = hyp_df[["subject", "block", "nth_letter", "correct"]]
                    hm_df = hm_df.pivot(
                        index=["subject", "block"], columns="nth_letter", values="correct"
                    )
                    hm_df = hm_df.fillna(0.5).astype(float)
                    # axes = axes.ravel()
                    g = sns.heatmap(
                        hm_df,
                        ax=axes[ai, ti],
                        square=True,
                        vmin=0,
                        vmax=1,
                        cbar=None,
                        cmap=cm,
                        linewidths=0.05,
                        linecolor="black",
                    )
                    [axes[ai, ti].axvline(l - 0.1, color="white", linewidth=1) for l in vline_pos]
                    [axes[ai, ti].axhline(l - 0.1, color="white", linewidth=1) for l in hline_pos]
                    cov_method = "Toeplitz cov" if toep else "Shrinkage cov"
                    mean_method = "aggregated mean" if agg_mean else "only current trial mean"
                    axes[ai, ti].set_title(f"{cov_method}, {mean_method}")
                    axes[ai, ti].set_xlabel("Nth letter")
                    axes[ai, ti].set_ylabel("Subject")
            fig.savefig(FIG_PATH / f"heatmap_{descr}.png", dpi=150)
            plt.show()

            # fig_softmax, axes_softmax = plt.subplots(2, 2, figsize=(8, 7), sharex="all", sharey="row")
            # fig_softmax.suptitle(f"Dataset: {dcode}, n_channels: {nch}")
            # for ti, toep in enumerate([True, False]):
            #     for ai, agg_mean in enumerate([True, False]):
            #         hyp_df = df.loc[(df["toeplitz_covariance"] == toep) & (df["aggregated_mean"] == agg_mean)]
            #         sns.barplot(data=hyp_df, x="correct", y="softmax_logratio_to_second", ax=axes_softmax[ai, ti])
            #         cov_method = "Toeplitz cov" if toep else "Shrinkage cov"
            #         mean_method = "aggregated mean" if agg_mean else "only current trial mean"
            #         axes_softmax[ai, ti].set_title(f"{cov_method}, {mean_method}")
            #         axes_softmax[ai, ti].set_xlabel("Was the classification correct?")
            #         axes_softmax[ai, ti].set_ylabel("Softmax log10 diff")
            # plt.show()
