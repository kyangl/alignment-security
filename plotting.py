import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class Plotter:
    def __init__(self, save_to_paper=True, show_plots=False, no_save=False):
        self.save_to_paper = save_to_paper
        self.show_plots = show_plots
        self.save = not no_save

    def robust_only_correlation(self, specifier="*", **kwargs):
        # get the correlation between each benchmark and robust accuracy
        df = load_merged_results().astype(float)
        # replace all underscores in the column names with spaces
        df.columns = df.columns.str.replace("_", " ")
        df = df.corr(method="spearman")
        # take out all the robust accuracy columns
        df = df.filter(regex=r"Top-\d+ Accuracy", axis=0)
        df = df.drop(
            [col for col in df.index if "Linf" in col],
            axis=1,
        )
        # rename the indices to remove the "Linf" part
        df.index = [
            (
                (col.split(" Accuracy")[0] + " L∞" + col.split("Linf")[1])
                if "Linf" in col
                else col.split(" Accuracy")[0] + " Clean"
            )
            for col in df.index
        ]
        # remove all index and columns with Top-5 in the name
        df = df.drop(
            [col for col in df.columns if "Top-5" in col],
            axis=1,
        )
        df = df.drop(
            [col for col in df.index if "Top-5" in col],
            axis=0,
        )
        df.sort_index(inplace=True)

        # take out all the indices that are robust accuracy
        fig, ax = plt.subplots(figsize=(15, 7))
        # create a heatmap of the models vs the benchmarks with their scores inside the cells, place cbar at the top
        sns.heatmap(
            df,
            # annot=True,
            # fmt=".2f",
            # cmap=sns.diverging_palette(220, 20, as_cmap=True),
            cmap="bwr",
            ax=ax,
            cbar_kws={"location": "top", "shrink": 0.5},
            center=0,
        )

        # Add vertical lines to separate different measurements
        for col in ["neural vision", "behavior vision", "engineering vision"]:
            if col in df.columns:
                ax.axvline(df.columns.get_loc(col), color="black", linewidth=3)

        display_all_xticks(
            [
                (
                    f"$\\bf{{{c.replace(' ', '~')}}}$"
                    if c in ["neural vision", "behavior vision", "engineering vision"]
                    else c.replace("_", " ")
                )
                for c in df.columns
            ],
            ax,
            fontsize=7,
            rotation=70,
        )
        display_all_yticks(
            [
                (
                    f"\nL∞".join(i.split("L∞"))
                    if "L∞" in i
                    else f"\nClean".join(i.split("Clean"))
                )
                for i in df.index
            ],
            ax,
            fontsize=7,
            rotation=20,
        )
        ax.set_xlabel("Benchmarks")
        ax.set_ylabel("Robust Accuracy")
        # tight layout
        fig.tight_layout()
        self._save_and_show(fig, "heatmaps/robust_only_correlation", **kwargs)

    def tsne_plot(self, specifier="*", **kwargs):
        df = load_merged_results()
        df_curr = df.drop(
            columns=[
                # "model_registry_name",
                "Top-1 Accuracy Linf 0.001",
                "Top-5 Accuracy Linf 0.001",
                "Top-1 Accuracy Linf 0.00196",
                "Top-5 Accuracy Linf 0.00196",
                "Top-1 Accuracy Linf 0.00392",
                "Top-5 Accuracy Linf 0.00392",
                # "Top-1 Accuracy Clean",
                # "Top-5 Accuracy Clean",
            ]
        )
        # replace all na values with 0
        df_curr = df_curr.fillna(0)
        df_curr = df_curr.drop(
            columns=[
                "average_vision",
                "neural_vision",
                "behavior_vision",
                "engineering_vision",
                "V1",
                "Marques2020",
                "V1-response_magnitude",
                "V1-texture_modulation",
                "V1-surround_modulation",
                "V1-receptive_field_size",
                "V1-response_selectivity",
                "V1-spatial_frequency",
                "V1-orientation",
                "V2",
                "V4",
                "IT",
                "Geirhos2021-error_consistency",
                "Maniquet2024",
                "Ferguson2024",
                "Baker2022",
                "BMD2024",
                "ImageNet-C-top1",
                "Geirhos2021-top1",
                "Hermann2020",
            ]
        )

        # create a t-SNE plot of the data
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=20, random_state=42)
        tsne_results = tsne.fit_transform(df_curr)

        # create a scatter plot of the t-SNE results with the hue based on
        # “Top-1 Accuracy Linf 0.001” column of the df
        fig, ax = plt.subplots(figsize=(6, 4))  # Make the figure more squared
        scatter = ax.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=df["Top-1 Accuracy Linf 0.001"],
            cmap="coolwarm",
            s=100,
            alpha=0.8,
        )
        # save the plot as a pdf
        # ax.set_title("t-SNE Plot of Benchmark Scores")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        # add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Top-1 Accuracy Linf 0.001")
        fig.tight_layout()
        # save the plot
        self._save_and_show(fig, "tsne_plot", **kwargs)
        # show the plot
        plt.show()

        return

    def avg_alignment_vs_robust(self, specifier="*", **kwargs):
        df_all = load_merged_results()
        # get all the columns that contain "Top-1 Accuracy Linf" in them
        robust_cols = df_all.filter(regex="Top-1 Accuracy Linf", axis=None).columns
        for b in [
            "average_vision",
            "neural_vision",
            "behavior_vision",
            "engineering_vision",
        ]:
            df = df_all[[b] + robust_cols.to_list()].dropna()
            # convert all columns to float
            df = df.astype(float)
            x = df[b].values.reshape(-1, 1).astype(float)
            colors = ["#6a0dad", "#228B22", "#FF4500"]
            fig, ax = plt.subplots(figsize=(5, 3))
            for i, c in enumerate(robust_cols):
                y = df[c].values
                color = colors[i % len(colors)]
                # Fit linear model
                model = LinearRegression().fit(x, y)
                # Predictions
                y_pred = model.predict(x)
                # Calculate R^2 value
                r2 = r2_score(y, y_pred)
                # Calculate significance value (p-value)
                n = len(y)
                p = 1  # number of predictors
                df_resid = n - p - 1
                ss_resid = np.sum((y - y_pred) ** 2)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                f_stat = (ss_total - ss_resid) / p / (ss_resid / df_resid)
                p_value = 1 - stats.f.cdf(f_stat, p, df_resid)
                sns.scatterplot(
                    x=df[b],
                    y=df[c],
                    color=color,
                    alpha=0.7,
                    label=f"ϵ={'0.'+c.split('.')[1]} (R²={r2:.3f}, p={p_value:.4f})",
                    ax=ax,
                )
                # Plot regression line
                ax.plot(
                    df[b],
                    y_pred,
                    linestyle="dotted",
                    color=color,
                )
                # Add R² text annotation
                ax.text(
                    np.max(df[b]) * 0.8,
                    np.max(y_pred),
                    f"R²={r2:.3f}",
                    color=color,
                    fontsize=10,
                )
            b_pretty = b.replace("_", " ").title()
            ax.set_xlabel(f"{b_pretty} Alignment Score")
            ax.set_ylabel("Robust Accuracy")
            ax.legend()
            plt.tight_layout()
            self._save_and_show(
                fig,
                f"scatterplots/{b.replace('average', 'avg')}_vs_robust",
                **kwargs,
            )

    def _save_and_show(self, fig, name, paper="rep-align-sec-paper", **kwargs):
        save_to_paper = (
            kwargs["save_to_paper"] if "save_to_paper" in kwargs else self.save_to_paper
        )
        save = kwargs["save"] if "save" in kwargs else self.save
        show_plots = kwargs["show_plots"] if "show_plots" in kwargs else self.show_plots
        # check if fig is a PIL image or plot
        if isinstance(fig, plt.Figure):
            save_method = fig.savefig
            show_method = plt.show
            close_method = plt.close
        # create case for plotly figures
        elif isinstance(fig, go.Figure):
            save_method = fig.write_image
            show_method = fig.show
            close_method = lambda: None
        else:
            save_method = fig.save
            show_method = fig.show
            close_method = fig.close
        # the name passed in contains multiple directories, ensure that all directories in the path exist before saving
        if save:
            base_dir = "figures/" + "/".join(name.split("/")[:-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            save_method(f"figures/{name}.pdf")
        if show_plots:
            show_method()
        if save_to_paper:
            if not os.path.exists(f"../papers/{paper}/{base_dir}"):
                os.makedirs(f"../papers/{paper}/{base_dir}", exist_ok=True)
            save_method(f"../papers/{paper}/figures/{name}.pdf")
        # close the figure to avoid memory leaks
        close_method()

    def table_save(self, df, name, paper="rep-align-sec-paper", **kwargs):
        # check for any underscores in column names or anywhere in the table and replace them with spaces
        df.columns = df.columns.str.replace("_", " ")
        # check the entries in the table
        df = df.map(lambda x: x.replace("_", " ") if isinstance(x, str) else x)
        # get rid of all trailing zeros
        df.to_latex(f"figures/tables/{name}.tex", index=False, **kwargs)
        if self.save_to_paper:
            df.to_latex(
                f"../papers/{paper}/figures/tables/{name}.tex",
                index=False,
            )


def load_robust_results():
    # Load the robust benchmark scores
    df = pd.read_csv("results/evaluation_results_merged_robust.csv")
    # df_linf = pd.read_csv("results/evaluation_results_merged_linf.csv")
    # df = pd.concat([df, df_linf], axis=0)
    for i, eps in enumerate(np.sort(df["Epsilon"].unique())):
        df_curr = df[df["Epsilon"] == eps]
        df_curr = df_curr.drop(
            columns=[
                "Epsilon",
                "Threat Model",
                "Top-1 Subset Clean Accuracy",
                "Top-5 Subset Clean Accuracy",
            ]
        )
        df_curr = df_curr.set_index("Model Name")
        # round epsilon to 5 decimal places
        df_curr.columns = [f"{col} Linf {round(eps, 5)}" for col in df_curr.columns]
        if i == 0:
            robust_df = df_curr
        else:
            robust_df = pd.concat([robust_df, df_curr], axis=1)
    return robust_df


def load_merged_results():
    # Load the merged benchmark scores
    merged_benchmark_scores = pd.read_csv(
        "results/benchmark_scores/benchmark_scores_registry_merged.csv"
    ).drop(columns=["model_name", "og_model_name", "layers"])
    # any cell that is just an X should be replaced with NaN
    merged_benchmark_scores = merged_benchmark_scores.replace("X", np.nan)
    robust_results = load_robust_results()
    merged_benchmark_scores = pd.merge(
        merged_benchmark_scores,
        robust_results,
        left_on="model_registry_name",
        right_index=True,
        how="inner",
    )
    merged_benchmark_scores.to_csv("results/total_scores.csv", index=False)
    merged_benchmark_scores = merged_benchmark_scores.set_index("model_registry_name")
    # drop any column that is all na values
    merged_benchmark_scores = merged_benchmark_scores.dropna(axis=1, how="all")
    merged_benchmark_scores.drop("Kar2019-ost", axis=1, inplace=True)
    return merged_benchmark_scores


def display_all_xticks(labels, ax, fontsize=None, rotation=90):
    # change the x ticks so they all appear and are rotated 60 degrees
    ax.set_xticks([i + 0.5 for i in range(len(labels))])
    if rotation == 90:
        ha = "center"
    elif rotation > 90:
        ha = "left"
    else:
        ha = "right"
    # ensure that the xticks are in the center of the cell
    if fontsize is not None:
        ax.set_xticklabels(labels, rotation=rotation, ha=ha, fontsize=fontsize)
    else:
        ax.set_xticklabels(labels, rotation=rotation, ha=ha)


def display_all_yticks(labels, ax, fontsize=None, rotation=0):
    # change the x ticks so they all appear and are rotated 60 degrees
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    if rotation == 0:
        va = "center"
    elif rotation > 0:
        va = "top"
    else:
        va = "bottom"
    if fontsize is not None:
        ax.set_yticklabels(labels, rotation=rotation, va=va, fontsize=fontsize)
    else:
        ax.set_yticklabels(labels, rotation=rotation, va=va)


def main(plots, specifier, save_to_paper, show_plots, no_save):
    plotter = Plotter(save_to_paper, show_plots, no_save)
    for plot in plots:
        getattr(plotter, plot)(specifier=specifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots",
        type=str,
        help="Plots to make.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--specifier",
        type=str,
        help="Specifier for files to use.",
        default="*",
    )
    parser.add_argument(
        "--save_to_paper",
        action="store_true",
        help="Save to paper directory.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Show plots.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save plots.",
    )
    args = parser.parse_args()
    main(
        args.plots,
        args.specifier,
        args.save_to_paper,
        args.show_plots,
        args.no_save,
    )
