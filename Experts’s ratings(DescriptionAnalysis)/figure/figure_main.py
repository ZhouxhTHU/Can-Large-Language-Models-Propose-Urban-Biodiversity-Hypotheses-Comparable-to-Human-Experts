import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# Configuration
# ======================================

input_file = "../Hypothesis_Analysis_Result.xlsx"

plot_dir = "figures"
os.makedirs(plot_dir, exist_ok=True)

dimensions = ["Novelty", "Significance", "Testability"]

SOURCE_COLORS = {
    "Human": "#56B4E9",
    "LLM": "#E69F00",
    "Tie": "#CC79A7",
}

DIMENSION_COLORS = {
    "Novelty": "#56B4E9",
    "Significance": "#E69F00",
    "Testability": "#CC79A7",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ======================================
# Load the existing results from Excel
# ======================================

# Used for Figure 1: Cross_HH_vs_HL
cross_df = pd.read_excel(
    input_file,
    sheet_name="Cross_HH_vs_HL",
    index_col=0
)

# Used for Figures 2 and 3: Expert_Cross_Level
expert_cross_df = pd.read_excel(
    input_file,
    sheet_name="Expert_Cross_Level"
)


# ======================================
# Figure 1: cross-group HH vs HL selection results
# The y-axis shows the number of wins rather than proportions
# ======================================

cross_plot_df = cross_df[["HH_win", "HL_win", "Tie"]].copy()

x = np.arange(len(cross_plot_df.index))
width = 0.24

plt.figure(figsize=(9.6, 4.8))

plt.bar(
    x - width,
    cross_plot_df["HH_win"],
    width=width,
    label="Human-proposed\nhypothesis selected",
    color=SOURCE_COLORS["Human"],
    edgecolor="black",
    linewidth=0.5
)

plt.bar(
    x,
    cross_plot_df["HL_win"],
    width=width,
    label="LLM-generated\nhypothesis selected",
    color=SOURCE_COLORS["LLM"],
    edgecolor="black",
    linewidth=0.5
)

plt.bar(
    x + width,
    cross_plot_df["Tie"],
    width=width,
    label="Tie",
    color=SOURCE_COLORS["Tie"],
    edgecolor="black",
    linewidth=0.5
)

plt.xticks(x, cross_plot_df.index)
plt.xlabel("Dimension")
plt.ylabel("Number of selections")
plt.title("Selection counts")

plt.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    labelspacing=1.0,
)

plt.tight_layout()

fig1_path = os.path.join(
    plot_dir,
    "figure1_cross_source_counts.png"
)

plt.savefig(fig1_path, dpi=600, bbox_inches="tight")
plt.close()


# ======================================
# Reshape into a long-form expert-by-dimension table
# Used for Figures 2 and 3
# ======================================

expert_dim_results = []

for _, row in expert_cross_df.iterrows():

    expert_id = row["Expert_ID"]

    for dim in dimensions:

        hh_win = row[f"{dim}_Cross_HH_win"]
        hl_win = row[f"{dim}_Cross_HL_win"]
        tie = row[f"{dim}_Cross_Tie"]

        total = hh_win + hl_win + tie
        decisive = hh_win + hl_win

        expert_dim_results.append({
            "Expert_ID": expert_id,
            "Dimension": dim,

            # Proportion of decisive choices in which the LLM was selected
            "LLM_Share_Among_Decisive_Dim": (
                hl_win / decisive if decisive > 0 else np.nan
            ),

            # Proportion of ties in cross-group comparisons
            "Cross_Tie_Rate_Dim": (
                tie / total if total > 0 else np.nan
            )
        })

expert_dim_df = pd.DataFrame(expert_dim_results)


# ======================================
# General-purpose function: boxplot with jittered points
# ======================================

def plot_boxplot_with_points(
    df,
    value_col,
    ylabel,
    title,
    output_path,
    ref_line=None,
    ylim=(0, 1),
    show_expert_lines=False,
    ax=None,
):
    standalone = ax is None
    if standalone:
        plt.figure(figsize=(7.2, 4.8))
    else:
        plt.sca(ax)

    positions = np.arange(1, len(dimensions) + 1)

    data = [
        df[df["Dimension"] == dim][value_col].dropna().values
        for dim in dimensions
    ]

    # Boxplot
    plt.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        boxprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"}
    )

    # Optional: connect the three dimensions for each expert
    if show_expert_lines:

        for expert_id, sub_df in df.sourceby("Expert_ID"):

            sub_df = sub_df.set_index("Dimension").reindex(dimensions)

            y_values = sub_df[value_col].values

            if np.all(pd.notna(y_values)):
                plt.plot(
                    positions,
                    y_values,
                    linewidth=0.8,
                    alpha=0.25
                )

    # Jittered points: each point represents one expert
    rng = np.random.default_rng(2026)

    for pos, dim in zip(positions, dimensions):

        values = df[df["Dimension"] == dim][value_col].dropna().values

        jitter = rng.normal(
            loc=0,
            scale=0.04,
            size=len(values)
        )

        plt.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=36,
            color=DIMENSION_COLORS[dim],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85
        )

    # Reference line
    if ref_line is not None:
        plt.axhline(
            y=ref_line,
            linewidth=1,
            linestyle="--"
        )

    plt.xticks(positions, dimensions)
    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=20 if standalone else 16, pad=12)

    if ylim is not None:
        plt.ylim(*ylim)

    # Identical canvas and axes bounds for panels b and c, regardless of
    # their different title and y-label lengths. Avoid per-panel cropping.
    if standalone:
        plt.subplots_adjust(left=0.21, right=0.97, bottom=0.20, top=0.76)
        plt.savefig(output_path, dpi=600, bbox_inches=None)
        plt.close()


# ======================================
# Figure 2: boxplot of expert-level LLM share across the three dimensions
# ======================================

fig2_path = os.path.join(
    plot_dir,
    "figure2_expert_llm_share_by_dimension_boxplot.png"
)

# Figure 2: LLM share
plot_boxplot_with_points(
    df=expert_dim_df,
    value_col="LLM_Share_Among_Decisive_Dim",
    ylabel="Proportion of LLM-generated\nhypotheses selected",
    title="LLM-generated hypothesis selections\nby human experts",
    output_path=fig2_path,
    ref_line=0.5,
    ylim=(0, 1),
    show_expert_lines=False
)

# ======================================
# Figure 3: boxplot of expert-level tie rate across the three dimensions
# ======================================

fig3_path = os.path.join(
    plot_dir,
    "figure3_expert_tie_rate_by_dimension_boxplot.png"
)

# Figure 3: Tie rate
plot_boxplot_with_points(
    df=expert_dim_df,
    value_col="Cross_Tie_Rate_Dim",
    ylabel="Tie rate",
    title="Tie rates among human experts",
    output_path=fig3_path,
    ref_line=None,
    ylim=(0, 1),
    show_expert_lines=False
)


# ======================================
# Save the processed expert-by-dimension data
# ======================================

expert_dim_data_path = os.path.join(
    plot_dir,
    "expert_dimension_plot_data.xlsx"
)

# Label-only reruns must not overwrite the existing processed data workbook.
if not os.path.exists(expert_dim_data_path):
    expert_dim_df.to_excel(expert_dim_data_path, index=False)


# ======================================
# Print the output paths
# ======================================

print("\nSaved figures:")
print(fig1_path)
print(fig2_path)
print(fig3_path)

print("\nSaved plot data:")
print(expert_dim_data_path)


# ======================================
# Create the combined a/b/c figure
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

CAPTION = (
    "Figure 1. Human expert selections (n = 30). "
    "a, Numbers of judgements favouring human-proposed hypotheses, "
    "LLM-generated hypotheses, or a tie in each evaluation dimension. "
    "b, Expert-level shares of LLM selections among decisive cross-source "
    "judgements, excluding ties. "
    "c, Tie rates among human experts in cross-source judgements."
)


def combine_figures():
    # Draw directly on the final canvas so panel text is not shrunk as images.
    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    }):
        fig = plt.figure(figsize=(10.5, 8.8))
        fig.suptitle("Human expert selections (n = 30)", fontsize=21, y=0.98)
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], left=0.11,
                               right=0.98, bottom=0.10, top=0.88,
                               hspace=0.66, wspace=0.43)
        top_grid = grid[0, :].subgridspec(1, 2, width_ratios=[3.2, 1], wspace=0.04)
        ax_a = fig.add_subplot(top_grid[0, 0])
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = fig.add_subplot(grid[1, 1])
        for offset, column, label, colour in [
            (-width, "HH_win", "Human-proposed\nhypothesis selected", SOURCE_COLORS["Human"]),
            (0, "HL_win", "LLM-generated\nhypothesis selected", SOURCE_COLORS["LLM"]),
            (width, "Tie", "Tie", SOURCE_COLORS["Tie"]),
        ]:
            ax_a.bar(x + offset, cross_plot_df[column], width=width,
                     label=label, color=colour, edgecolor="black", linewidth=0.5)
        ax_a.set_xticks(x, labels=cross_plot_df.index)
        ax_a.set_xlabel("Dimension")
        ax_a.set_ylabel("Number of selections")
        ax_a.set_title("Selection counts", pad=12)
        ax_a.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1),
                    borderaxespad=0, labelspacing=1.0, handlelength=1.5)
        plot_boxplot_with_points(
            df=expert_dim_df, value_col="LLM_Share_Among_Decisive_Dim",
            ylabel="Proportion of LLM-generated\nhypotheses selected",
            title="LLM-generated hypothesis\nselections by human experts",
            output_path=None, ref_line=0.5, ylim=(0, 1), ax=ax_b,
        )
        plot_boxplot_with_points(
            df=expert_dim_df, value_col="Cross_Tie_Rate_Dim",
            ylabel="Tie rate", title="Tie rates among\nhuman experts",
            output_path=None, ref_line=None, ylim=(0, 1), ax=ax_c,
        )
        for label, ax in zip(("a", "b", "c"), (ax_a, ax_b, ax_c)):
            ax.text(-0.16, 1.19, label, transform=ax.transAxes,
                    fontsize=20, fontweight="bold", ha="left", va="bottom")
        assert abs(ax_b.get_position().width - ax_c.get_position().width) < 1e-12
        assert abs(ax_b.get_position().height - ax_c.get_position().height) < 1e-12

    output_png = os.path.join(
        FIGURE_DIR,
        "figureX_descriptive_outcomes_cross_source_comparisons.png",
    )
    output_pdf = os.path.join(
        FIGURE_DIR,
        "figureX_descriptive_outcomes_cross_source_comparisons.pdf",
    )
    caption_file = os.path.join(
        FIGURE_DIR,
        "figureX_descriptive_outcomes_cross_source_comparisons_caption.txt",
    )

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    with open(caption_file, "w", encoding="utf-8") as file:
        file.write(textwrap.fill(CAPTION, width=100))
        file.write("\n")

    print("\nSaved combined figure:")
    print(output_png)
    print(output_pdf)
    print("\nSaved caption:")
    print(caption_file)


combine_figures()
