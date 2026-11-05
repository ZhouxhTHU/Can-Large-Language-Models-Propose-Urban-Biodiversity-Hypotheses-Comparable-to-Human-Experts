# Script for plotting open-ended expert feedback without rerunning API.
# It reads the already coded file: coded_responses_empirical.csv
# and combines pros and cons into the two-panel Figure 3.

import os
import ast
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# ----------------------------------------------------------
# Compatibility patch:
# This keeps seaborn usable with newer matplotlib versions.
# ----------------------------------------------------------
import matplotlib as mpl
import matplotlib.cm as cm

if not hasattr(cm, "register_cmap"):
    def _register_cmap_compat(name=None, cmap=None, **kwargs):
        if cmap is None:
            cmap = kwargs.get("cmap", None)

        if cmap is None:
            return

        if name is None:
            name = getattr(cmap, "name", None)

        try:
            mpl.colormaps.register(cmap, name=name, force=True)
        except ValueError:
            pass

    cm.register_cmap = _register_cmap_compat

import seaborn as sns
from thefuzz import process, fuzz


# ==========================================
# 1. Configuration
# ==========================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 优先读取 Outputs 里的 coded_responses_empirical.csv
# 如果没有，则读取当前文件夹下的 coded_responses_empirical.csv
CANDIDATE_CODED_FILES = [
    os.path.join(OUTPUT_DIR, "coded_responses_empirical.csv"),
    "coded_responses_empirical.csv"
]

OUTPUT_STEM = "open_feedback_ab"

SENTIMENT_ORDER = ["Positive", "Mixed", "Neutral", "Negative"]

# Explicitly merge wording variants that describe the same concept. These
# aliases are applied before fuzzy matching so that low-frequency synonyms are
# combined consistently in the visualization.
PROS_CATEGORY_ALIASES = {
    "High Novelty": "Highly Novel",
    "Highly Novel": "Highly Novel",
    "Generates Novel Hypotheses": "Highly Novel",
    "Highly Structured": "Well-Structured",
    "Well-Structured": "Well-Structured",
}

CONS_CATEGORY_ALIASES = {
    "Overly Complex": "Overly Complex",
    "Too Complicated": "Overly Complex",
    "Too Complex": "Overly Complex",
    "Vague Formulation": "Vague Formulation",
    "Lack Of Specificity": "Vague Formulation",
    "Too Long": "Excessive Length",
    "Excessive Length": "Excessive Length",
    "Difficult To Test": "Difficult To Test",
    "Hard To Test": "Difficult To Test",
}


# ==========================================
# 2. Load coded data
# ==========================================

def find_coded_file() -> str:
    for file_path in CANDIDATE_CODED_FILES:
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(
        "Cannot find coded_responses_empirical.csv. "
        "Please put it either in ./Outputs/ or in the current working directory."
    )


def parse_list_cell(value):
    """
    Convert a CSV cell back to a Python list.

    The coded CSV usually stores pros and cons as strings like:
    "['Highly Novel', 'Structured']"
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() == "nan":
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
        except Exception:
            # Fallback: treat as one item
            return [value]

    return [str(value)]


def load_coded_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required_cols = [
        "sentiment_category",
        "pros",
        "cons"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns in {file_path}: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    df["pros"] = df["pros"].apply(parse_list_cell)
    df["cons"] = df["cons"].apply(parse_list_cell)

    return df


# ==========================================
# 3. Semantic clustering
# ==========================================

def group_synonyms(
    items: list[str],
    threshold: int = 75,
    category_aliases: dict[str, str] = None,
) -> list[tuple[str, int]]:
    if not items:
        return []

    cleaned_items = [
        item.strip().title()
        for item in items
        if item is not None and str(item).strip() != ""
    ]

    if not cleaned_items:
        return []

    if category_aliases:
        cleaned_items = [
            category_aliases.get(item, item)
            for item in cleaned_items
        ]

    raw_counts = Counter(cleaned_items)
    sorted_unique_items = [
        item for item, count in raw_counts.most_common()
    ]

    canonical_mapping = {}

    for item in sorted_unique_items:

        established_terms = list(set(canonical_mapping.values()))

        if established_terms:
            best_match, score = process.extractOne(
                item,
                established_terms,
                scorer=fuzz.token_sort_ratio
            )

            if score >= threshold:
                canonical_mapping[item] = best_match
                continue

        canonical_mapping[item] = item

    clustered_items = [
        canonical_mapping[item]
        for item in cleaned_items
    ]

    return Counter(clustered_items).most_common(10)


# ==========================================
# 4. Plotting
# ==========================================

def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left"
    )


def plot_bar_values_vertical(ax):
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.08,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11
        )


def plot_bar_values_horizontal(ax):
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + 0.04,
            y,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10.5
        )


def visualize_findings_ab(df: pd.DataFrame, output_dir: str = "./Outputs"):
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans"
            ],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    # Pros and cons counts
    all_pros = [
        item
        for sublist in df["pros"]
        for item in sublist
    ]

    all_cons = [
        item
        for sublist in df["cons"]
        for item in sublist
    ]

    pros_counts = group_synonyms(
        all_pros,
        threshold=75,
        category_aliases=PROS_CATEGORY_ALIASES,
    )
    cons_counts = group_synonyms(
        all_cons,
        threshold=75,
        category_aliases=CONS_CATEGORY_ALIASES,
    )

    pros_df = pd.DataFrame(
        pros_counts,
        columns=["descriptor", "frequency"]
    )

    cons_df = pd.DataFrame(
        cons_counts,
        columns=["descriptor", "frequency"]
    )

    # Save counts for checking
    counts_output_path = os.path.join(
        output_dir,
        f"{OUTPUT_STEM}_counts.csv"
    )

    counts_for_save = pd.concat(
        [
            pros_df.assign(panel="a_pros")[["panel", "descriptor", "frequency"]],
            cons_df.assign(panel="b_cons")[["panel", "descriptor", "frequency"]],
        ],
        ignore_index=True
    )

    counts_for_save.to_csv(
        counts_output_path,
        index=False
    )

    # Figure 3: the former sentiment panel is omitted.
    # Keep the existing pros/cons aggregation, bar types and colours.
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.subplots_adjust(wspace=0.85, top=0.84, bottom=0.15)

    # Panel a: pros
    if not pros_df.empty:
        sns.barplot(
            data=pros_df,
            x="frequency",
            y="descriptor",
            ax=ax_b,
            color="#0072B2",
            saturation=1
        )

    ax_b.set_title("Most Frequently Cited Pros", loc="left")
    ax_b.set_xlabel("Frequency")
    ax_b.set_ylabel("")
    ax_b.set_xlim(
        0,
        max(
            2.2,
            pros_df["frequency"].max() + 0.45
            if not pros_df.empty else 2.2
        )
    )

    #plot_bar_values_horizontal(ax_b)
    add_panel_label(ax_b, "a", x=-0.08, y=1.08)

    # Panel b: cons
    if not cons_df.empty:
        sns.barplot(
            data=cons_df,
            x="frequency",
            y="descriptor",
            ax=ax_c,
            color="#D55E00",
            saturation=1
        )

    ax_c.set_title("Most Frequently Cited Cons", loc="left")
    ax_c.set_xlabel("Frequency")
    ax_c.set_ylabel("")
    ax_c.set_xlim(
        0,
        max(
            2.2,
            cons_df["frequency"].max() + 0.45
            if not cons_df.empty else 2.2
        )
    )

    #plot_bar_values_horizontal(ax_c)
    add_panel_label(ax_c, "b", x=-0.08, y=1.08)

    # Make grid less visually heavy
    for ax in [ax_b, ax_c]:
        ax.grid(axis="y", color="0.90", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # For horizontal bar charts, vertical grid is more useful
    for ax in [ax_b, ax_c]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.grid(axis="x", color="0.90", linewidth=0.7)
        ax.grid(axis="y", visible=False)

    png_path = os.path.join(
        output_dir,
        f"{OUTPUT_STEM}.png"
    )

    pdf_path = os.path.join(
        output_dir,
        f"{OUTPUT_STEM}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved combined figure:")
    print(png_path)
    print(pdf_path)

    print("\nSaved counts table:")
    print(counts_output_path)


# ==========================================
# 5. Main
# ==========================================

if __name__ == "__main__":

    coded_file = find_coded_file()

    print(f"Loading coded data from: {coded_file}")

    analyzed_df = load_coded_data(coded_file)

    print(f"Loaded {len(analyzed_df)} coded responses.")

    visualize_findings_ab(
        df=analyzed_df,
        output_dir=OUTPUT_DIR
    )
