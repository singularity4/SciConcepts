"""figures.py - figures of the corpus and the concept networks (step 4)

SciConcepts.

Input
    Output folders of sci_concepts.py (--input) and concept_network.py (--networks).

Figures (PNG and PDF in <input>/figures/)
    fig1a   papers per month and field (log scale)
    fig1b   papers per category and year
    fig2    MultipartiteRank candidates and concepts per paper and field (dashed lines: medians)
    fig3    15 concepts with the most abstracts in each field
    fig4    new links per month and field
    fig5    metrics of the cumulative network per quarter
    fig6    degree distribution of the final network (logarithmic bins); the degree of a concept is its number of
            distinct co-occurring concepts
    fig7    network of significant links of all fields (network_full, from concept_network.py --all): of the
            NETWORK_FREQUENT concepts with the most abstracts, the NETWORK_SHOWN with the most significant links to
            each other. Size: number of abstracts. Colour: the field with the most abstracts that contain the concept,
            relative to the size of the field. Lines: the NETWORK_LINES significant links between these concepts with
            the largest association strength c_uv / (n_u n_v), where c_uv is the number of abstracts that contain both
            concepts and n_u the number of abstracts that contain u. Labels: the NETWORK_LABELS concepts with the
            most abstracts. Layout: Kamada-Kawai (networkx).

Usage
    python figures.py --input sci_concepts --networks networks
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------- parameters
# Field labels and colours, in the order used in the legends
FIELD_LABELS = {"artificial_intelligence": "AI", "machine_learning": "Machine Learning",
                "social_information_networks": "Networks", "economics": "Economics",
                "quantitative_finance": "Quant. Finance", "physics_society": "Physics and Society"}
FIELD_COLORS = {"artificial_intelligence": "#A51C30", "machine_learning": "#56B4E9",
                "social_information_networks": "#009E73", "economics": "#E69F00",
                "quantitative_finance": "#8C8C8C", "physics_society": "#7B3294"}
FIELD_NAMES = {"artificial_intelligence": "Artificial intelligence", "machine_learning": "Machine learning",
               "social_information_networks": "Social and information networks", "economics": "Economics",
               "quantitative_finance": "Quantitative finance", "physics_society": "Physics and society"}

# Network of significant links (fig7)
NETWORK_FREQUENT = 500        # concepts with the most abstracts from which the shown concepts are selected
NETWORK_SHOWN = 120           # concepts shown: those with the most significant links to each other
NETWORK_LINES = 500           # links drawn: those with the largest association strength
NETWORK_LABELS = 70           # labelled concepts: those with the most abstracts


def gini(values):
    """Return the Gini coefficient of non-negative values."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.sum() == 0:
        return 0.0
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(values) / (n * values.sum()))


def _pyplot():
    """Return matplotlib.pyplot with a non-interactive backend and a common style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "figure.dpi": 150})
    return plt


def _plain_log_axis(ax):
    """Label a logarithmic y axis with plain numbers (1, 10, 100, 1,000) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _save(fig, folder, name):
    """Save a figure as PNG and PDF."""
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"{name}.{extension}"), bbox_inches="tight", dpi=200)


def make_figures(papers, concepts, paper_concepts, links, links_summary, metrics, degree, candidates, folder):
    """Write all figures to folder (inputs as produced by sci_concepts.py and concept_network.py)."""
    plt = _pyplot()
    os.makedirs(folder, exist_ok=True)
    fields = [field for field in FIELD_LABELS if field in set(papers["group"])]
    field_of_category = dict(zip(papers["category"], papers["group"]))
    categories_of_field = {field: sorted(papers.loc[papers["group"] == field, "category"].unique()) for field in fields}
    colors, labels = FIELD_COLORS, FIELD_LABELS

    # fig1a: papers per month and field
    fig, ax = plt.subplots(figsize=(7, 3))
    for field in fields:
        per_month = papers[papers["group"] == field].groupby("month").size()
        ax.plot(pd.to_datetime(per_month.index), per_month.values, color=colors[field], label=f"{labels[field]} ({per_month.sum():,})")
    ax.set_ylabel("papers per month"); ax.set_yscale("log"); _plain_log_axis(ax); ax.legend(fontsize=8, ncol=2)
    _save(fig, folder, "fig1a_papers_per_month"); plt.close(fig)

    # fig1b: papers per category and year
    table = papers.assign(year=papers["month"].str[:4]).groupby(["category", "year"]).size().unstack(fill_value=0)
    table = table.loc[[category for field in fields for category in categories_of_field[field]]]
    fig, ax = plt.subplots(figsize=(8, 3))
    x, width = np.arange(len(table)), 0.8 / table.shape[1]
    shades = plt.cm.Greys(np.linspace(0.35, 0.85, table.shape[1]))
    for position, (year, shade) in enumerate(zip(table.columns, shades)):
        ax.bar(x + (position - (table.shape[1] - 1) / 2) * width, table[year].values, width=width, color=shade, label=year)
    ax.set_xticks(x); ax.set_xticklabels(table.index, rotation=60, ha="right")
    for tick, category in zip(ax.get_xticklabels(), table.index):
        tick.set_color(colors[field_of_category[category]])
    ax.set_ylabel("papers"); ax.set_yscale("log"); _plain_log_axis(ax); ax.legend(fontsize=8, ncol=4)
    _save(fig, folder, "fig1b_papers_per_category"); plt.close(fig)

    # fig2: candidates and concepts per paper
    candidates_per_paper = candidates.groupby("paper").size().reindex(range(len(papers)), fill_value=0).values
    concepts_per_paper = paper_concepts.groupby("paper").size().reindex(range(len(papers)), fill_value=0).values
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, values, title in [(axes[0], candidates_per_paper, "MultipartiteRank candidates per paper"), (axes[1], concepts_per_paper, "concepts per paper")]:
        bins = np.arange(0, np.percentile(values, 99.5) + 2) - 0.5
        for field in fields:
            field_values = values[papers["group"].values == field]
            if len(field_values) == 0:
                continue
            ax.hist(field_values, bins=bins, weights=np.ones(len(field_values)) / len(field_values), histtype="step", color=colors[field], lw=1.4,
                    label=f"{labels[field]} (median {np.median(field_values):.0f})")
            ax.axvline(np.median(field_values), color=colors[field], ls="--", lw=0.8)
        ax.set_xlabel(title); ax.legend(fontsize=7)
    axes[0].set_ylabel("fraction of papers")
    _save(fig, folder, "fig2_concepts_per_paper"); plt.close(fig)

    # fig3: 15 concepts with the most abstracts in each field
    fig, axes = plt.subplots(1, len(fields), figsize=(3.3 * len(fields), 4.2))
    for ax, field in zip(np.atleast_1d(axes), fields):
        top = concepts.nlargest(15, f"docs_{field}")
        ax.barh(top["label"][::-1], top[f"docs_{field}"][::-1], color=colors[field])
        ax.set_title(labels[field]); ax.tick_params(axis="y", labelsize=7); ax.set_xlabel("papers")
    fig.tight_layout(); _save(fig, folder, "fig3_top_concepts_per_group"); plt.close(fig)

    # fig4: new links per month and field (a link is new in its first month)
    fig, ax = plt.subplots(figsize=(7, 3))
    new_links = links.merge(links_summary[["c1", "c2", "first_month"]], on=["c1", "c2"])
    new_links = new_links[new_links["month"] == new_links["first_month"]]
    for field in fields:
        per_month = new_links[new_links["group"] == field].groupby("month").size()
        if len(per_month):
            ax.plot(pd.to_datetime(per_month.index), per_month.values, color=colors[field], label=labels[field])
    ax.set_ylabel("new links per month"); ax.set_yscale("log"); _plain_log_axis(ax); ax.legend(fontsize=8, ncol=2)
    _save(fig, folder, "fig4_new_links_per_month"); plt.close(fig)

    # fig5: metrics of the cumulative network per quarter
    columns = [("concepts", "concepts"), ("links", "links"), ("density", "density"), ("average_degree", "average degree"),
               ("degree_gini", "degree Gini"), ("largest_component_share", "largest component share"),
               ("clustering_sampled", "clustering (sampled)"), ("new_link_share", "new links / all links")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6))
    x = np.arange(len(metrics)); step = max(1, len(x) // 4)
    for ax, (column, label) in zip(axes.ravel(), columns):
        ax.plot(x, metrics[column].values, color="#4D4D4D", marker="o", ms=2.5, lw=1)
        ax.set_title(label); ax.set_xticks(x[::step]); ax.set_xticklabels(metrics["quarter"].values[::step], fontsize=7)
    fig.tight_layout(); _save(fig, folder, "fig5_network_metrics_per_quarter"); plt.close(fig)

    # fig6: degree distribution of the final network
    linked = degree[degree > 0]
    fig, ax = plt.subplots(figsize=(4, 3))
    bins = np.unique(np.logspace(0, np.log10(linked.max() + 1), 30).astype(int))
    density, edges = np.histogram(linked, bins=bins, density=True)
    centres = np.sqrt(edges[:-1] * edges[1:])
    ax.loglog(centres[density > 0], density[density > 0], "o", color="#A51C30", ms=3)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("degree"); ax.set_ylabel("probability density"); ax.set_title(f"{len(linked):,} concepts, Gini {gini(linked):.2f}")
    _save(fig, folder, "fig6_degree_distribution"); plt.close(fig)


def network_figure(abstract_concept, network_concepts, significant_links, field_of_paper, label_of_key, folder):
    """Draw the network of significant links of all fields (fig7; selection and drawing as in the module docstring).

    abstract_concept   table with columns paper, concept_id (network_full/abstract_concept.parquet)
    network_concepts   table with columns concept_id, key (network_full/concepts.parquet)
    significant_links  arrays u, v and C from network_full/significant_links.npz
    field_of_paper     field of each paper (row order of papers_small)
    label_of_key       label of each concept key (concepts.csv); a key without a label is shown as it is
    """
    import networkx as nx
    from matplotlib import font_manager
    from scipy.sparse import csr_matrix
    plt = _pyplot()
    os.makedirs(folder, exist_ok=True)
    n_concepts = int(significant_links["C"])
    paper, concept = abstract_concept["paper"].values, abstract_concept["concept_id"].values
    abstracts = np.bincount(concept, minlength=n_concepts)

    # concepts: of the NETWORK_FREQUENT most frequent, the NETWORK_SHOWN with the most significant links to each other
    frequent = np.argsort(-abstracts, kind="stable")[:NETWORK_FREQUENT]
    is_frequent = np.zeros(n_concepts, bool); is_frequent[frequent] = True
    u, v = significant_links["u"], significant_links["v"]
    among_frequent = is_frequent[u] & is_frequent[v]
    degree = np.bincount(np.r_[u[among_frequent], v[among_frequent]], minlength=n_concepts)
    shown = frequent[np.argsort(-degree[frequent], kind="stable")[:NETWORK_SHOWN]]
    position_of = np.full(n_concepts, -1); position_of[shown] = np.arange(len(shown))

    # co-occurrences of the shown concepts; colour: the field with the most abstracts that contain the concept,
    # relative to the size of the field
    rows = position_of[concept] >= 0
    incidence = csr_matrix((np.ones(rows.sum()), (paper[rows], position_of[concept[rows]])),
                           shape=(len(field_of_paper), len(shown)))
    cooccurrences = (incidence.T @ incidence).toarray()
    fields = [field for field in FIELD_NAMES if field in set(field_of_paper)]
    field_number = pd.Series(field_of_paper).map({field: number for number, field in enumerate(fields)}).values
    abstracts_per_field = np.bincount(field_number, minlength=len(fields))
    field_counts = np.zeros((len(shown), len(fields)))
    np.add.at(field_counts, (position_of[concept[rows]], field_number[paper[rows]]), 1)
    field_of_concept = np.array(fields)[np.argmax(field_counts / abstracts_per_field, axis=1)]

    # links: the NETWORK_LINES significant links between shown concepts with the largest association strength
    among_shown = (position_of[u] >= 0) & (position_of[v] >= 0)
    first, second = position_of[u[among_shown]], position_of[v[among_shown]]
    size = abstracts[shown]
    strength = cooccurrences[first, second] / (size[first].astype(float) * size[second])
    strongest = np.argsort(-strength, kind="stable")[:NETWORK_LINES]
    first, second = first[strongest], second[strongest]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(shown)))
    graph.add_edges_from(zip(first.tolist(), second.tolist()))
    layout = nx.kamada_kawai_layout(graph)
    xy = np.array([layout[node] for node in range(len(shown))])

    family = "Inter" if "Inter" in {font.name for font in font_manager.fontManager.ttflist} else None
    fig, ax = plt.subplots(figsize=(7, 6.6))
    for a, b in zip(first, second):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#D2D0C9", lw=0.5, zorder=1)
    area = 12 + 250 * size / size.max()
    ax.scatter(xy[:, 0], xy[:, 1], s=area, c=[FIELD_COLORS[field] for field in field_of_concept], edgecolors="white",
               linewidths=0.6, zorder=2)
    keys = network_concepts.set_index("concept_id")["key"].loc[shown].values
    for node in np.argsort(-size, kind="stable")[:NETWORK_LABELS]:
        ax.annotate(label_of_key.get(keys[node], keys[node]), xy[node], xytext=(0, 3 + np.sqrt(area[node]) / 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5, family=family, zorder=3)
    for field in fields:
        ax.scatter([], [], s=30, color=FIELD_COLORS[field], label=FIELD_NAMES[field])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, prop={"family": family, "size": 7}, handletextpad=0.3,
              columnspacing=2.0)
    ax.set_axis_off()
    _save(fig, folder, "fig7_concept_network"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="sci_concepts", help="output folder of sci_concepts.py")
    parser.add_argument("--networks", default="networks", help="output folder of concept_network.py")
    args = parser.parse_args()
    folder = os.path.dirname(sorted(glob.glob(os.path.join(args.input, "**", "papers.parquet"), recursive=True))[0])
    network_folder = os.path.dirname(sorted(glob.glob(os.path.join(args.networks, "**", "links_summary.parquet"), recursive=True))[0])
    read = lambda name: pd.read_parquet(os.path.join(folder, name))
    papers, concepts = read("papers.parquet"), pd.read_csv(os.path.join(folder, "concepts.csv"))
    links_summary = pd.read_parquet(os.path.join(network_folder, "links_summary.parquet"))
    degree = np.bincount(np.r_[links_summary["c1"].values, links_summary["c2"].values], minlength=len(concepts))
    make_figures(papers, concepts, read("paper_concepts.parquet"), pd.read_parquet(os.path.join(network_folder, "links_by_month_group.parquet")),
                 links_summary, pd.read_csv(os.path.join(network_folder, "network_metrics_per_quarter.csv")), degree,
                 read("candidates.parquet"), os.path.join(folder, "figures"))
    full_network = sorted(glob.glob(os.path.join(args.networks, "**", "network_full", "significant_links.npz"), recursive=True))
    if full_network:
        full_folder = os.path.dirname(full_network[0])
        network_figure(pd.read_parquet(os.path.join(full_folder, "abstract_concept.parquet")),
                       pd.read_parquet(os.path.join(full_folder, "concepts.parquet")), np.load(full_network[0]),
                       papers["group"].values, dict(zip(concepts["key"], concepts["label"])), os.path.join(folder, "figures"))


if __name__ == "__main__":
    main()
