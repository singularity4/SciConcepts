"""concept_network.py - concept co-occurrence network and networks of significant links (step 2)

SciConcepts.

Input
    Output of sci_concepts.py (--input): papers_small.csv.gz (papers in corpus order), concepts.csv,
    paper_concepts.parquet and candidates.parquet (column paper = row in papers_small).

Co-occurrence network (all fields together)
    Two concepts (concepts.csv) are linked when they appear in the same abstract. Links are counted per month and field
    (number of abstracts containing the pair), so that any period or field can be selected. A link is new in the month
    in which its two concepts first appear together.
    Metrics of the cumulative network (all links first observed up to the end of each quarter): number of concepts with
    at least one link, number of links, density, average degree, Gini coefficient of the degrees, fraction of linked
    concepts in the largest connected component, average clustering coefficient over a random sample of
    CLUSTERING_SAMPLE concepts with at least two links (seed SEED), and the number and fraction of new links in the
    quarter.

Networks of significant links (one per field; with --all also one for all fields together)
    Concepts: candidate keys from the papers in the network after thesaurus version 2 (data/thesaurus_v2.csv: version 1
    with 11 further generic words removed). A key is a concept if it appears in a second abstract within WINDOW_MONTHS
    (12) months of its first use, provided that its first use is no later than the last month in which pairs can be
    evaluated.
    No minimum number of abstracts over the whole corpus is required.
    Significant links: a pair of concepts (u, v) first seen together in an abstract in month t is evaluated over
    months t to t + 12. It is a significant link if it appears in n_uv >= MIN_COOCCURRENCES (2) abstracts in these
    months and
        n_uv >= MIN_LIFT (3) x E_uv,    E_uv = sum over months m = t .. t + 12 and fields f of n_u(m, f) n_v(m, f) / N(m, f),
    where n_u(m, f) is the number of abstracts in field f and month m that contain u, and N(m, f) is the total number of
    abstracts in field f and month m. E_uv is the expected number of co-occurrences if u and v occur independently, and
    n_uv / E_uv is the lift (its logarithm is the pointwise mutual information). A field network uses only the abstracts
    in that field. The link enters the network in month t. Pairs first seen in the last 12 months cannot be evaluated.
    False discovery rate (reported, but not used to select links): for pairs first seen from FDR_REPORT_FROM onwards
    and for each lift threshold in FDR_LIFT_GRID, the expected number of pairs that pass by chance,
    sum over pairs of P(Poisson(E_uv) >= max(2, lift E_uv)), divided by the number of pairs that pass.

Outputs (--out)
    cooccurrence/links_by_month_group.parquet        c1, c2 (concept_id of the two concepts, c1 < c2), month, group
                                                     (field), count (abstracts)
    cooccurrence/links_summary.parquet               c1, c2, weight (abstracts containing the pair), first_month
    cooccurrence/network_metrics_per_quarter.csv     metrics of the cumulative network per quarter
    network_fields/<field>/abstract_concept.parquet  paper (row in papers_small), concept_id
    network_fields/<field>/concepts.parquet          concept_id, key, first_month, abstracts_12m (abstracts in the
                                                     first 12 months)
    network_fields/<field>/significant_links.npz     u, v (concept_id), keys (u C + v), first_month, n_12m,
                                                     expected_12m, lift, C (number of concepts)
    network_full/                                    the same files for all fields together (--all)
    network_summary.json                             size of the co-occurrence network; concepts, links and false
                                                     discovery rates per network of significant links
    Months in the networks of significant links are counted from FIRST_MONTH (0 = January 2021).

Usage
    python concept_network.py --input sci_concepts --out networks --all
    python concept_network.py --input sci_concepts --out networks --quick      (small clustering sample for testing)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson

# ---------------------------------------------------------------------------------------------------------- parameters
# Order of the fields in the cell indices and output tables (as in sci_concepts.py)
FIELD_ORDER = ["artificial_intelligence", "machine_learning", "social_information_networks", "economics",
               "quantitative_finance", "physics_society"]
FIRST_MONTH = "2021-01"       # first month of the corpus (month 0)
LAST_MONTH = "2026-10"        # last month of the corpus

# Co-occurrence network
SEED = 42                     # seed for the clustering sample
CLUSTERING_SAMPLE = 2000      # concepts sampled for the average clustering coefficient

# Networks of significant links
THESAURUS = "thesaurus_v2.csv"   # in data/: version 1 with 11 additional generic words removed
WINDOW_MONTHS = 12            # entry: second abstract within 12 months of the first;
                              # test window of a pair: months t to t + 12, t = month of first co-occurrence
MIN_COOCCURRENCES = 2         # minimum number of abstracts with the pair in its test window
MIN_LIFT = 3.0                # minimum ratio of observed to expected co-occurrences
FDR_LIFT_GRID = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]   # lift thresholds at which the false discovery rate is reported
FDR_REPORT_FROM = "2023-01"   # first month of the pairs used for the reported false discovery rate
PAIRS_PER_BLOCK = 200000      # pairs per block when computing E_uv (limits memory)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def month_index(month, first=FIRST_MONTH):
    """Return the number of months since the first month of the corpus ('2021-01' -> 0); month as 'YYYY-MM' or 'YYYY-MM-DD'."""
    return (int(month[:4]) - int(first[:4])) * 12 + int(month[5:7]) - int(first[5:7])


def build_links(paper_concepts, papers):
    """Count the abstracts containing each pair of concepts per month and field. Returns these counts, a summary per
    pair and the list of months."""
    months = sorted(papers["month"].unique())
    month_number = {month: number for number, month in enumerate(months)}
    field_number = {field: number for number, field in enumerate(FIELD_ORDER)}
    month_of_paper = papers["month"].map(month_number).values
    field_of_paper = papers["group"].map(field_number).values
    n_concepts, n_months, n_fields = int(paper_concepts["concept_id"].max()) + 1, len(months), len(FIELD_ORDER)
    codes = []                    # code for a (pair, month, field): ((first C + second) M + month) F + field
    for paper, concept_ids in paper_concepts.groupby("paper")["concept_id"]:
        concept_ids = np.unique(concept_ids.values)
        if len(concept_ids) < 2:
            continue
        first, second = np.triu_indices(len(concept_ids), k=1)
        pair = concept_ids[first].astype(np.int64) * n_concepts + concept_ids[second]
        codes.append((pair * n_months + month_of_paper[paper]) * n_fields + field_of_paper[paper])
    codes, counts = np.unique(np.concatenate(codes), return_counts=True)
    field = codes % n_fields
    month = (codes // n_fields) % n_months
    pair = codes // (n_fields * n_months)
    links = pd.DataFrame({"c1": (pair // n_concepts).astype(np.int32), "c2": (pair % n_concepts).astype(np.int32),
                          "month": np.array(months)[month], "group": np.array(list(FIELD_ORDER))[field],
                          "count": counts.astype(np.int32)})
    summary = links.groupby(["c1", "c2"]).agg(weight=("count", "sum"), first_month=("month", "min")).reset_index()
    return links, summary, months


def gini(values):
    """Return the Gini coefficient of non-negative values."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.sum() == 0:
        return 0.0
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(values) / (n * values.sum()))


def network_metrics(summary, n_concepts):
    """Compute the metrics of the cumulative network at the end of each quarter. Also returns the final adjacency
    matrix and degrees."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rng = np.random.default_rng(SEED)
    quarter_of = lambda month: f"{month[:4]}Q{(int(month[5:7]) - 1) // 3 + 1}"
    summary = summary.assign(quarter=summary["first_month"].map(quarter_of))
    rows = []
    for quarter in sorted(summary["quarter"].unique()):
        links = summary[summary["quarter"] <= quarter]
        n_new = int((summary["quarter"] == quarter).sum())
        adjacency = coo_matrix((np.ones(len(links)), (links["c1"].values, links["c2"].values)), shape=(n_concepts, n_concepts))
        adjacency = ((adjacency + adjacency.T) > 0).astype(np.int8).tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        linked = np.where(degree > 0)[0]
        n_linked, n_links = len(linked), len(links)
        _, component = connected_components(adjacency[linked][:, linked], directed=False)
        largest_fraction = np.bincount(component).max() / n_linked if n_linked else 0.0
        eligible = linked[degree[linked] >= 2]
        sample = rng.choice(eligible, size=min(CLUSTERING_SAMPLE, len(eligible)), replace=False) if len(eligible) else []
        clustering = []
        for concept in sample:
            neighbours = adjacency.indices[adjacency.indptr[concept]:adjacency.indptr[concept + 1]]
            k = len(neighbours)
            clustering.append(adjacency[neighbours][:, neighbours].nnz / (k * (k - 1)))
        rows.append({"quarter": quarter, "concepts": n_linked, "links": n_links,
                     "density": 2 * n_links / (n_linked * (n_linked - 1)) if n_linked > 1 else 0,
                     "average_degree": 2 * n_links / n_linked if n_linked else 0, "degree_gini": gini(degree[linked]),
                     "largest_component_share": largest_fraction,
                     "clustering_sampled": float(np.mean(clustering)) if clustering else 0,
                     "new_links": n_new, "new_link_share": n_new / n_links if n_links else 0})
    return pd.DataFrame(rows), adjacency, degree


def log(*parts, start=[time.time()]):
    """Print a message with the time elapsed since the start of the run."""
    print(f"[{time.time() - start[0]:6.0f}s]", *parts, flush=True)


def build_field_network(papers, candidates, thesaurus, field, out_dir):
    """Construct one network of significant links and write the three output files to out_dir.

    papers      compact paper table (arxiv_id, group, first_submitted); row numbers are the paper numbers in candidates
    candidates  table with columns paper, key
    thesaurus   table with columns concept, action, replace_by
    field       field name, or "all" for all fields together
    """
    fields = [name for name in FIELD_ORDER if name in set(papers["group"])]     # fields present, in the order of FIELD_ORDER
    n_fields = len(fields)
    n_months = month_index(LAST_MONTH) + 1                                          # FIRST_MONTH .. LAST_MONTH
    window = WINDOW_MONTHS
    last_judged = n_months - 1 - window                                            # last month with a complete window
    month_of_paper = papers["first_submitted"].map(month_index).values
    field_of_paper = papers["group"].map({name: number for number, name in enumerate(fields)}).values
    in_network = np.ones(len(papers), bool) if field == "all" else field_of_paper == fields.index(field)

    # --- concepts: apply the thesaurus, then require a second abstract within the window ---
    entries = candidates[["paper", "key"]]
    entries = entries[in_network[entries["paper"].values]]
    removed = set(thesaurus.loc[thesaurus["action"] == "remove", "concept"])
    merged = dict(zip(thesaurus.loc[thesaurus["action"] == "merge", "concept"], thesaurus.loc[thesaurus["action"] == "merge", "replace_by"]))
    entries = entries[~entries["key"].isin(removed)].copy()
    entries["key"] = entries["key"].map(lambda key: merged.get(key, key))
    entries = entries.drop_duplicates(["paper", "key"])
    key_number, keys = pd.factorize(entries["key"])
    paper_of_entry = entries["paper"].values.astype(np.int64)
    key_of_entry = key_number.astype(np.int64)
    month_of_entry = month_of_paper[paper_of_entry]
    first_use = np.full(len(keys), n_months, np.int64)
    np.minimum.at(first_use, key_of_entry, month_of_entry)
    abstracts_in_first_window = np.bincount(key_of_entry[month_of_entry <= first_use[key_of_entry] + window], minlength=len(keys))
    is_concept = (abstracts_in_first_window >= 2) & (first_use <= last_judged)
    concept_of_key = np.full(len(keys), -1, np.int64)
    concept_of_key[is_concept] = np.arange(is_concept.sum())
    kept = is_concept[key_of_entry]
    paper_of_entry, concept_of_entry = paper_of_entry[kept], concept_of_key[key_of_entry[kept]]
    n_concepts = int(is_concept.sum())
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"paper": paper_of_entry, "concept_id": concept_of_entry}).to_parquet(os.path.join(out_dir, "abstract_concept.parquet"))
    pd.DataFrame({"concept_id": np.arange(n_concepts), "key": keys[np.flatnonzero(is_concept)], "first_month": first_use[is_concept],
                  "abstracts_12m": abstracts_in_first_window[is_concept]}).to_parquet(os.path.join(out_dir, "concepts.parquet"))
    log(field, "concepts", n_concepts, "concept-abstract entries", len(concept_of_entry))

    # --- expected co-occurrences: E_uv = sum over (month, field) cells of x_u x_v, with x_u = n_u / sqrt(N) ---
    cell_of_entry = month_of_paper[paper_of_entry] * n_fields + field_of_paper[paper_of_entry]
    abstracts_per_cell = np.bincount((month_of_paper * n_fields + field_of_paper)[in_network], minlength=n_months * n_fields).astype(float)
    scaled_counts = np.zeros((n_concepts, n_months * n_fields), np.float32)
    np.add.at(scaled_counts, (concept_of_entry, cell_of_entry), 1)
    scaled_counts /= np.sqrt(np.maximum(abstracts_per_cell, 1))[None, :]

    # --- observed co-occurrences per pair and month ---
    order = np.lexsort((concept_of_entry, paper_of_entry))
    paper_sorted, concept_sorted = paper_of_entry[order], concept_of_entry[order]
    starts = np.r_[0, np.flatnonzero(np.diff(paper_sorted)) + 1]
    sizes = np.r_[starts[1:], len(paper_sorted)] - starts
    parts = []
    for size in np.unique(sizes):                    # papers with the same number of concepts are processed together
        if size < 2:
            continue
        papers_of_size = np.flatnonzero(sizes == size)
        concept_rows = np.sort(concept_sorted[starts[papers_of_size][:, None] + np.arange(size)[None, :]], axis=1)
        first, second = np.triu_indices(size, 1)
        pair_month = (concept_rows[:, first] * n_concepts + concept_rows[:, second]) * n_months \
            + month_of_paper[paper_sorted[starts[papers_of_size]]][:, None]
        parts.append(pair_month.ravel())
    pair_month, abstracts = np.unique(np.concatenate(parts), return_counts=True)   # sorted by pair, then by month

    # --- per pair: first month, co-occurrences in the window, expected co-occurrences, lift ---
    pair_of_code = pair_month // n_months
    first_of_pair = np.r_[0, np.flatnonzero(pair_of_code[1:] != pair_of_code[:-1]) + 1]
    pair_keys = pair_of_code[first_of_pair]
    first_month = pair_month[first_of_pair] % n_months
    cumulative = np.r_[0, np.cumsum(abstracts)]
    window_end = np.searchsorted(pair_month, pair_keys * n_months + np.minimum(first_month + window, n_months - 1), side="right")
    n_window = cumulative[window_end] - cumulative[first_of_pair]
    judged = np.flatnonzero((n_window >= MIN_COOCCURRENCES) & (first_month <= last_judged))
    expected = np.empty(len(judged))
    for month in range(last_judged + 1):
        rows = np.flatnonzero(first_month[judged] == month)
        if len(rows) == 0:
            continue
        cells = slice(month * n_fields, (month + window + 1) * n_fields)
        u, v = pair_keys[judged[rows]] // n_concepts, pair_keys[judged[rows]] % n_concepts
        for start in range(0, len(rows), PAIRS_PER_BLOCK):
            block = slice(start, start + PAIRS_PER_BLOCK)
            expected[rows[block]] = np.einsum("ij,ij->i", scaled_counts[u[block], cells], scaled_counts[v[block], cells])
    lift = n_window[judged] / expected
    log(field, "pairs", len(pair_keys), "evaluated (at least 2 co-occurrences in the window, first seen no later than the last evaluated month)", len(judged))

    # --- false discovery rate at several lift thresholds (pairs first seen from FDR_REPORT_FROM onwards) ---
    reported = first_month[judged] >= month_index(FDR_REPORT_FROM)
    observed_reported, expected_reported = n_window[judged][reported], expected[reported]
    fdr = {}
    for threshold in FDR_LIFT_GRID:
        needed = np.maximum(MIN_COOCCURRENCES, np.ceil(threshold * expected_reported))
        n_passing = int((observed_reported / expected_reported >= threshold).sum())
        fdr[threshold] = poisson.sf(needed - 1, expected_reported).sum() / max(n_passing, 1)
        log(f"  lift >= {threshold}: links {n_passing:,}  false discovery rate {fdr[threshold]:.3f}")

    # --- significant links ---
    significant = lift >= MIN_LIFT
    links = judged[significant]
    np.savez(os.path.join(out_dir, "significant_links.npz"), u=pair_keys[links] // n_concepts, v=pair_keys[links] % n_concepts,
             keys=pair_keys[links], first_month=first_month[links], n_12m=n_window[links], expected_12m=expected[significant],
             lift=lift[significant], C=n_concepts)
    log(field, "significant links", len(links), "| false discovery rate at the lift threshold", round(fdr.get(MIN_LIFT, np.nan), 3))
    return {"field": field, "concepts": n_concepts, "significant_links": int(len(links)), "fdr": fdr}


def find(name, folder):
    """Return the first file with the given name under folder."""
    hits = sorted(glob.glob(os.path.join(folder, "**", name), recursive=True))
    if not hits:
        sys.exit(f"{name} not found under {folder}")
    return hits[0]


def main():
    global CLUSTERING_SAMPLE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="sci_concepts", help="output folder of sci_concepts.py")
    parser.add_argument("--out", default="networks")
    parser.add_argument("--fields", nargs="*", default=None, help="fields for the networks of significant links (default: all fields in the corpus)")
    parser.add_argument("--all", action="store_true", help="also construct the network of significant links for all fields together")
    parser.add_argument("--quick", action="store_true", help="clustering sample of 200 concepts (for testing)")
    args = parser.parse_args()
    if args.quick:
        CLUSTERING_SAMPLE = 200
    papers = pd.read_csv(find("papers_small.csv.gz", args.input))
    papers["month"] = papers["first_submitted"].str[:7]

    # ---- co-occurrence network ----
    concepts = pd.read_csv(find("concepts.csv", args.input))
    paper_concepts = pd.read_parquet(find("paper_concepts.parquet", args.input))
    links, links_summary, months = build_links(paper_concepts, papers)
    log(f"co-occurrence network: {len(links_summary):,} unique pairs, {int(links['count'].sum()):,} co-occurrences")
    metrics, _, _ = network_metrics(links_summary, len(concepts))
    print(metrics.to_string(index=False))
    folder = os.path.join(args.out, "cooccurrence")
    os.makedirs(folder, exist_ok=True)
    links.to_parquet(os.path.join(folder, "links_by_month_group.parquet"), index=False)
    links_summary.to_parquet(os.path.join(folder, "links_summary.parquet"), index=False)
    metrics.to_csv(os.path.join(folder, "network_metrics_per_quarter.csv"), index=False)

    # ---- networks of significant links ----
    candidates = pd.read_parquet(find("candidates.parquet", args.input), columns=["paper", "key"])
    thesaurus = pd.read_csv(os.path.join(DATA_DIR, THESAURUS))                  # read as stored (keys not lowercased)
    fields = args.fields or [field for field in FIELD_ORDER if field in set(papers["group"])]
    log(f"{len(papers):,} papers, {len(candidates):,} candidate rows; fields: {', '.join(fields)}" + ("; and all fields" if args.all else ""))
    summary = [build_field_network(papers, candidates, thesaurus, field, os.path.join(args.out, "network_fields", field)) for field in fields]
    if args.all:
        summary.append(build_field_network(papers, candidates, thesaurus, "all", os.path.join(args.out, "network_full")))
    json.dump({"cooccurrence": {"unique_pairs": int(len(links_summary)), "co_occurrences": int(links["count"].sum()),
                                "months": [months[0], months[-1]]},
               "significant_links": summary}, open(os.path.join(args.out, "network_summary.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
