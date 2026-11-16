"""sci_concepts.py - arXiv corpus, candidate concepts and concept list (step 1)

SciConcepts.

Runs on a CPU; MultipartiteRank takes about 7 hours for 186,112 papers on a Kaggle CPU.

Corpus
    The corpus consists of the papers in the arXiv metadata snapshot (arxiv-metadata-oai-snapshot.json, provided by
    Cornell University as the Kaggle dataset Cornell-University/arxiv) whose primary category (the first category
    listed) belongs to a field in FIELDS and whose first version was submitted between FIRST_DATE and LAST_DATE
    (submission date of the first version, or the update date if the first version has no date). Withdrawn papers
    (withdrawal stated in the comments or in the first 300 characters of the abstract), duplicate records and abstracts
    shorter than MIN_ABSTRACT_WORDS words are removed. Papers are sorted by
    date of first submission and arXiv identifier; this order defines the paper index in all output files.

Candidate concepts
    MultipartiteRank (Boudin, NAACL 2018), as implemented in pke by its author, is applied to the lowercased text
    "title. abstract". Lowercasing is needed because spaCy tags capitalised words (for example in titles or in "LLMs")
    as proper nouns and does not lemmatise them. Candidates are noun phrases consisting of adjectives and nouns and
    ending in a noun (NOUN_PHRASE_GRAMMAR), filtered with the pke defaults (stopwords, at most 5 words) and identified by
    their spaCy lemmas (the candidate key). Candidate weights are computed with alpha = 1.1 (position adjustment;
    Boudin 2018), a clustering threshold of 0.74 and average linkage (pke defaults). All candidates are kept with
    their weights.
    Papers are processed in parallel and saved in checkpoint files of CHUNK_SIZE papers, so that an interrupted run
    resumes at the first missing file; checkpoint files from an earlier run (--checkpoints) are reused if they contain
    the same papers. A batch exceeding BATCH_TIMEOUT_S is repeated paper by paper; a paper exceeding PAPER_TIMEOUT_S
    receives no candidates and is listed in the manifest. These settings do not change the candidates.

Concepts
    A hand-made thesaurus (data/thesaurus_v1.csv) removes generic words referring to the paper or the research process
    (120 words, for example paper, result, approach) and merges variants such as an acronym and its full form (27
    pairs); keys are compared in lower case without surrounding spaces. A candidate key is defined as a concept if it
    appears in at least MIN_ABSTRACTS (5) abstracts in the whole corpus; merged variants count once per paper. The
    label of a concept is its most frequent surface form. The threshold depends on the whole corpus, so the concept
    list is recomputed when papers are added.

Modes (--mode)
    full      all fields in FIELDS (--fields selects a subset)
    extend    adds papers to an earlier output (--earlier): only the papers in the new fields (--new-fields; default:
              the fields absent from the earlier output) are processed with MultipartiteRank, and the concept list
              is recomputed for all papers
    counts    papers per category and month only
    --quick N selects N papers per field and lowers the concept threshold to 2 abstracts (for testing).

Outputs (--out; default: sci_concepts/, and a zip file of this folder without the checkpoint files)
    papers.parquet                 arxiv_id, group (field), category, categories, first_submitted, month, title, abstract
    papers_small.csv.gz            the same papers without title and abstract, with the abstract length in words
    candidates.parquet             paper (row in papers), key, surface (first surface form in the text), score (weight)
    candidates_new.parquet         extend mode: candidates of the new papers, by arXiv identifier
    concepts.csv                   concept_id, key, label, n_docs (number of abstracts), first_month, docs_<field>
                                   (number of abstracts per field)
    paper_concepts.parquet         paper, concept_id, score (highest first within each paper)
    counts_by_category_month.csv, manifest.json

Usage
    python sci_concepts.py --mode full --snapshot arxiv-metadata-oai-snapshot.json
    python sci_concepts.py --mode extend --earlier <earlier output> --snapshot arxiv-metadata-oai-snapshot.json
    python sci_concepts.py --mode full --snapshot arxiv-metadata-oai-snapshot.json --checkpoints <earlier run>/candidates
    python sci_concepts.py --mode full --snapshot arxiv-metadata-oai-snapshot.json --quick 100
"""
import argparse
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from email.utils import parsedate_to_datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------- parameters
# Corpus: the field of a paper is its primary arXiv category (the first category listed)
FIELDS = {
    "artificial_intelligence":     ["cs.AI"],
    "machine_learning":            ["cs.LG", "stat.ML"],
    "social_information_networks": ["cs.SI"],
    "economics":                   ["econ.EM", "econ.GN", "econ.TH"],
    "quantitative_finance":        ["q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM",
                                    "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR"],
    "physics_society":             ["physics.soc-ph"],
}
FIRST_DATE = "2021-01-01"     # earliest date of first submission
LAST_DATE = "2026-10-31"      # latest date of first submission
MIN_ABSTRACT_WORDS = 80       # minimum abstract length in words
SEED = 42                     # seed for the test sample (--quick)

# Candidate concepts: MultipartiteRank (Boudin, NAACL 2018), implemented in pke
NOUN_PHRASE_GRAMMAR = "NP: {<NOUN|PROPN|ADJ>*<NOUN|PROPN>}"   # adjectives and nouns ending in a noun
MPR_ALPHA = 1.1               # position adjustment of candidate weights (value from Boudin 2018)
MPR_THRESHOLD = 0.74          # threshold for clustering candidates into topics (pke default)
MPR_LINKAGE = "average"       # linkage for clustering candidates into topics (pke default)
CHUNK_SIZE = 2000             # papers per checkpoint file
BATCH_SIZE = 100              # papers per task of a worker process
BATCH_TIMEOUT_S = 600         # time limit per batch; a batch exceeding it is repeated paper by paper
PAPER_TIMEOUT_S = 60          # time limit per paper; a paper exceeding it receives no candidates and is reported
TASKS_PER_WORKER = 50         # batches before a worker process is restarted (limits memory growth)

# Concepts
THESAURUS = "thesaurus_v1.csv"   # in data/: removes 120 generic words, merges 27 variants (acronyms)
MIN_ABSTRACTS = 5                # minimum number of abstracts per concept
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


WITHDRAWN = re.compile(r"(paper|article|manuscript|submission|version)\s+(has been|is|was)\s+withdrawn"
                       r"|withdrawn by the authors?|^withdrawn\b", re.I)


def normalise_whitespace(text):
    """Replace runs of whitespace with single spaces and remove leading and trailing spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def read_papers(snapshot_path, fields=None, first_date=FIRST_DATE, last_date=LAST_DATE):
    """Read the papers in the given fields (default: all fields in FIELDS) from the snapshot. Returns the papers,
    sorted by date of first submission and arXiv identifier, and the numbers of records read, included and excluded
    (by reason)."""
    fields = fields or list(FIELDS)
    field_of_category = {category: field for field, categories in FIELDS.items() if field in fields
                         for category in categories}
    rows, seen_ids, counts = [], set(), Counter()
    with open(snapshot_path) as snapshot:
        for line in snapshot:
            counts["snapshot"] += 1
            record = json.loads(line)
            categories = (record.get("categories") or "").split()
            if not categories or categories[0] not in field_of_category:
                continue
            try:
                first_submitted = parsedate_to_datetime(record["versions"][0]["created"]).date().isoformat()
            except Exception:
                first_submitted = record.get("update_date") or ""
            if first_submitted < first_date:
                counts["before_first_date"] += 1
                continue
            if first_submitted > last_date:
                counts["after_last_date"] += 1
                continue
            if record["id"] in seen_ids:
                counts["duplicate"] += 1
                continue
            seen_ids.add(record["id"])
            title, abstract = normalise_whitespace(record.get("title")), normalise_whitespace(record.get("abstract"))
            comments = normalise_whitespace(record.get("comments"))
            if WITHDRAWN.search(comments) or WITHDRAWN.search(abstract[:300]):
                counts["withdrawn"] += 1
                continue
            if len(abstract.split()) < MIN_ABSTRACT_WORDS:
                counts["too_short"] += 1
                continue
            rows.append({"arxiv_id": record["id"], "group": field_of_category[categories[0]], "category": categories[0],
                         "categories": " ".join(categories), "first_submitted": first_submitted,
                         "month": first_submitted[:7], "title": title, "abstract": abstract})
            counts["kept"] += 1
    papers = pd.DataFrame(rows).sort_values(["first_submitted", "arxiv_id"]).reset_index(drop=True)
    return papers, dict(counts)


def sample_per_field(papers, n):
    """Draw a random sample of at most n papers per field (seed SEED), sorted by date."""
    samples = [field_papers.sample(n=min(n, len(field_papers)), random_state=SEED)
               for _, field_papers in papers.groupby("group")]
    return pd.concat(samples).sort_values(["first_submitted", "arxiv_id"]).reset_index(drop=True)


def papers_small(papers):
    """Return the paper table without titles and abstracts, with the abstract length in words."""
    return papers[["arxiv_id", "group", "category", "first_submitted"]].assign(words=papers["abstract"].str.split().str.len())


FAILED_PAPERS = []            # arXiv identifiers of papers that received no candidates (time limit exceeded)
_SPACY_MODEL = None           # spaCy model, loaded once in each worker process


def log(*parts):
    """Print a message with the current time."""
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def ensure_dependencies():
    """Install pke (from its GitHub repository) and the spaCy model en_core_web_sm if they are missing."""
    try:
        import pke  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/boudinfl/pke.git@69871ffdb720b83df23684fea53ec8776fd87e63"], check=True)
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)


def _load_spacy_in_worker():
    """Load the spaCy model once per worker process."""
    global _SPACY_MODEL
    import spacy
    _SPACY_MODEL = spacy.load("en_core_web_sm")


def candidates_for_text(text):
    """Extract all MultipartiteRank candidates from a text as (lemma key, surface form, weight), by decreasing weight."""
    import pke
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text.lower(), language="en", spacy_model=_SPACY_MODEL, normalization=None)
    for sentence in extractor.sentences:
        sentence.stems = [lemma.lower() for lemma in sentence.meta["lemmas"]]    # candidates are identified by lemmas
    extractor.grammar_selection(grammar=NOUN_PHRASE_GRAMMAR)
    extractor.candidate_filtering()
    if not extractor.candidates:
        return []
    extractor.candidate_weighting(alpha=MPR_ALPHA, threshold=MPR_THRESHOLD, method=MPR_LINKAGE)
    return [(key, " ".join(extractor.candidates[key].surface_forms[0]).lower(), round(float(weight), 6))
            for key, weight in sorted(extractor.weights.items(), key=lambda item: -item[1])]


def _candidates_for_batch(texts):
    """Extract the candidates for a batch of texts."""
    return [candidates_for_text(text) for text in texts]


def _new_pool(workers):
    """Start a pool of worker processes, each with the spaCy model loaded."""
    return Pool(workers, initializer=_load_spacy_in_worker, maxtasksperchild=TASKS_PER_WORKER)


def _paper_by_paper(pool, workers, texts, ids):
    """Repeat an unfinished batch one paper at a time, each paper with the time limit PAPER_TIMEOUT_S."""
    results = []
    for arxiv_id, text in zip(ids, texts):
        try:
            results.append(pool.apply_async(candidates_for_text, (text,)).get(timeout=PAPER_TIMEOUT_S))
        except Exception as error:                       # time limit exceeded, worker failure or pke error
            pool.terminate()
            pool = _new_pool(workers)
            FAILED_PAPERS.append(arxiv_id)
            results.append([])
            log(f"    {arxiv_id}: no candidates ({type(error).__name__})")
    return results, pool


def _available_memory_gb():
    """Return the available memory in GB (NaN if /proc/meminfo cannot be read)."""
    try:
        with open("/proc/meminfo") as meminfo:
            return int(next(line for line in meminfo if line.startswith("MemAvailable")).split()[1]) / 1e6
    except Exception:
        return float("nan")


def restore_checkpoints(checkpoint_dir, earlier_dir):
    """Copy the checkpoint files of an earlier run from earlier_dir to checkpoint_dir."""
    hits = sorted(glob.glob(os.path.join(earlier_dir, "chunk_*.jsonl.gz")))
    copied = 0
    for source in hits:
        target = os.path.join(checkpoint_dir, os.path.basename(source))
        if not os.path.exists(target):
            shutil.copy(source, target)
            copied += 1
    if hits:
        log(f"{copied} checkpoint files copied from {os.path.dirname(hits[0])}")


def extract_candidates(papers, checkpoint_dir, earlier_checkpoints=None, workers=None):
    """Extract the candidates for all papers. Returns a table with the columns paper (row in `papers`), key, surface
    and score."""
    workers = workers or os.cpu_count() or 1
    os.makedirs(checkpoint_dir, exist_ok=True)
    if earlier_checkpoints:
        restore_checkpoints(checkpoint_dir, earlier_checkpoints)
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    ids = papers["arxiv_id"].tolist()
    chunks = [(number, slice(start, start + CHUNK_SIZE))
              for number, start in enumerate(range(0, len(texts), CHUNK_SIZE))]
    checkpoint_path = lambda number: os.path.join(checkpoint_dir, f"chunk_{number:05d}.jsonl.gz")
    pending = []
    for number, rows in chunks:
        if os.path.exists(checkpoint_path(number)):
            with gzip.open(checkpoint_path(number), "rt") as checkpoint:
                if json.loads(checkpoint.readline())["arxiv_id"] == ids[rows.start]:
                    continue                                  # chunk completed earlier for the same papers
        pending.append((number, rows))
    log(f"MultipartiteRank: {len(texts)} papers, {len(chunks)} chunks ({len(chunks) - len(pending)} completed), {workers} workers")
    started = time.time()
    pool = _new_pool(workers)
    try:
        for completed, (number, rows) in enumerate(pending, 1):
            chunk_texts, chunk_ids = texts[rows], ids[rows]
            batch_starts = range(0, len(chunk_texts), BATCH_SIZE)
            batches = pool.imap(_candidates_for_batch, [chunk_texts[batch_start:batch_start + BATCH_SIZE] for batch_start in batch_starts])
            results = []
            for start in batch_starts:
                try:
                    results += batches.next(timeout=BATCH_TIMEOUT_S)
                except Exception as error:                # time limit exceeded, worker failure or pke error
                    log(f"  chunk {number + 1}: batch {start // BATCH_SIZE + 1} did not finish "
                        f"({type(error).__name__}); repeating it paper by paper")
                    pool.terminate()
                    pool = _new_pool(workers)
                    end = start + BATCH_SIZE
                    repeated, pool = _paper_by_paper(pool, workers, chunk_texts[start:end], chunk_ids[start:end])
                    results += repeated
                    batches = pool.imap(_candidates_for_batch, [chunk_texts[batch_start:batch_start + BATCH_SIZE]
                                                                for batch_start in range(end, len(chunk_texts), BATCH_SIZE)])
            with gzip.open(checkpoint_path(number) + ".tmp", "wt") as checkpoint:
                for arxiv_id, candidates in zip(chunk_ids, results):
                    checkpoint.write(json.dumps({"arxiv_id": arxiv_id, "candidates": candidates}) + "\n")
            os.replace(checkpoint_path(number) + ".tmp", checkpoint_path(number))
            seconds_per_paper = (time.time() - started) / min(completed * CHUNK_SIZE, len(texts))
            remaining = max(len(texts) - completed * CHUNK_SIZE, 0)
            log(f"  chunk {number + 1}/{len(chunks)} saved; {seconds_per_paper * 1000:.0f} ms per paper; "
                f"about {seconds_per_paper * remaining / 60:.0f} min remaining; {_available_memory_gb():.1f} GB memory available")
    finally:
        pool.terminate()
    row_of_id = {arxiv_id: row for row, arxiv_id in enumerate(ids)}
    table = []
    for number, _ in chunks:
        with gzip.open(checkpoint_path(number), "rt") as checkpoint:
            for line in checkpoint:
                record = json.loads(line)
                row = row_of_id[record["arxiv_id"]]
                table += [(row, key, surface, weight) for key, surface, weight in record["candidates"]]
    return pd.DataFrame(table, columns=["paper", "key", "surface", "score"])


def read_thesaurus(path):
    """Read the thesaurus as a mapping {candidate key: replacement key}, with "" for removed keys. Input columns: concept
    (lemma key), action (remove or merge), replace_by; rows without an action are ignored."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = table.apply(lambda column: column.str.strip().str.lower())
    table = table[table["action"].isin(["remove", "merge"])]
    if ((table["action"] == "merge") & (table["replace_by"] == "")).any():
        raise ValueError(f"{path}: every merge needs a replacement in replace_by")
    replacement = np.where(table["action"] == "remove", "", table["replace_by"])
    return dict(zip(table["concept"], replacement))


def build_concepts(candidates, papers, thesaurus):
    """Define the concepts and assign them to the papers.

    candidates: columns paper (row in `papers`), key, surface, score; papers: columns month and group (field).
    Returns the concept table, the paper-concept table and counts of candidate keys, concepts and thesaurus entries."""
    candidates = candidates.assign(key=candidates["key"].map(lambda key: thesaurus.get(key, key)))
    candidates = (candidates[candidates["key"] != ""].sort_values("score", ascending=False)
                  .drop_duplicates(["paper", "key"]))                          # merged variants count once per paper
    abstracts_per_key = candidates.groupby("key")["paper"].nunique()
    concept_keys = abstracts_per_key[abstracts_per_key >= MIN_ABSTRACTS].index
    occurrences = candidates[candidates["key"].isin(concept_keys)].copy()
    label = (occurrences.groupby(["key", "surface"]).size().reset_index(name="n")
             .sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key").set_index("key")["surface"])
    concepts = pd.DataFrame({"key": sorted(concept_keys)})
    concepts["concept_id"] = np.arange(len(concepts))
    concepts["label"] = concepts["key"].map(label)
    concepts["n_docs"] = concepts["key"].map(abstracts_per_key)
    occurrences = occurrences.merge(concepts[["key", "concept_id"]], on="key")
    occurrences["month"] = papers["month"].values[occurrences["paper"].values]
    occurrences["group"] = papers["group"].values[occurrences["paper"].values]
    concepts["first_month"] = concepts["concept_id"].map(occurrences.groupby("concept_id")["month"].min())
    abstracts_per_field = occurrences.groupby(["concept_id", "group"])["paper"].nunique().unstack(fill_value=0)
    for field in FIELDS:
        if field in abstracts_per_field.columns:
            concepts[f"docs_{field}"] = concepts["concept_id"].map(abstracts_per_field[field]).fillna(0).astype(int)
        else:
            concepts[f"docs_{field}"] = 0
    paper_concepts = occurrences[["paper", "concept_id", "score"]].sort_values(["paper", "score"], ascending=[True, False])
    counts = {"candidate_keys": int(abstracts_per_key.size), "concepts": int(len(concepts)),
              "excluded_rare": int((abstracts_per_key < MIN_ABSTRACTS).sum()),
              "thesaurus_removed": sum(value == "" for value in thesaurus.values()),
              "thesaurus_merged": sum(value != "" for value in thesaurus.values())}
    return concepts, paper_concepts, counts


def find_earlier(name, folder):
    """Return the path to a file from the earlier output, either under folder or inside a zip file under folder."""
    hits = sorted(glob.glob(os.path.join(folder, "**", name), recursive=True))
    if hits:
        return hits[0]
    for archive in glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True):
        with zipfile.ZipFile(archive) as zipped:
            member = next((member for member in zipped.namelist() if member.endswith("/" + name) or member == name), None)
            if member:
                target = os.path.join(os.path.dirname(os.path.abspath(folder)), "earlier_extracted")
                zipped.extract(member, target)
                return os.path.join(target, member)
    sys.exit(f"{name} from the earlier output was not found under {folder}")


def main():
    global MIN_ABSTRACTS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["full", "extend", "counts"], default="full")
    parser.add_argument("--snapshot", required=True, help="path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--earlier", default=None, help="folder with the earlier output (required in extend mode)")
    parser.add_argument("--checkpoints", default=None, help="folder with checkpoint files of an interrupted run")
    parser.add_argument("--fields", nargs="*", default=None, help="fields in a full run (default: all fields in FIELDS)")
    parser.add_argument("--new-fields", nargs="*", default=None, help="fields added in extend mode")
    parser.add_argument("--quick", type=int, default=0, help="papers per field in a test run")
    parser.add_argument("--out", default=None, help="output folder (default: sci_concepts, or sci_concepts_extend in extend mode)")
    args = parser.parse_args()
    if args.mode == "extend" and not args.earlier:
        parser.error("--earlier is required in extend mode")
    out = os.path.abspath(args.out or "sci_concepts" + ("_extend" if args.mode == "extend" else ""))
    os.makedirs(out, exist_ok=True)
    if args.quick:                                   # small thresholds that exercise every step
        MIN_ABSTRACTS = 2
    started = time.time()
    snapshot = args.snapshot
    log(f"mode {args.mode}; snapshot {snapshot} ({os.path.getsize(snapshot) / 1e9:.2f} GB)")
    manifest = {"mode": args.mode, "snapshot": snapshot, "snapshot_bytes": os.path.getsize(snapshot),
                "snapshot_modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(snapshot))),
                "fields": FIELDS, "first_date": FIRST_DATE, "last_date": LAST_DATE,
                "min_abstract_words": MIN_ABSTRACT_WORDS}

    # ---- papers and candidates ----
    if args.mode == "extend":
        earlier_papers = pd.read_parquet(find_earlier("papers.parquet", args.earlier),
                                         columns=["arxiv_id", "group", "category", "categories", "first_submitted", "month", "title", "abstract"])
        earlier_candidates = pd.read_parquet(find_earlier("candidates.parquet", args.earlier))
        new_fields = args.new_fields or [field for field in FIELDS if field not in set(earlier_papers["group"])]
        log(f"earlier output: {len(earlier_papers):,} papers, {len(earlier_candidates):,} candidate rows; new fields: {new_fields}")
        new_papers, read_counts = read_papers(snapshot, fields=new_fields)
        new_papers = new_papers[~new_papers["arxiv_id"].isin(earlier_papers["arxiv_id"])].reset_index(drop=True)
        log("new papers:", read_counts, new_papers["group"].value_counts().to_dict())
        new_papers.groupby(["group", "category", "month"]).size().rename("papers").reset_index() \
                  .to_csv(os.path.join(out, "counts_by_category_month_new.csv"), index=False)
        ensure_dependencies()
        new_candidates = extract_candidates(new_papers, os.path.join(out, "candidates_new"), args.checkpoints)
        new_candidates.assign(arxiv_id=new_papers["arxiv_id"].values[new_candidates["paper"].values]).drop(columns="paper") \
                      .to_parquet(os.path.join(out, "candidates_new.parquet"), index=False)
        papers = pd.concat([earlier_papers, new_papers], ignore_index=True).sort_values(["first_submitted", "arxiv_id"]).reset_index(drop=True)
        row_of_id = pd.Series(np.arange(len(papers)), index=papers["arxiv_id"])           # paper numbers by arXiv ID
        earlier_candidates["paper"] = row_of_id.loc[earlier_papers["arxiv_id"].values[earlier_candidates["paper"].values]].values
        new_candidates["paper"] = row_of_id.loc[new_papers["arxiv_id"].values[new_candidates["paper"].values]].values
        candidates = pd.concat([earlier_candidates, new_candidates], ignore_index=True)
        del earlier_candidates, new_candidates
        manifest.update({"new_fields": new_fields, "read_counts_new": read_counts})
    else:
        papers, read_counts = read_papers(snapshot, fields=args.fields)
        log("read:", read_counts)
        papers.groupby(["group", "category", "month"]).size().rename("papers").reset_index() \
              .to_csv(os.path.join(out, "counts_by_category_month.csv"), index=False)
        print(papers.assign(year=papers["month"].str[:4]).pivot_table(index="category", columns="year", values="arxiv_id",
                                                                       aggfunc="count", fill_value=0).to_string())
        manifest["read_counts"] = read_counts
        if args.mode == "counts":
            json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2, default=str)
            return
        if args.quick:
            papers = sample_per_field(papers, args.quick)
        ensure_dependencies()
        candidates = extract_candidates(papers, os.path.join(out, "candidates"), args.checkpoints)
    log(f"{len(papers):,} papers in the corpus:", papers["group"].value_counts().to_dict())

    # ---- concepts ----
    thesaurus_file = os.path.join(DATA_DIR, THESAURUS)
    concepts, paper_concepts, concept_counts = build_concepts(candidates, papers, read_thesaurus(thesaurus_file))
    log("concepts:", concept_counts)

    # ---- outputs ----
    papers.to_parquet(os.path.join(out, "papers.parquet"), index=False)
    papers_small(papers).to_csv(os.path.join(out, "papers_small.csv.gz"), index=False)
    candidates.to_parquet(os.path.join(out, "candidates.parquet"), index=False)
    concepts.to_csv(os.path.join(out, "concepts.csv"), index=False)
    paper_concepts.to_parquet(os.path.join(out, "paper_concepts.parquet"), index=False)
    manifest.update({"corpus_papers": int(len(papers)), "papers_per_field": papers["group"].value_counts().to_dict(),
                     "concepts": concept_counts, "thesaurus": thesaurus_file, "extraction_failed": FAILED_PAPERS,
                     "multipartiterank": {"grammar": NOUN_PHRASE_GRAMMAR, "alpha": MPR_ALPHA,
                                          "threshold": MPR_THRESHOLD, "linkage": MPR_LINKAGE,
                                          "normalisation": "lowercased text, spaCy lemmas"},
                     "min_abstracts": MIN_ABSTRACTS, "minutes": round((time.time() - started) / 60, 1)})
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2, default=str)
    with zipfile.ZipFile(out + ".zip", "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for root, _, files in os.walk(out):
            if os.path.basename(root).startswith("candidates"):
                continue                                  # checkpoint files; candidates.parquet contains the same data
            for name in files:
                zipped.write(os.path.join(root, name), os.path.relpath(os.path.join(root, name), os.path.dirname(out)))
    log(f"done in {manifest['minutes']} min: {out}.zip ({os.path.getsize(out + '.zip') / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
