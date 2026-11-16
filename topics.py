"""topics.py - topic model of the abstracts (KNSC-Rcut) for validating the field categories (step 3)

SciConcepts.

Uses a GPU if available.

Input
    papers.parquet from sci_concepts.py (--input). --fields selects a subset of the fields; default: all fields in the
    corpus.

Method
    KNSC-Rcut (Tolić, Antulov-Fantulin and Kopriva, Pattern Recognition 2018), kernel non-negative spectral
    clustering with a ratio-cut objective:
    multiplicative updates of the cluster indicators H and the prototypes F that minimise
    alpha ||Phi - Phi F H||^2 + mu ||H H' - I||^2 in kernel form (alpha 10, mu 100).

Protocol
    Sample        TOPIC_PAPERS_PER_FIELD (800) random papers per field (seed SEED)
    Features      TF-IDF of "title. abstract" (English stopwords, terms in at least 5 and at most 50% of the papers,
                  sublinear term frequency, L2 normalisation)
    Kernel        Gaussian, exp(-||x_i - x_j||^2 / sigma^2) = exp(-(2 - 2 cos_ij) / sigma^2)
    Topics        k = 4, 8, 16; for each k, sigma is selected on a grid from 0.2 to 2.0 (step 0.1) by the mean NPMI
                  coherence of the 10 top words over 8 random starts; the final model is the best of 30 starts (lowest
                  objective) with 300 updates each and no early stopping
    Assignment    each paper outside the sample is assigned to the topic t that maximises k(x, X) F_t / sqrt(F_t' K F_t),
                  the similarity to the topic prototype in the kernel space; the agreement between this rule and the
                  topics of the sample papers is reported
    Computation   GPU (PyTorch, float64) if available and if 20 updates agree with NumPy to 1e-9, otherwise CPU

Outputs (--out; default: the folder containing papers.parquet)
    topics.csv              per topic: k, sample size, NPMI, top words, fields of the sample papers in the topic
    paper_topics.parquet    arxiv_id and the topic of each paper for each k
    topics_summary.json     kernel width and coherence per k, validity of the runs, agreement of the assignment rule

Usage
    python topics.py --input sci_concepts
    python topics.py --input sci_concepts --quick      (small settings for testing)
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------- parameters
SEED = 42                               # seed for the sample and the random starts
TOPIC_PAPERS_PER_FIELD = 800            # size of the balanced sample per field
TOPIC_K = [4, 8, 16]                    # numbers of topics
TOPIC_SIGMA_GRID = [0.2, 2.0, 0.1]      # kernel width: start, stop, step
TOPIC_SCREEN_RESTARTS = 8               # random starts per kernel width in the screening
TOPIC_FINAL_RESTARTS = 30               # random starts of the final model
TOPIC_MAX_ITER = 300                    # multiplicative updates per start
TOPIC_TOLERANCE = None                  # relative decrease of the objective for early stopping (None: no early stopping)
TOPIC_TOP_WORDS = 10                    # top words per topic for the coherence
KNSC_PARAMS = {"alpha": 10.0, "mu": 100.0}
TFIDF_PARAMS = dict(stop_words="english", min_df=5, max_df=0.5, sublinear_tf=True, norm="l2")
EPS = 1e-12
PAPERS_PER_CHUNK = 5000                 # papers per chunk in the out-of-sample assignment (limits memory)


def sample_per_field(papers, n):
    """Draw a random sample of at most n papers per field (seed SEED), sorted by date."""
    samples = [field_papers.sample(n=min(n, len(field_papers)), random_state=SEED)
               for _, field_papers in papers.groupby("group")]
    return pd.concat(samples).sort_values(["first_submitted", "arxiv_id"]).reset_index(drop=True)


def log(*parts):
    """Print a message with the current time."""
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


class NumpyBackend:
    """Array operations on the CPU."""
    name = "CPU (NumPy, float64)"
    def asarray(self, array):   return np.asarray(array, dtype=np.float64)
    def to_numpy(self, array):  return np.asarray(array)
    def clamp(self, array):     return np.maximum(array, EPS)
    def relu(self, array):      return np.maximum(array, 0.0)
    def exp(self, array):       return np.exp(array)
    def eye(self, size):        return np.eye(size)
    def total(self, array):     return float(np.sum(array))
    def trace(self, array):     return float(np.trace(array))


class TorchBackend:
    """The same array operations on the GPU."""
    name = "GPU (PyTorch, float64)"
    def __init__(self):
        import torch
        self.torch, self.device = torch, torch.device("cuda")
    def asarray(self, array):   return self.torch.as_tensor(np.asarray(array), dtype=self.torch.float64, device=self.device)
    def to_numpy(self, array):  return array.detach().cpu().numpy()
    def clamp(self, array):     return self.torch.clamp_min(array, EPS)
    def relu(self, array):      return self.torch.clamp_min(array, 0.0)
    def exp(self, array):       return self.torch.exp(array)
    def eye(self, size):        return self.torch.eye(size, dtype=self.torch.float64, device=self.device)
    def total(self, array):     return float(self.torch.sum(array))
    def trace(self, array):     return float(self.torch.trace(array))


def knsc_update(state, context, backend):
    """Perform one multiplicative update of the cluster indicators H (k x n) and the prototypes F (n x k); matrix products
    are grouped to reduce the cost."""
    H, F, K, alpha, mu = state["H"], state["F"], context["K"], context["alpha"], context["mu"]
    FtK = F.T @ K
    H = H * (alpha * FtK + 2 * mu * H) / backend.clamp(alpha * ((FtK @ F) @ H) + 2 * mu * (H @ H.T) @ H)
    F = F * (K @ H.T) / backend.clamp((K @ F) @ (H @ H.T))
    return {"H": H, "F": F}


def knsc_objective(state, context, backend):
    """Return the objective alpha ||Phi - Phi F H||^2 (in kernel form) + mu ||H H' - I||^2."""
    H, F, K, alpha, mu = state["H"], state["F"], context["K"], context["alpha"], context["mu"]
    FtK, HHt = F.T @ K, H @ H.T
    deviation = HHt - backend.eye(H.shape[0])
    return float(alpha * (context["trace_K"] - 2 * backend.total(FtK * H) + backend.total((FtK @ F) * HHt))
                 + mu * backend.total(deviation * deviation))


def gaussian_kernel(cosine, sigma, backend):
    """Return the Gaussian kernel exp(-||x_i - x_j||^2 / sigma^2) for L2-normalised TF-IDF vectors
    (||x_i - x_j||^2 = 2 - 2 cos)."""
    return backend.exp(-backend.relu(2.0 - 2.0 * cosine) / sigma ** 2)


def run_knsc(K, k, seed, backend):
    """Run KNSC-Rcut once with k topics from a random start. Returns H, F, the final objective and validity checks."""
    rng = np.random.default_rng(seed)
    n_papers = K.shape[0]
    state = {"H": backend.asarray(rng.random((k, n_papers))), "F": backend.asarray(rng.random((n_papers, k)))}
    context = {"K": K, "trace_K": backend.trace(K), **KNSC_PARAMS}
    objective = [knsc_objective(state, context, backend)]
    for _ in range(TOPIC_MAX_ITER):
        state = knsc_update(state, context, backend)
        objective.append(knsc_objective(state, context, backend))
        if TOPIC_TOLERANCE is not None and objective[-2] - objective[-1] <= TOPIC_TOLERANCE * max(1.0, abs(objective[-2])):
            break
    objective = np.array(objective)
    H, F = backend.to_numpy(state["H"]), backend.to_numpy(state["F"])
    H_normalised = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), EPS)
    return {"H": H, "F": F, "objective": float(objective[-1]),
            "orthogonality_normalised": float(np.linalg.norm(H_normalised @ H_normalised.T - np.eye(k))),
            "valid": bool(np.all(np.isfinite(H)) and np.all(np.isfinite(F)) and H.min() >= 0 and F.min() >= 0
                          and np.all(np.diff(objective) <= 1e-9 * np.maximum(1.0, np.abs(objective[:-1]))))}


def top_words(X, F, vocabulary, top=TOPIC_TOP_WORDS):
    """Return the indices and words of the terms with the largest weights X' F in each topic."""
    order = np.argsort(-np.asarray(X.T @ F), axis=0)[:top]
    word_ids = [[int(word) for word in order[:, topic]] for topic in range(F.shape[1])]
    return word_ids, [[vocabulary[word] for word in topic] for topic in word_ids]


def npmi(word_ids_per_topic, binary, n_docs):
    """Return the mean normalised pointwise mutual information between the top words of each topic (co-occurrence in
    documents)."""
    scores = []
    for word_ids in word_ids_per_topic:
        joint = np.asarray((binary[:, word_ids].T @ binary[:, word_ids]).todense(), dtype=float) / n_docs
        marginal, values = np.diag(joint).copy(), []
        for first in range(len(word_ids)):
            for second in range(first + 1, len(word_ids)):
                p_joint = joint[first, second]
                values.append(-1.0 if p_joint <= 0 else 1.0 if p_joint >= 1
                              else float(np.log(p_joint / (marginal[first] * marginal[second])) / -np.log(p_joint)))
        scores.append(float(np.mean(values)))
    return scores


def choose_backend():
    """Return the GPU backend if a GPU is available and 20 updates agree with NumPy to 1e-9 (relative difference);
    otherwise return the CPU backend."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu, cpu = TorchBackend(), NumpyBackend()
            rng = np.random.default_rng(0)
            vectors = rng.random((300, 50))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            traces = []
            for backend in (cpu, gpu):
                K = gaussian_kernel(backend.asarray(vectors @ vectors.T), 0.8, backend)
                context = {"K": K, "trace_K": backend.trace(K), **KNSC_PARAMS}
                state = {"H": backend.asarray(np.random.default_rng(1).random((6, 300))),
                         "F": backend.asarray(np.random.default_rng(2).random((300, 6)))}
                trace = []
                for _ in range(20):
                    state = knsc_update(state, context, backend)
                    trace.append(knsc_objective(state, context, backend))
                traces.append(np.array(trace))
            difference = float(np.max(np.abs(traces[0] - traces[1]) / np.abs(traces[0])))
            log(f"GPU self-test: max relative difference to NumPy = {difference:.1e}")
            if difference < 1e-9:
                return gpu
    except Exception as error:
        log("GPU not used:", type(error).__name__, str(error)[:80])
    return NumpyBackend()


def fit_topics(papers):
    """Fit the topic models for each k on the balanced sample.

    Returns the vectoriser, the TF-IDF matrix of the sample, the models {k: (sigma, F, K, assignment)}, a summary per k
    and a table of topics (top words, NPMI, fields of the member papers)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    backend = choose_backend()
    log("Topic model backend:", backend.name)
    sample = sample_per_field(papers, TOPIC_PAPERS_PER_FIELD)
    texts = (sample["title"] + ". " + sample["abstract"]).tolist()
    vectoriser = TfidfVectorizer(**TFIDF_PARAMS)
    X = vectoriser.fit_transform(texts)
    vocabulary = vectoriser.get_feature_names_out()
    binary = (X > 0).astype(np.int32).tocsc()
    cosine = backend.asarray((X @ X.T).toarray())
    n_papers = X.shape[0]
    sigma_first, sigma_last, sigma_step = TOPIC_SIGMA_GRID
    sigmas = np.round(np.arange(sigma_first, sigma_last + sigma_step / 2, sigma_step), 10)
    log(f"Topics: {n_papers} sample papers, {len(vocabulary)} terms, sigma grid {sigmas[0]}..{sigmas[-1]} ({len(sigmas)} values)")
    summary, rows, models = {}, [], {}
    for k in TOPIC_K:
        started = time.time()
        screen = []                                  # (sigma, mean NPMI over the screening starts)
        for sigma in sigmas:
            K = gaussian_kernel(cosine, float(sigma), backend)
            coherence = [np.mean(npmi(top_words(X, run_knsc(K, k, SEED + restart, backend)["F"], vocabulary)[0], binary, n_papers))
                         for restart in range(TOPIC_SCREEN_RESTARTS)]
            screen.append((float(sigma), float(np.mean(coherence))))
        sigma = max(screen, key=lambda item: item[1])[0]
        K = gaussian_kernel(cosine, sigma, backend)
        runs = [run_knsc(K, k, SEED + 1000 + restart, backend) for restart in range(TOPIC_FINAL_RESTARTS)]
        best = min(runs, key=lambda run: run["objective"])
        assignment = best["H"].argmax(axis=0)
        word_ids, words = top_words(X, best["F"], vocabulary)
        coherence = npmi(word_ids, binary, n_papers)
        for topic in range(k):
            members = np.where(assignment == topic)[0]
            rows.append({"k": k, "topic": topic, "sample_size": int(len(members)), "npmi": round(coherence[topic], 4),
                         "top_words": " ".join(words[topic]), "groups": sample.iloc[members]["group"].value_counts().to_dict()})
        summary[f"k{k}"] = {"sigma": sigma, "sigma_screen_npmi": {f"{value:.2f}": round(score, 4) for value, score in screen},
                            "mean_npmi": round(float(np.mean(coherence)), 4),
                            "orthogonality_normalised": round(best["orthogonality_normalised"], 4),
                            "all_final_runs_valid": bool(all(run["valid"] for run in runs)),
                            "minutes": round((time.time() - started) / 60, 1)}
        models[k] = (sigma, best["F"], backend.to_numpy(K), assignment)
        log(f"k={k:2d} sigma={sigma:.2f} mean NPMI={summary[f'k{k}']['mean_npmi']:.3f} "
            f"orthogonality={best['orthogonality_normalised']:.3f} valid={summary[f'k{k}']['all_final_runs_valid']} ({summary[f'k{k}']['minutes']} min)")
    return vectoriser, X, models, summary, pd.DataFrame(rows)


def assign_topics(papers, vectoriser, X_sample, models):
    """Assign each paper to the topic t that maximises k(x, X) F_t / sqrt(F_t' K F_t), the similarity to the topic
    prototype in the kernel space. Also returns the agreement between this rule and the topics of the sample papers."""
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    assigned = {k: np.empty(len(texts), dtype=np.int16) for k in models}
    norms = {k: np.sqrt(np.maximum(np.einsum("it,ij,jt->t", F, K, F), EPS)) for k, (sigma, F, K, _) in models.items()}
    for start in range(0, len(texts), PAPERS_PER_CHUNK):
        X_chunk = vectoriser.transform(texts[start:start + PAPERS_PER_CHUNK])
        cosine = (X_chunk @ X_sample.T).toarray()
        for k, (sigma, F, K, _) in models.items():
            K_chunk = np.exp(-np.maximum(2.0 - 2.0 * cosine, 0.0) / sigma ** 2)
            assigned[k][start:start + PAPERS_PER_CHUNK] = np.argmax((K_chunk @ F) / norms[k], axis=1)
    agreement = {}
    for k, (sigma, F, K, assignment) in models.items():
        agreement[f"k{k}"] = round(float(np.mean(np.argmax((K @ F) / norms[k], axis=1) == assignment)), 3)
    return assigned, agreement


def main():
    global TOPIC_PAPERS_PER_FIELD, TOPIC_SIGMA_GRID, TOPIC_SCREEN_RESTARTS, TOPIC_FINAL_RESTARTS, TOPIC_MAX_ITER, TFIDF_PARAMS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="sci_concepts", help="output folder of sci_concepts.py")
    parser.add_argument("--fields", nargs="*", default=None, help="fields to include (default: all fields in the corpus)")
    parser.add_argument("--out", default=None, help="output folder (default: the folder containing papers.parquet)")
    parser.add_argument("--quick", action="store_true", help="small settings that exercise every step (for testing)")
    args = parser.parse_args()
    if args.quick:
        TOPIC_PAPERS_PER_FIELD, TOPIC_SIGMA_GRID = 40, [0.6, 1.0, 0.4]
        TOPIC_SCREEN_RESTARTS, TOPIC_FINAL_RESTARTS, TOPIC_MAX_ITER = 1, 2, 30
        TFIDF_PARAMS = dict(TFIDF_PARAMS, min_df=2)
    papers_file = sorted(glob.glob(os.path.join(args.input, "**", "papers.parquet"), recursive=True))[0]
    out = args.out or os.path.dirname(papers_file)
    papers = pd.read_parquet(papers_file)
    if args.fields:
        papers = papers[papers["group"].isin(args.fields)].reset_index(drop=True)
    log(f"{len(papers):,} papers from {papers_file}")
    vectoriser, X_sample, models, summary, topics = fit_topics(papers)
    assigned, agreement = assign_topics(papers, vectoriser, X_sample, models)
    log("agreement between the out-of-sample rule and the topics of the sample papers:", agreement)
    topics.to_csv(os.path.join(out, "topics.csv"), index=False)
    pd.DataFrame({"arxiv_id": papers["arxiv_id"], **{f"topic_k{k}": topic_numbers for k, topic_numbers in assigned.items()}}).to_parquet(
        os.path.join(out, "paper_topics.parquet"), index=False)
    json.dump({"results": summary, "out_of_sample_agreement": agreement}, open(os.path.join(out, "topics_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
