"""Paired Neutral vs Conditioned F1 against a fixed perspective proxy label.

Offline evaluation only. Historical Global vs Perspective evaluation remains in
eval.py and evaluation_results_semiaggregated.* (never overwritten here).
See perspectives_analysis.md for the protocol and interpretation limits.
"""

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import bootstrap as scipy_bootstrap, permutation_test
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
MODELS = {
    "Qwen_Qwen3.5-9B": "Qwen3.5-9B",
    "mistralai_Ministral-3-8B-Instruct-2512": "Ministral3-8B",
    "google_gemma-4-E4B-it": "Gemma4-E4B",
}
STRATEGIES = {
    "gender_female": ("aggregation_on_female", "Female", "traits"),
    "gender_male": ("aggregation_on_male", "Male", "traits"),
    "generation_gen_x": ("aggregation_on_genx", "Gen X", "traits"),
    "generation_gen_y": ("aggregation_on_geny", "Gen Y", "traits"),
    "generation_gen_z": ("aggregation_on_genz", "Gen Z", "traits"),
    "generation_boomer": ("aggregation_on_boomer", "Boomer", "traits"),
    "relationship_colleagues": ("aggregation_on_colleague", "Colleagues", "relationships"),
    "relationship_friends": ("aggregation_on_best_friend", "Best Friends", "relationships"),
}
SUFFIX = "_meta-llama_Llama-3.1-8B-Instruct_n_turn_2"


def relative_or_absolute(path, root):
    path = path.resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def stable_seed(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def f1_from_counts(counts):
    """Vectorized resampling kernel, equivalent to sklearn F1(positive=1).

    sklearn.f1_score has no axis argument for batches of confusion counts.
    Keep this small NumPy kernel for resampling; reported point scores use
    sklearn directly. Equivalence is checked on weighted confusion counts.
    """
    tp, fp, fn = np.moveaxis(np.asarray(counts), -1, 0)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                     where=denominator != 0)


def positive_class_f1(references, predictions):
    """Public metric implementation, including explicit zero_division=0."""
    y, p = np.asarray(references), np.asarray(predictions)
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("F1 requires binary labels and predictions")
    return float(f1_score(y.astype(int), p.astype(int), pos_label=1,
                          average="binary", zero_division=0))


def grouped_counts(y, predictions, group_codes, n_groups):
    indicators = np.column_stack(((y == 1) & (predictions == 1),
                                  (y == 0) & (predictions == 1),
                                  (y == 1) & (predictions == 0)))
    counts = np.zeros((n_groups, 3), dtype=np.int64)
    np.add.at(counts, group_codes, indicators)
    return counts


def cluster_f1_statistic(a, b):
    """Create the shared pooled-F1 statistic for SciPy resampling functions."""
    counts = np.concatenate((a, b))

    def statistic(neutral_ids, conditioned_ids, axis=-1):
        neutral_ids = np.moveaxis(neutral_ids, axis, -1)
        conditioned_ids = np.moveaxis(conditioned_ids, axis, -1)
        return (f1_from_counts(counts[conditioned_ids].sum(axis=-2))
                - f1_from_counts(counts[neutral_ids].sum(axis=-2)))

    return statistic


def paired_bootstrap_f1(a, b, n_resamples, seed):
    """SciPy percentile CI, resampling paired whole-cluster indices."""
    n_groups = len(a)
    if n_groups == 1:
        # All possible resamples contain the same sole cluster. SciPy requires
        # two observations, so return the degenerate interval analytically.
        delta = float(f1_from_counts(b[0]) - f1_from_counts(a[0]))
        return dict(delta_ci95_low=delta, delta_ci95_high=delta,
                    bootstrap_iterations=n_resamples, bootstrap_degenerate=True)
    result = scipy_bootstrap(
        data=(np.arange(n_groups), np.arange(n_groups, 2 * n_groups)),
        statistic=cluster_f1_statistic(a, b),
        paired=True,
        vectorized=True,
        method="percentile",
        confidence_level=0.95,
        alternative="two-sided",
        n_resamples=n_resamples,
        batch=128,
        rng=np.random.default_rng(seed),
    )
    return dict(delta_ci95_low=float(result.confidence_interval.low),
                delta_ci95_high=float(result.confidence_interval.high),
                bootstrap_iterations=len(result.bootstrap_distribution),
                bootstrap_degenerate=bool(np.ptp(result.bootstrap_distribution) == 0))


def paired_randomization_f1(a, b, n_resamples, seed):
    """SciPy paired sample swaps, treating a cluster's counts as one object.

    The permuted observations are indices of whole (TP, FP, FN) vectors, so
    counts belonging to one cluster can never be swapped independently.
    The statistic is pooled Conditioned F1 - Neutral F1, not mean cluster F1.
    """
    n_groups = len(a)
    if n_groups == 1:
        # SciPy requires at least two observations per sample. With one cluster,
        # the only two assignments have +/-delta, giving two-sided p=1 exactly.
        return dict(p_value=1.0, permutation_iterations=2, permutation_exact=True)
    # SciPy 1.16 represents the number of assignments (2**n_groups) as a float.
    # Above 1023 clusters this becomes inf, correctly selecting Monte Carlo.
    # Suppress the expected overflow warning; no assignments are enumerated.
    with np.errstate(over="ignore"):
        result = permutation_test(
            data=(np.arange(n_groups), np.arange(n_groups, 2 * n_groups)),
            statistic=cluster_f1_statistic(a, b),
            permutation_type="samples",
            vectorized=True,
            alternative="two-sided",
            n_resamples=n_resamples,
            batch=128,
            rng=np.random.default_rng(seed),
        )
    return dict(p_value=float(result.pvalue),
                permutation_iterations=len(result.null_distribution),
                permutation_exact=n_resamples >= 2 ** n_groups)


def paired_f1_statistics(y, neutral, conditioned, groups, bootstrap=10_000,
                         permutations=10_000, seed=42):
    """Cluster-paired percentile bootstrap and two-sided randomization test.

    Resample whole clusters with replacement, retaining all their paired items.
    For the test, independently exchange the two prediction vectors in each
    cluster. Recompute pooled item-level F1 each time (not mean cluster F1).
    The randomization null assumes exchangeability of conditions by cluster.
    """
    y, neutral, conditioned = [np.asarray(x) for x in (y, neutral, conditioned)]
    if not len(y) or not (len(y) == len(neutral) == len(conditioned) == len(groups)):
        raise ValueError("Nonempty, equally sized paired arrays required")
    if any(not np.isin(x, [0, 1]).all() for x in (y, neutral, conditioned)):
        raise ValueError("Statistics require binary labels and predictions")
    if bootstrap < 1 or permutations < 1:
        raise ValueError("Resampling counts must be positive")
    codes, names = pd.factorize(np.asarray(groups), sort=True)
    n_groups = len(names)
    a = grouped_counts(y, neutral, codes, n_groups)
    b = grouped_counts(y, conditioned, codes, n_groups)
    fa, fb = positive_class_f1(y, neutral), positive_class_f1(y, conditioned)
    delta = fb - fa
    confidence_interval = paired_bootstrap_f1(a, b, bootstrap,
                                              stable_seed(seed, "bootstrap"))
    randomization = paired_randomization_f1(a, b, permutations,
                                           stable_seed(seed, "permutation"))
    return dict(f1_neutral=fa, f1_conditioned=fb, f1_delta=delta,
                n_clusters=n_groups,
                f1_method="sklearn.metrics.f1_score:binary:pos_label=1:zero_division=0",
                bootstrap_method="scipy.stats.bootstrap:paired:percentile:cluster",
                permutation_resamples_requested=permutations,
                permutation_method="scipy.stats.permutation_test:samples:two-sided",
                comparison_seed=seed, **confidence_interval, **randomization)


def index_entries(path):
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    ids = [e.get("unique_id") for e in entries]
    if any(not isinstance(uid, str) or not uid.strip() for uid in ids):
        raise ValueError(f"Missing/non-string unique_id in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate unique_id in {path}; refusing ambiguous labels")
    return {e["unique_id"]: e for e in entries}


def read_predictions(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"unique_id": str}, keep_default_na=False)
    required = {"unique_id", "model_answer", "prompt"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns {required - set(df.columns)} in {path}")
    if df.unique_id.str.strip().eq("").any():
        raise ValueError(f"Missing unique_id in {path}")
    duplicates = df.loc[df.unique_id.duplicated(False), "unique_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate unique_id in {path}: {duplicates[:10]}; "
                         "refusing ambiguous pairing (including identical duplicates)")
    df["prediction"] = pd.to_numeric(df.model_answer, errors="coerce")
    return df.set_index("unique_id", verify_integrity=True)


def prompt_renderer():
    """Reuse the actual prompt method without loading models, torch or tokens."""
    path = HERE / "src" / "decoder_handler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecoderHandler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    instance = SimpleNamespace(PROMPT_STRATEGIES=["no_perspective", *STRATEGIES])
    return lambda strategy, text: namespace["_get_prompt"](instance, strategy, text)


def conversation(entry):
    turns = [dict(t) for t in entry["dialogue"]["turns"]]
    if turns:
        turns[-1].pop("u2", None)
    return "".join(f"{speaker}: {turn[speaker]}\n" for turn in turns
                   for speaker in ("u1", "u2") if speaker in turn)


def valid_prompt(prompt, strategy, text, render):
    """Exact conversation; normalize only template whitespace outside it."""
    if not isinstance(prompt, str) or prompt.count("<CONVERSATION>") != 1:
        return False
    before, rest = prompt.split("<CONVERSATION>")
    if rest.count("</CONVERSATION>") != 1:
        return False
    actual, after = rest.split("</CONVERSATION>")
    expected_before, expected_rest = render(strategy, text).split("<CONVERSATION>")
    _, expected_after = expected_rest.split("</CONVERSATION>")
    return (actual == text and before.split() == expected_before.split()
            and after.split() == expected_after.split())


def pair_examples(entries, valid_ids, target, strategy, neutral, conditioned, render, dataset):
    """Outer audit by ID; a shared valid intersection defines both F1 samples."""
    records = []
    all_ids = set(entries) | set(valid_ids)
    for df in (neutral, conditioned):
        if df is not None:
            all_ids.update(df.index)
    aligned = pd.DataFrame(index=pd.Index(sorted(all_ids), name="unique_id"))
    for condition, df in (("neutral", neutral), ("conditioned", conditioned)):
        if df is None:
            aligned[condition] = np.nan
            aligned[f"{condition}_prompt"] = None
            aligned[f"{condition}_present"] = False
        else:
            values = df[["prediction", "prompt"]].rename(
                columns={"prediction": condition, "prompt": f"{condition}_prompt"})
            values[f"{condition}_present"] = True
            aligned = aligned.join(values, how="left", validate="one_to_one")
            aligned[f"{condition}_present"] = aligned[f"{condition}_present"].eq(True)
    for uid in sorted(all_ids):
        entry = entries.get(uid)
        ann = entry.get("annotations", {}) if entry else {}
        vote_present = f"{target}_n" in ann  # same eligibility as historical eval.py
        label = ann.get(target)
        row = dict(unique_id=uid, in_final=uid in valid_ids,
                   has_annotations=entry is not None, has_votes=vote_present,
                   reference=label, reference_valid=label in (0, 1),
                   cluster=(f"{entry['source']}:{entry['key_1' if dataset == 'CSC' else 'key_2']}"
                            if entry else ""))
        reasons = []
        if uid not in valid_ids:
            reasons.append("outside_final")
        if entry is None:
            reasons.append("missing_annotations")
        if not vote_present:
            reasons.append("no_perspective_votes")
        if label not in (0, 1):
            reasons.append("invalid_or_missing_reference")
        text = conversation(entry) if entry else None
        for condition, prompt_strategy in (("neutral", "no_perspective"),
                                            ("conditioned", strategy)):
            present = bool(aligned.at[uid, f"{condition}_present"])
            prediction = aligned.at[uid, condition]
            binary = present and prediction in (0, 1)
            protocol_ok = (present and text is not None
                           and valid_prompt(aligned.at[uid, f"{condition}_prompt"],
                                            prompt_strategy, text, render))
            row.update({f"{condition}_present": present, condition: prediction,
                        f"{condition}_valid": binary,
                        f"{condition}_prompt_ok": protocol_ok})
            if not present:
                reasons.append(f"missing_{condition}")
            elif not binary:
                reasons.append(f"invalid_{condition}")
            if present and entry and not protocol_ok:
                reasons.append(f"protocol_mismatch_{condition}")
        row["included"] = not reasons
        row["exclusion_reasons"] = ";".join(reasons)
        records.append(row)
    audit = pd.DataFrame(records)
    eligible = audit.in_final & audit.has_annotations & audit.has_votes & audit.reference_valid
    both = audit.neutral_present & audit.conditioned_present
    paired = audit.loc[audit.included].copy()
    counts = dict(n_final=len(valid_ids), n_eligible=int(eligible.sum()),
                  n_without_votes=int((audit.in_final & ~audit.has_votes).sum()),
                  n_invalid_references=int((audit.in_final & audit.has_votes & ~audit.reference_valid).sum()),
                  n_missing_annotations=int((audit.in_final & ~audit.has_annotations).sum()),
                  n_available_pairs=int((eligible & both).sum()), n_samples=len(paired),
                  n_excluded_eligible=int(eligible.sum()) - len(paired),
                  paired_coverage=len(paired) / eligible.sum() if eligible.any() else 0.0,
                  n_duplicates_neutral=0, n_duplicates_conditioned=0)
    for name in ("neutral", "conditioned"):
        present, binary = audit[f"{name}_present"], audit[f"{name}_valid"]
        counts.update({f"n_rows_{name}": int(present.sum()),
                       f"n_outside_final_{name}": int((present & ~audit.in_final).sum()),
                       f"n_missing_{name}": int((eligible & ~present).sum()),
                       f"n_invalid_{name}": int((eligible & present & ~binary).sum()),
                       f"n_invalid_all_rows_{name}": int((present & ~binary).sum()),
                       f"n_protocol_mismatch_{name}": int((eligible & present & ~audit[f'{name}_prompt_ok']).sum())})
    return paired, counts, audit


def apply_holm(results):
    """Preserve historical dataset-model families, including CSC relationships."""
    for (_, _), indices in results.groupby(["dataset", "model"]).groups.items():
        # Missing comparisons conservatively occupy their planned family slot.
        p = results.loc[indices, "p_value"].fillna(1.0)
        reject, adjusted, _, _ = multipletests(p, method="holm", alpha=0.05)
        results.loc[indices, "p_value_holm"] = adjusted
        results.loc[indices, "significant_holm"] = reject
        results.loc[indices, "holm_family_size"] = len(indices)
    results.loc[results.p_value.isna(), ["p_value_holm", "significant_holm"]] = np.nan
    return results


def latex_table(df, path, complementary=False):
    columns = ["dataset", "model_name", "perspective", "n_samples", "n_clusters",
               "positive_prevalence", "f1_neutral", "f1_conditioned", "f1_delta",
               "delta_ci95_low", "delta_ci95_high", "p_value", "p_value_holm"]
    headers = ["Dataset", "Model", "Perspective", "N", "Clusters", "Positive prevalence",
               "Neutral F1", "Conditioned F1", "Delta F1", "CI 95 low", "CI 95 high", "p", "p Holm"]
    if complementary:
        columns = ["dataset", "model_name", "perspective", "n_samples", "f1_global",
                   "f1_perspective", "f1_delta_reference"]
        headers = ["Dataset", "Model", "Perspective", "N", "Global F1 (conditioned)",
                   "Perspective F1 (conditioned)", "Delta reference F1"]
    df.reindex(columns=columns).set_axis(headers, axis=1).to_latex(
        path, index=False, escape=True, float_format=lambda x: f"{x:.4f}", na_rep="--",
        longtable=True)


def publication_score_pair(row):
    """Bold displayed winners/ties; star only the strictly higher raw score."""
    if row is None or pd.isna(row.get("f1_neutral")) or pd.isna(row.get("f1_conditioned")):
        return "--", "--"
    a, b = float(row["f1_neutral"]), float(row["f1_conditioned"])
    shown_a, shown_b = f"{a:.2f}", f"{b:.2f}"
    tied = shown_a == shown_b
    p = row.get("p_value_holm", np.nan)
    significant = pd.notna(p) and p < 0.05

    def cell(shown, bold, star):
        value = rf"\mathbf{{{shown}}}" if bold else shown
        return "$" + value + (r"^{*}" if star else "") + "$"

    return (cell(shown_a, tied or a > b, significant and a > b),
            cell(shown_b, tied or b > a, significant and b > a))


def publication_latex_table(df, path, kind, n_resamples):
    """Paper tables: paired scores, datasets side by side, models in blocks."""
    relationships = kind == "relationships"
    caption = (
        r"Positive-class $F_1$ scores on \texttt{CANTRIP}$_{CSC}$ "
        r"for neutral (\emph{Neutral}) and relationship-conditioned (\emph{Cond.}) predictions,"
        if relationships else
        r"Positive-class $F_1$ scores for neutral (\emph{Neutral}) "
        r"and trait-conditioned (\emph{Cond.}) predictions,"
    )
    caption += (
        " both evaluated against the same matching "
        + ("relationship" if relationships else "trait")
        + "-specific proxy labels on the same examples.\n"
        "Boldface marks the higher score within each pair; "
        "ties at the displayed precision are bolded in both columns.\n"
        r"An asterisk ($^{*}$) marks the higher unrounded score when the paired "
        r"difference is significant after Holm correction ($p_{\mathrm{Holm}} < 0.05$)." "\n"
        "Significance is assessed using a two-sided paired approximate randomization "
        f"test with {n_resamples:,} randomizations, swapping whole "
        + ("original-context clusters." if relationships else "original-post (MP) or original-context (CSC) clusters.")
        + " Holm families comprise all comparisons within each dataset--model pair "
        "(six for MP; eight for CSC, including traits and relationships)."
    )
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{", caption, "}",
             rf"\setlength{{\tabcolsep}}{{{'4' if relationships else '3'}pt}}"]
    lines += ([r"\begin{tabular}{llcc}"] if relationships else [
        r"\begin{tabular}{", r"l@{\hspace{4pt}}", r"l@{\hspace{6pt}}",
        r"c@{\hspace{2pt}}", r"c@{\hspace{6pt}}", r"c@{\hspace{2pt}}", "c", "}"])
    lines += [r"\toprule"]
    if relationships:
        lines += [r"\textbf{Model} & \textbf{Relationship} &",
                  r"\multicolumn{2}{c}{\textbf{\texttt{CANTRIP}$_{\mathbf{CSC}}$}} \\",
                  r"\cmidrule(lr){3-4}", r"& & Neutral & Cond. \\"]
    else:
        lines += [r"\textbf{Model} & \textbf{Trait} &",
                  r"\multicolumn{2}{c}{\textbf{\texttt{CANTRIP}$_{\mathbf{MP}}$}} &",
                  r"\multicolumn{2}{c}{\textbf{\texttt{CANTRIP}$_{\mathbf{CSC}}$}} \\",
                  r"\cmidrule(lr){3-4}", r"\cmidrule(lr){5-6}",
                  r"& & Neutral & Cond. & Neutral & Cond. \\"]
    lines += [r"\midrule", ""]
    perspectives = (["Colleagues", "Best Friends"] if relationships else
                    ["Female", "Male", "Boomer", "Gen X", "Gen Y", "Gen Z"])
    datasets = ["CSC"] if relationships else ["MultiPICo", "CSC"]
    for index, (model, model_name) in enumerate(MODELS.items()):
        if index:
            lines += [r"\midrule", ""]
        name = model_name.replace("-", "--")
        for number, perspective in enumerate(perspectives):
            if not relationships and number == 2:
                lines += [r"\cmidrule(lr){2-6}"]
            cells = []
            for dataset in datasets:
                rows = df.loc[(df.model == model) & (df.perspective == perspective)
                              & (df.dataset == dataset)]
                if len(rows) > 1:
                    raise ValueError(f"Duplicate table comparison: {model}/{dataset}/{perspective}")
                cells.extend(publication_score_pair(rows.iloc[0] if len(rows) else None))
            prefix = rf"\multirow{{{len(perspectives)}}}{{*}}{{{name}}}" if number == 0 else ""
            lines += [prefix + " & " + perspective + " & " + " & ".join(cells) + r" \\"]
        lines += [""]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\label{{tab:experiment2_{kind}_neutral_conditioned}}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=HERE / "output" / "decoders")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "perspectives")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.predictions_dir = args.predictions_dir.resolve()
    if args.bootstrap < 1 or args.permutations < 1:
        parser.error("Resampling counts must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    render = prompt_renderer()
    results, audits, complementary, inputs = [], [], [], set()
    for dataset in ("MultiPICo", "CSC"):
        labels_path = HERE / "disaggregated_dataset" / f"{dataset}{SUFFIX}_filtered_with_semiaggregates.json"
        final_path = HERE.parent / "datasets" / f"{dataset}{SUFFIX}_final.json"
        entries, final = index_entries(labels_path), index_entries(final_path)
        inputs.update((labels_path, final_path))
        # Final-set conversations must be the same as the label source.
        for uid, entry in final.items():
            if uid in entries and conversation(entry) != conversation(entries[uid]):
                raise ValueError(f"Final/annotation conversation mismatch: {dataset}/{uid}")
        for model, model_name in MODELS.items():
            directory = args.predictions_dir / model
            neutral_path = directory / f"no_perspective_{model}_{dataset}_conversation.csv"
            neutral = read_predictions(neutral_path)
            if neutral is not None:
                inputs.add(neutral_path)
            for strategy, (target, perspective, kind) in STRATEGIES.items():
                if dataset == "MultiPICo" and kind == "relationships":
                    continue
                conditioned_path = directory / f"{strategy}_{model}_{dataset}_conversation.csv"
                conditioned = read_predictions(conditioned_path)
                if conditioned is not None:
                    inputs.add(conditioned_path)
                paired, counts, audit = pair_examples(entries, final, target, strategy,
                                                      neutral, conditioned, render, dataset)
                metadata = dict(dataset=dataset, model=model, model_name=model_name,
                                strategy=strategy, perspective=perspective, kind=kind,
                                target_label=target)
                audit = audit.assign(**metadata)
                audits.append(audit)
                result = dict(**metadata, **counts, neutral_file=relative_or_absolute(neutral_path, HERE),
                              conditioned_file=relative_or_absolute(conditioned_path, HERE),
                              status="ok" if len(paired) else "no_valid_pairs",
                              protocol_status="prompts_verified_generation_unverified",
                              resampling_unit="source:context_name" if dataset == "CSC" else "source:post_id",
                              ci_multiple_testing_adjusted=False, seed=args.seed,
                              p_value=np.nan)
                if counts["n_protocol_mismatch_neutral"] or counts["n_protocol_mismatch_conditioned"]:
                    result["protocol_status"] = "prompt_mismatches_excluded_generation_unverified"
                if len(paired):
                    y, a, b = [paired[c].to_numpy(dtype=int) for c in ("reference", "neutral", "conditioned")]
                    result.update(paired_f1_statistics(
                        y, a, b, paired.cluster.to_numpy(), args.bootstrap, args.permutations,
                        stable_seed(args.seed, dataset, model, strategy)))
                    result.update(n_positive=int(y.sum()), positive_prevalence=float(y.mean()))
                    global_key = "aggregate_sarcasm_label" if dataset == "CSC" else "aggregate_irony_label"
                    global_y = np.array([entries[uid]["annotations"].get(global_key)
                                         for uid in paired.unique_id])
                    mask = np.isin(global_y, [0, 1])
                    if mask.any():
                        global_f1 = positive_class_f1(global_y[mask], b[mask])
                        perspective_f1 = positive_class_f1(y[mask], b[mask])
                        complementary.append(dict(**metadata, n_samples=int(mask.sum()),
                                                  f1_global=global_f1, f1_perspective=perspective_f1,
                                                  f1_delta_reference=perspective_f1-global_f1))
                results.append(result)
                print(f"{dataset} / {model_name} / {perspective}: N={len(paired)}, "
                      f"delta={result.get('f1_delta', float('nan')):+.4f}", flush=True)
    df = apply_holm(pd.DataFrame(results))
    df.to_csv(output / "neutral_conditioned.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(output / "pair_audit.csv.gz", index=False,
                                             compression={"method": "gzip", "mtime": 0})
    for kind in ("traits", "relationships"):
        subset = df.loc[df.kind == kind]
        subset.to_csv(output / f"{kind}.csv", index=False)
        latex_table(subset, output / f"{kind}_statistics.tex")
        publication_latex_table(subset, output / f"{kind}.tex", kind, args.permutations)
    comp = pd.DataFrame(complementary)
    comp.to_csv(output / "complementary_global_perspective.csv", index=False)
    latex_table(comp, output / "complementary_global_perspective.tex", complementary=True)
    inputs.update([Path(__file__), HERE / "test_eval_perspectives.py",
                   HERE / "src" / "config.py", HERE / "src" / "decoder_handler.py",
                   HERE / "src" / "model_backend.py", HERE / "src" / "dataset_loader.py",
                   HERE / "disaggregated_dataset" / "updatingSemiaggregationMultiPICoCSC.ipynb"])
    manifest = dict(bootstrap=args.bootstrap, permutations=args.permutations, seed=args.seed,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__,
                    scikit_learn=sklearn.__version__,
                    f1_method="sklearn.metrics.f1_score",
                    bootstrap_method="scipy.stats.bootstrap",
                    bootstrap_paired=True, bootstrap_interval="percentile",
                    bootstrap_confidence_level=0.95,
                    permutation_method="scipy.stats.permutation_test",
                    permutation_type="samples", permutation_alternative="two-sided",
                    files={relative_or_absolute(p, HERE.parent.parent): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(inputs)})
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(df)} comparisons to {output}")


if __name__ == "__main__":
    main()
