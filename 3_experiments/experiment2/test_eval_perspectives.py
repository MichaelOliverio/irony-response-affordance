"""Run with: python -m unittest discover -s 3_experiments/experiment2 -p test_eval_perspectives.py"""

import itertools
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from sklearn.metrics import f1_score

import eval_perspectives as ep


class PairedStatisticsTests(unittest.TestCase):
    def test_vectorized_resampling_f1_matches_sklearn_weighted_confusion_counts(self):
        # One weighted representative of each TP/FP/FN outcome is sufficient
        # for positive-class F1. TNs do not affect the score.
        rng = np.random.default_rng(197)
        counts = np.vstack((np.zeros((1, 3), dtype=int), np.eye(3, dtype=int),
                            rng.integers(0, 5000, size=(64, 3))))
        expected = [f1_score([1, 0, 1], [1, 1, 0], sample_weight=weights,
                             average="binary", pos_label=1, zero_division=0)
                    for weights in counts]
        np.testing.assert_allclose(ep.f1_from_counts(counts), expected, rtol=1e-14)

    def test_public_metric_rejects_invalid_values_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            ep.positive_class_f1([0, 1], [0.7, 1])

    def test_identical_predictions(self):
        result = ep.paired_f1_statistics([0, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0],
                                         ["a", "a", "b", "c"], 200, 200)
        for key in ("f1_delta", "delta_ci95_low", "delta_ci95_high"):
            self.assertEqual(result[key], 0)
        self.assertEqual(result["p_value"], 1)
        self.assertTrue(result["bootstrap_degenerate"])

    def test_delta_direction_and_positive_class(self):
        # Neutral F1(positive=1)=0; conditioned F1=1. Class 0 is not positive.
        forward = ep.paired_f1_statistics([1, 0], [0, 0], [1, 0], ["a", "b"], 200, 200)
        reverse = ep.paired_f1_statistics([1, 0], [1, 0], [0, 0], ["a", "b"], 200, 200)
        self.assertEqual(forward["f1_neutral"], 0)
        self.assertEqual(forward["f1_conditioned"], 1)
        self.assertEqual(forward["f1_delta"], 1)
        self.assertEqual(reverse["f1_delta"], -1)
        self.assertEqual(reverse["p_value"], forward["p_value"])
        self.assertEqual(ep.f1_from_counts(np.zeros(3)), 0)

    def test_cluster_bootstrap_and_permutation_match_direct_calculation(self):
        y = np.array([1, 0, 1, 1, 0])
        a = np.array([0, 1, 1, 0, 0])
        b = np.array([1, 0, 1, 1, 1])
        groups = np.array(["a", "a", "b", "c", "c"])
        seed, iterations = 73, 80
        result = ep.paired_f1_statistics(y, a, b, groups, iterations, iterations, seed)

        def f1(yy, pp):
            tp = sum((yy == 1) & (pp == 1))
            denominator = sum(yy == 1) + sum(pp == 1)
            return 2 * tp / denominator if denominator else 0.0

        rng = np.random.default_rng(ep.stable_seed(seed, "bootstrap"))
        deltas = []
        for _ in range(iterations):
            sampled = rng.integers(3, size=3)
            indices = np.concatenate([np.flatnonzero(groups == "abc"[i]) for i in sampled])
            deltas.append(f1(y[indices], b[indices]) - f1(y[indices], a[indices]))
        np.testing.assert_allclose([result["delta_ci95_low"], result["delta_ci95_high"]],
                                   np.quantile(deltas, [0.025, 0.975]))
        # With three clusters SciPy enumerates all 2**3 assignments exactly.
        null = []
        for swap_groups in itertools.product((0, 1), repeat=3):
            swap = np.array([swap_groups["abc".index(g)] for g in groups], dtype=bool)
            aa, bb = np.where(swap, b, a), np.where(swap, a, b)
            null.append(f1(y, bb) - f1(y, aa))
        null = np.asarray(null)
        less = np.mean(null <= result["f1_delta"] + 1e-12)
        greater = np.mean(null >= result["f1_delta"] - 1e-12)
        self.assertEqual(result["p_value"], min(1.0, 2 * min(less, greater)))
        self.assertEqual(result["permutation_iterations"], 8)
        self.assertTrue(result["permutation_exact"])
        self.assertEqual(result["n_clusters"], 3)

    def test_scipy_monte_carlo_matches_direct_item_level_cluster_swaps(self):
        groups = np.repeat(np.arange(6), [1, 2, 3, 1, 4, 2])
        y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
        a = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        b = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
        seed, iterations = 431, 31

        def f1(predictions):
            denominator = y.sum() + predictions.sum()
            return 2 * np.count_nonzero((y == 1) & (predictions == 1)) / denominator

        def direct_statistic(neutral_ids, conditioned_ids):
            # Reconstruct complete swapped prediction vectors, independently
            # of the implementation's TP/FP/FN representation.
            swap = (neutral_ids >= 6)[groups]
            neutral = np.where(swap, b, a)
            conditioned = np.where(swap, a, b)
            return f1(conditioned) - f1(neutral)

        expected = permutation_test(
            (np.arange(6), np.arange(6, 12)), direct_statistic,
            permutation_type="samples", vectorized=False, alternative="two-sided",
            n_resamples=iterations, batch=128,
            rng=np.random.default_rng(ep.stable_seed(seed, "permutation")))
        actual = ep.paired_f1_statistics(y, a, b, groups, 20, iterations, seed)
        self.assertEqual(actual["p_value"], expected.pvalue)
        self.assertFalse(actual["permutation_exact"])
        self.assertEqual(actual["permutation_iterations"], iterations)

    def test_one_cluster_has_no_two_sided_randomization_power(self):
        actual = ep.paired_f1_statistics([1, 0], [0, 0], [1, 0], ["a", "a"], 20, 20)
        self.assertEqual(actual["f1_delta"], 1)
        self.assertEqual(actual["p_value"], 1)
        self.assertEqual(actual["permutation_iterations"], 2)
        self.assertTrue(actual["permutation_exact"])

    def test_many_clusters_use_requested_monte_carlo_without_count_overflow_warning(self):
        a = np.tile([0, 1, 1], (1024, 1))
        b = np.tile([1, 0, 0], (1024, 1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = ep.paired_randomization_f1(a, b, 16, 431)
        self.assertEqual(result["permutation_iterations"], 16)
        self.assertFalse(result["permutation_exact"])
        self.assertFalse(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_holm_preserves_dataset_model_families(self):
        df = pd.DataFrame(dict(dataset=["CSC", "CSC", "CSC", "MP"],
                               model=["A", "A", "B", "A"],
                               p_value=[0.01, 0.04, 0.01, 0.03]))
        adjusted = ep.apply_holm(df)
        np.testing.assert_allclose(adjusted.p_value_holm, [0.02, 0.04, 0.01, 0.03])

    def test_missing_comparison_retains_planned_holm_slot(self):
        df = pd.DataFrame(dict(dataset=["CSC", "CSC"], model=["A", "A"],
                               p_value=[0.03, np.nan]))
        adjusted = ep.apply_holm(df)
        self.assertEqual(adjusted.loc[0, "p_value_holm"], 0.06)
        self.assertTrue(pd.isna(adjusted.loc[1, "p_value_holm"]))
        self.assertEqual(adjusted.loc[0, "holm_family_size"], 2)


class PublicationTableTests(unittest.TestCase):
    def test_displayed_ties_are_both_bold(self):
        cells = ep.publication_score_pair(dict(f1_neutral=0.5759, f1_conditioned=0.5792,
                                               p_value_holm=1.0))
        self.assertEqual(cells, (r"$\mathbf{0.58}$", r"$\mathbf{0.58}$"))

    def test_stars_use_holm_and_raw_winner(self):
        cells = ep.publication_score_pair(dict(f1_neutral=0.40, f1_conditioned=0.20,
                                               p_value_holm=0.01))
        self.assertEqual(cells, (r"$\mathbf{0.40}^{*}$", "$0.20$"))
        cells = ep.publication_score_pair(dict(f1_neutral=0.591, f1_conditioned=0.594,
                                               p_value_holm=0.01))
        self.assertEqual(cells, (r"$\mathbf{0.59}$", r"$\mathbf{0.59}^{*}$"))
        cells = ep.publication_score_pair(dict(f1_neutral=0.30, f1_conditioned=0.40,
                                               p_value=0.01, p_value_holm=0.10))
        self.assertTrue(all("^{*}" not in cell for cell in cells))

    def test_publication_layout_and_missing_values(self):
        rows = []
        for model in ep.MODELS:
            for _, (_, perspective, kind) in ep.STRATEGIES.items():
                for dataset in ("MultiPICo", "CSC"):
                    if kind == "relationships" and dataset != "CSC":
                        continue
                    rows.append(dict(model=model, perspective=perspective, dataset=dataset,
                                     kind=kind, f1_neutral=0.30, f1_conditioned=0.40,
                                     p_value_holm=0.01))
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            for kind, nrows in (("traits", 6), ("relationships", 2)):
                path = Path(directory) / f"{kind}.tex"
                ep.publication_latex_table(df[df.kind == kind], path, kind, 10_000)
                tex = path.read_text(encoding="utf-8")
                self.assertIn(r"\begin{table}[t]", tex)
                self.assertIn(rf"\multirow{{{nrows}}}{{*}}{{Qwen3.5--9B}}", tex)
                self.assertIn("10,000 randomizations", tex)
                self.assertIn(r"$p_{\mathrm{Holm}} < 0.05$", tex)
                self.assertNotIn("longtable", tex)
                if kind == "traits":
                    self.assertLess(tex.index(" & Boomer & "), tex.index(" & Gen X & "))
                    self.assertIn(r"\cmidrule(lr){2-6}", tex)
        self.assertEqual(ep.publication_score_pair(None), ("--", "--"))


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.target = "aggregation_on_female"
        self.entries = {
            uid: dict(source="fictionary", key_1="context1", key_2=uid,
                      dialogue={"turns": [{"u1": uid, "u2": "target reply"}]},
                      annotations={self.target: label, f"{self.target}_n": 1})
            for uid, label in (("a", 1), ("b", 0), ("c", 1))}
        self.render = ep.prompt_renderer()

    def predictions(self, strategy, values):
        return pd.DataFrame([
            dict(unique_id=uid, prediction=value,
                 prompt=self.render(strategy, ep.conversation(self.entries[uid])))
            for uid, value in values]).set_index("unique_id")

    def pair(self, neutral, conditioned):
        return ep.pair_examples(self.entries, self.entries, self.target, "gender_female",
                                neutral, conditioned, self.render, "CSC")

    def test_id_alignment_is_independent_of_row_order(self):
        neutral = self.predictions("no_perspective", [("c", 1), ("a", 0), ("b", 1)])
        conditioned = self.predictions("gender_female", [("b", 0), ("c", 1), ("a", 1)])
        paired, counts, _ = self.pair(neutral, conditioned)
        self.assertEqual(paired.unique_id.tolist(), ["a", "b", "c"])
        self.assertEqual(paired.reference.tolist(), [1, 0, 1])
        self.assertEqual(paired.neutral.tolist(), [0, 1, 1])
        self.assertEqual(paired.conditioned.tolist(), [1, 0, 1])
        self.assertEqual(counts["paired_coverage"], 1)
        self.assertEqual(paired.cluster.nunique(), 1)

    def test_invalid_and_missing_predictions_are_reported_on_same_sample(self):
        neutral = self.predictions("no_perspective", [("a", np.nan), ("b", 0), ("c", 1)])
        conditioned = self.predictions("gender_female", [("a", 1), ("b", 0)])
        paired, counts, audit = self.pair(neutral, conditioned)
        self.assertEqual(paired.unique_id.tolist(), ["b"])
        self.assertEqual(counts["n_invalid_neutral"], 1)
        self.assertEqual(counts["n_missing_conditioned"], 1)
        self.assertEqual(counts["n_excluded_eligible"], 2)
        self.assertIn("invalid_neutral", audit.set_index("unique_id").at["a", "exclusion_reasons"])

    def test_wrong_prompt_or_context_is_not_treated_as_conditioning(self):
        neutral = self.predictions("no_perspective", [("a", 1), ("b", 0)])
        conditioned = self.predictions("gender_male", [("a", 1), ("b", 0)])
        paired, counts, _ = self.pair(neutral, conditioned)
        self.assertTrue(paired.empty)
        self.assertEqual(counts["n_protocol_mismatch_conditioned"], 2)

    def test_references_are_read_once_from_original_perspective_labels(self):
        self.entries["a"]["annotations"]["aggregate_sarcasm_label"] = 0
        del self.entries["c"]["annotations"][f"{self.target}_n"]
        neutral = self.predictions("no_perspective", [("a", 1), ("b", 0), ("c", 1)])
        conditioned = self.predictions("gender_female", [("a", 1), ("b", 0), ("c", 1)])
        paired, counts, _ = self.pair(neutral, conditioned)
        self.assertEqual(paired.reference.tolist(), [1, 0])
        self.assertEqual(counts["n_without_votes"], 1)

    def test_missing_file_is_explicit_and_conversation_mismatch_excluded(self):
        neutral = self.predictions("no_perspective", [("a", 1)])
        paired, counts, _ = self.pair(neutral, None)
        self.assertTrue(paired.empty)
        self.assertEqual(counts["n_missing_conditioned"], 3)
        conditioned = self.predictions("gender_female", [("a", 1)])
        conditioned.loc["a", "prompt"] = self.render("gender_female", "u1: different\n")
        paired, counts, _ = self.pair(neutral, conditioned)
        self.assertTrue(paired.empty)
        self.assertEqual(counts["n_protocol_mismatch_conditioned"], 1)

    def test_duplicates_fail_instead_of_arbitrary_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.csv"
            path.write_text("unique_id,model_answer,prompt\na,1,p\na,0,p\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate unique_id"):
                ep.read_predictions(path)

    def test_pandas_join_rejects_duplicate_pairs_even_if_loader_is_bypassed(self):
        neutral = self.predictions("no_perspective", [("a", 1), ("b", 0)])
        conditioned = self.predictions("gender_female", [("a", 1), ("b", 0)])
        duplicated = pd.concat([conditioned, conditioned.loc[["a"]]])
        with self.assertRaises(pd.errors.MergeError):
            self.pair(neutral, duplicated)


if __name__ == "__main__":
    unittest.main()
