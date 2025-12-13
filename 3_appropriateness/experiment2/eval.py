import pandas as pd
from evaluate import load
from statsmodels.stats.contingency_tables import mcnemar

STRATEGY_TO_TARGET = {
    "gender_male": "aggregation_on_male",
    "gender_female": "aggregation_on_female",
    "generation_boomer": "aggregation_on_boomer",
    "generation_gen_z": "aggregation_on_genz",
    "generation_gen_y": "aggregation_on_geny",
    "generation_gen_x": "aggregation_on_genx",
}

OUTPUT_CSV = "./evaluation_results_semiaggregated.csv"

# =========================================================
def compute_metrics(df, target_col, accuracy_metric, precision_metric, recall_metric, f1_metric):
    df_eval = df[df[target_col].notna()].copy()
    if len(df_eval) == 0:
        return None

    df_eval["model_answer"] = df_eval["model_answer"].astype(int)
    df_eval[target_col] = df_eval[target_col].astype(int)

    return {
        "accuracy": accuracy_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col]
        )["accuracy"],
        "precision": precision_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col],
            pos_label=1,
            average="binary"
        )["precision"],
        "recall": recall_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col],
            pos_label=1,
            average="binary"
        )["recall"],
        "f1": f1_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col],
            pos_label=1,
            average="binary"
        )["f1"],
        "n_samples": len(df_eval),
    }

def mcnemar_test(df, label_col, semi_col):
    df_eval = df[df[semi_col].notna()].copy()
    if len(df_eval) == 0:
        return None

    correct_label = df_eval["model_answer"] == df_eval[label_col]
    correct_semi = df_eval["model_answer"] == df_eval[semi_col]

    b = ((correct_label == True) & (correct_semi == False)).sum()
    c = ((correct_label == False) & (correct_semi == True)).sum()

    table = [[0, b],
             [c, 0]]

    result = mcnemar(table, exact=(b + c < 25))

    return {
        "mcnemar_p_value": result.pvalue,
        "mcnemar_b": b,
        "mcnemar_c": c,
    }

# =========================================================
def main():
    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    results = []

    files = {
        'CSC': {
            "Llama-3.2-1B-Instruct": [
                # ---- I TUOI FILE ----
            ],
        },
        'MultiPICo': {
            "Llama-3.2-1B-Instruct": [
                # ---- I TUOI FILE ----
            ],
        }
    }

    for dataset_name, models in files.items():
        for model_name, file_paths in models.items():
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                df["model_answer"] = df["model_answer"].astype(int)

                # -----------------------------
                # 1) Aggregated label
                # -----------------------------
                res_label = compute_metrics(
                    df, "label",
                    accuracy_metric, precision_metric, recall_metric, f1_metric
                )

                if res_label is not None:
                    results.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "file": file_path,
                        "strategy": "aggregated",
                        "target_label": "label",
                        **res_label,
                        "mcnemar_p_value": None,
                        "mcnemar_b": None,
                        "mcnemar_c": None,
                    })

                # -----------------------------
                # 2) Semi-aggregated (se strategy nel filename)
                # -----------------------------
                strategy = None
                for s in STRATEGY_TO_TARGET:
                    if s in file_path:
                        strategy = s
                        break

                if strategy is None:
                    continue

                semi_col = STRATEGY_TO_TARGET[strategy]
                if semi_col not in df.columns:
                    continue

                res_semi = compute_metrics(
                    df, semi_col,
                    accuracy_metric, precision_metric, recall_metric, f1_metric
                )

                if res_semi is None:
                    continue

                mc = mcnemar_test(df, "label", semi_col)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "file": file_path,
                    "strategy": strategy,
                    "target_label": semi_col,
                    **res_semi,
                    "mcnemar_p_value": None if mc is None else mc["mcnemar_p_value"],
                    "mcnemar_b": None if mc is None else mc["mcnemar_b"],
                    "mcnemar_c": None if mc is None else mc["mcnemar_c"],
                })

    # =====================================================
    # SALVATAGGIO
    # =====================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Results saved to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df_results)}")

# =========================================================
if __name__ == "__main__":
    main()