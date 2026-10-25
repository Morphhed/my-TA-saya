import csv
import re
import numpy as np
import matplotlib.pyplot as plt

from bab4_config import FINETUNE_REPORTS, OUT_45, TEMPERATURE

PATTERNS = {
    "Accuracy": r"Accuracy\s*:\s*([0-9.]+)",
    "Precision": r"Precision \(Weighted\)\s*:\s*([0-9.]+)",
    "Recall": r"Recall \(Weighted\)\s*:\s*([0-9.]+)",
    "F1-Score": r"F1-Score \(Weighted\)\s*:\s*([0-9.]+)",
    "Sensitivity": r"Sensitivity \(Abnormal\)\s*:\s*([0-9.]+)",
    "Specificity": r"Specificity \(Normal\)\s*:\s*([0-9.]+)",
    "ROC-AUC": r"ROC-AUC\s*:\s*([0-9.]+)"
}

def parse_report(model_name, path):
    if not path.exists():
        print(f"[Warning] Report tidak ditemukan: {path}")
        return None

    text = path.read_text(encoding="utf-8")
    row = {"Model": model_name}

    for metric, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        row[metric] = float(match.group(1)) if match else None

    return row

def main():
    rows = []

    for model_name, report_path in FINETUNE_REPORTS.items():
        row = parse_report(model_name, report_path)

        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("Tidak ada report fine-tuning yang dapat dibaca.")

    fieldnames = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity", "ROC-AUC"]

    csv_path = OUT_45 / "4_5_model_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    priority_metrics = ["Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]
    model_names = [row["Model"] for row in rows]

    x = np.arange(len(priority_metrics))
    width = 0.8 / len(rows)

    plt.figure(figsize=(12, 6))

    for index, row in enumerate(rows):
        values = [row[m] for m in priority_metrics]
        offset = (index - (len(rows) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=row["Model"])

    plt.xticks(x, priority_metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title(f"Perbandingan Kinerja Model - SimCLR Temperature {TEMPERATURE}")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    chart_path = OUT_45 / "4_5_model_comparison_bar.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {chart_path}")

    print("\n" + "=" * 80)

    for row in rows:
        print(row)

if __name__ == "__main__":
    main()