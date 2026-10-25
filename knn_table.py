import csv
import re

from config import KNN_DIR, OUT_43

PATTERNS = {
    "Best CV Balanced Accuracy": r"Best CV Balanced Accuracy\s*:\s*([0-9.]+)",
    "Accuracy": r"Accuracy\s*:\s*([0-9.]+)",
    "Precision": r"Precision \(Weighted\)\s*:\s*([0-9.]+)",
    "Recall": r"Recall \(Weighted\)\s*:\s*([0-9.]+)",
    "F1-Score": r"F1-Score \(Weighted\)\s*:\s*([0-9.]+)",
    "Sensitivity": r"Sensitivity \(Abnormal\)\s*:\s*([0-9.]+)",
    "Specificity": r"Specificity \(Normal\)\s*:\s*([0-9.]+)",
    "ROC-AUC": r"ROC-AUC\s*:\s*([0-9.]+)"
}

def extract_value(text, pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def main():
    reports = sorted(KNN_DIR.glob("Report_KNN_*.txt"))

    if not reports:
        raise FileNotFoundError(f"Tidak ditemukan Report_KNN di {KNN_DIR}")

    rows = []

    for report in reports:
        text = report.read_text(encoding="utf-8")

        parameter_match = re.search(r"Best Parameters\s*:\s*(.+)", text)

        row = {
            "Model": report.stem.replace("Report_KNN_", ""),
            "Best Parameters": parameter_match.group(1).strip() if parameter_match else ""
        }

        for metric, pattern in PATTERNS.items():
            row[metric] = extract_value(text, pattern)

        rows.append(row)

    output_path = OUT_43 / "4_3_knn_results.csv"

    fieldnames = [
        "Model",
        "Best Parameters",
        "Best CV Balanced Accuracy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Sensitivity",
        "Specificity",
        "ROC-AUC"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Saved] {output_path}")

    for row in rows:
        print("\n" + "=" * 60)
        print(row["Model"])
        print("=" * 60)

        for key in fieldnames[1:]:
            print(f"{key:28}: {row[key]}")

if __name__ == "__main__":
    main()