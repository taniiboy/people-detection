from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "output" / "model_comparison" / "comparison_results.csv"
OUTPUT_DIR = CSV_PATH.parent

FAMILY_COLORS = {
    "YOLOv8": "red",
    "YOLO11": "orange",
    "YOLO26": "blue",
}

SIZE_ORDER = {
    "n": 0,
    "s": 1,
    "m": 2,
    "l": 3,
    "x": 4,
}

def get_family(model_name):
    model_name = model_name.lower()
    if "yolov8" in model_name:
        return "YOLOv8"
    if "yolo11" in model_name:
        return "YOLO11"
    if "yolo26" in model_name:
        return "YOLO26"
    return "Unknown"

def get_size(model_name):
    stem = Path(model_name).stem
    return stem[-1]

def load_results():
    df = pd.read_csv(CSV_PATH)
    if "total_error" not in df.columns:
        df["total_error"] = df["entry_error"] + df["exit_error"]
    df["family"] = df["model"].apply(get_family)
    df["size"] = df["model"].apply(get_size)
    df["size_rank"] = df["size"].map(SIZE_ORDER)
    df = df.sort_values(["family", "size_rank"])
    return df

def plot_accuracy_vs_speed(df):
    plt.figure(figsize=(10, 7))

    for family, group in df.groupby("family"):
        group = group.sort_values("size_rank")
        color = FAMILY_COLORS.get(family, "gray")
        plt.plot(
            group["fps_processing"],
            group["total_error"],
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
            label=family,
        )

        for _, row in group.iterrows():
            label = row["model"].replace(".pt", "")
            plt.annotate(
                label,
                (row["fps_processing"], row["total_error"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )

    plt.xlabel("FPS")
    plt.ylabel("Gesamtfehler")
    plt.title("Genauigkeit vs Geschwindigkeit nach YOLO-Modellfamilie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = OUTPUT_DIR / "accuracy_vs_speed_family_lines.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

def main():
    print("Lade CSV:", CSV_PATH)
    df = load_results()
    print("\nGeladene Ergebnisse:")
    print(
        df[
            [
                "model",
                "family",
                "size",
                "fps_processing",
                "entry_count",
                "exit_count",
                "entry_error",
                "exit_error",
                "total_error",
            ]
        ]
    )

    plot_accuracy_vs_speed(df)

if __name__ == "__main__":
    main()
