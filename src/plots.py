
import os
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["time", "path_length", "smoothness"]


def ensure_method_column(df: pd.DataFrame, default_method: str) -> pd.DataFrame:
    """Ensure a 'method' column exists (classical vs ppo)."""
    if "method" not in df.columns:
        df = df.copy()
        df["method"] = default_method
    return df


def safe_boxplot(d: pd.DataFrame, metric: str, out_path: str, title: str):
    """Create and save a boxplot if the metric exists and has data."""
    if metric not in d.columns or d[metric].dropna().empty:
        print(f"[WARN] Skipping boxplot for '{metric}' (missing/empty).")
        return

    plt.figure()
    d.boxplot(column=metric, by="method")
    plt.suptitle("")
    plt.title(title)
    plt.xlabel("method")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def safe_bar_rates(d: pd.DataFrame, out_path: str, title: str):
    """Create success/collision rate bar chart if columns exist."""
    cols = [c for c in ["reached", "collision"] if c in d.columns]
    if not cols:
        print("[WARN] No reached/collision columns found; skipping rate bar plot.")
        return

    plt.figure()
    summary = d.groupby("method")[cols].mean()
    summary.plot(kind="bar")
    plt.title(title)
    plt.ylabel("rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs("outputs/plots", exist_ok=True)

    # Load metrics
    df_c = pd.read_csv("outputs/metrics_classical.csv")
    df_p = pd.read_csv("outputs/metrics_ppo.csv")

    df_c = ensure_method_column(df_c, "classical")
    df_p = ensure_method_column(df_p, "ppo")

    # Merge + save (useful for report + debugging)
    df = pd.concat([df_c, df_p], ignore_index=True)
    df.to_csv("outputs/metrics_all.csv", index=False)
    print("Saved merged metrics: outputs/metrics_all.csv")

    # Basic schema sanity print (helps you screenshot evidence)
    print("\nColumns in merged df:", list(df.columns))
    if "scenario" not in df.columns:
        raise ValueError("Expected a 'scenario' column in the metrics CSVs.")

    # Plot per scenario
    for scenario in sorted(df["scenario"].unique()):
        d = df[df["scenario"] == scenario].copy()

        for metric in METRICS:
            safe_boxplot(
                d,
                metric,
                f"outputs/plots/{scenario}_box_{metric}.png",
                title=f"{scenario}: {metric} distribution by method",
            )

        safe_bar_rates(
            d,
            f"outputs/plots/{scenario}_bar_success_collision.png",
            title=f"{scenario}: success & collision rate by method",
        )

    print("\nDone. Plots saved to outputs/plots/")


if __name__ == "__main__":
    main()

