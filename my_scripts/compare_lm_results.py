import pandas as pd
import os

base_dir = os.path.join(os.path.dirname(__file__), "..", "results")
baseline_csv = os.path.join(base_dir, "basic_lm_benchmark_results_mixed_precision_warmups5.csv")
compile_csv = os.path.join(base_dir, "basic_lm_benchmark_results_mixed_precision_torch_compile_warmups5.csv")
output_md = os.path.join(base_dir, "basic_lm_comparison_mixed_precision_vs_compile.md")

df_base = pd.read_csv(baseline_csv)
df_comp = pd.read_csv(compile_csv)

# Merge on size
df = df_base[["size", "d_model", "num_layers", "num_heads",
              "fw_mean_ms", "bw_only_mean_ms", "fwbw_mean_ms"]].merge(
    df_comp[["size", "fw_mean_ms", "bw_only_mean_ms", "fwbw_mean_ms"]],
    on="size",
    suffixes=("_base", "_compile"),
)

# Compute speedup (baseline / compile, >1 means compile is faster)
df["fw_speedup"] = df["fw_mean_ms_base"] / df["fw_mean_ms_compile"]
df["bw_speedup"] = df["bw_only_mean_ms_base"] / df["bw_only_mean_ms_compile"]
df["fwbw_speedup"] = df["fwbw_mean_ms_base"] / df["fwbw_mean_ms_compile"]

# Build markdown
cols = [
    ("size", "size", "s"),
    ("d_model", "d_model", ".0f"),
    ("num_layers", "num_layers", ".0f"),
    ("num_heads", "num_heads", ".0f"),
    ("fw_mean_ms_base", "fw_base_ms", ".2f"),
    ("fw_mean_ms_compile", "fw_compile_ms", ".2f"),
    ("fw_speedup", "fw_speedup", ".2f"),
    ("bw_only_mean_ms_base", "bw_base_ms", ".2f"),
    ("bw_only_mean_ms_compile", "bw_compile_ms", ".2f"),
    ("bw_speedup", "bw_speedup", ".2f"),
    ("fwbw_mean_ms_base", "fwbw_base_ms", ".2f"),
    ("fwbw_mean_ms_compile", "fwbw_compile_ms", ".2f"),
    ("fwbw_speedup", "fwbw_speedup", ".2f"),
]

header = "| " + " | ".join(c[1] for c in cols) + " |"
sep = "|" + "|".join("-" * (len(c[1]) + 2) for c in cols) + "|"

rows = []
for _, r in df.iterrows():
    cells = []
    for col_key, _, fmt in cols:
        cells.append(f"{r[col_key]:{fmt}}")
    rows.append("| " + " | ".join(cells) + " |")

md = "# Basic LM Benchmark: Mixed Precision vs torch.compile\n\n"
md += header + "\n" + sep + "\n" + "\n".join(rows) + "\n"

with open(output_md, "w") as f:
    f.write(md)

print(f"Written to {output_md}")
print()
print(md)
