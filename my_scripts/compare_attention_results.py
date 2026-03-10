import pandas as pd
import os

base_dir = os.path.join(os.path.dirname(__file__), "..", "results", "attention")
baseline_csv = os.path.join(base_dir, "attention_benchmark_results.csv")
compile_csv = os.path.join(base_dir, "attention_benchmark_results_torch_compile.csv")
output_md = os.path.join(base_dir, "attention_comparison.md")

df_base = pd.read_csv(baseline_csv)
df_comp = pd.read_csv(compile_csv)

# Filter out OOM rows
df_base = df_base[df_base["OOM"] == False].copy()
df_comp = df_comp[df_comp["OOM"] == False].copy()

# Merge on d_k and seq_len
df = df_base[["d_k", "seq_len", "fw_grad_mean_ms", "bw_mean_ms", "mem_with_grad_MB"]].merge(
    df_comp[["d_k", "seq_len", "fw_grad_mean_ms", "bw_mean_ms", "mem_with_grad_MB"]],
    on=["d_k", "seq_len"],
    suffixes=("_base", "_compile"),
)

# Compute speedup (baseline / compile, >1 means compile is faster)
df["fw_grad_speedup"] = df["fw_grad_mean_ms_base"] / df["fw_grad_mean_ms_compile"]
df["bw_speedup"] = df["bw_mean_ms_base"] / df["bw_mean_ms_compile"]
df["mem_ratio"] = df["mem_with_grad_MB_compile"] / df["mem_with_grad_MB_base"]

# Build markdown
cols = [
    ("d_k", "d_k", ".0f"),
    ("seq_len", "seq_len", ".0f"),
    ("fw_grad_mean_ms_base", "fw_grad_base", ".3f"),
    ("fw_grad_mean_ms_compile", "fw_grad_compile", ".3f"),
    ("fw_grad_speedup", "fw_grad_speedup", ".2f"),
    ("bw_mean_ms_base", "bw_base", ".3f"),
    ("bw_mean_ms_compile", "bw_compile", ".3f"),
    ("bw_speedup", "bw_speedup", ".2f"),
    ("mem_with_grad_MB_base", "mem_base_MB", ".2f"),
    ("mem_with_grad_MB_compile", "mem_compile_MB", ".2f"),
    ("mem_ratio", "mem_ratio", ".4f"),
]

header = "| " + " | ".join(c[1] for c in cols) + " |"
sep = "|" + "|".join("-" * (len(c[1]) + 2) for c in cols) + "|"

rows = []
for _, r in df.iterrows():
    cells = []
    for col_key, _, fmt in cols:
        cells.append(f"{r[col_key]:{fmt}}")
    rows.append("| " + " | ".join(cells) + " |")

md = header + "\n" + sep + "\n" + "\n".join(rows) + "\n"

with open(output_md, "w") as f:
    f.write(md)

print(f"Written to {output_md}")
print()
print(md)
