import re

with open("benchmark_50epochs.log", "r") as f:
    lines = f.readlines()

report = [
    "# Milestone A5 Architecture Benchmark: CGCNN vs ALIGNN",
    "\n## Overview",
    "This benchmark evaluates ALIGNN against our baseline CGCNN.",
    "Both models were trained using 3 random seeds (42, 123, 456), 50 epochs, identical node/edge features, and identical early stopping constraints.",
    "\n## Tc Prediction Performance (3 Seeds, 50 Epochs)"
]

mae_rmse_lines = []
for line in lines:
    if line.startswith("| MAE |") or line.startswith("| RMSE |") or line.startswith("| Metric |") or line.startswith("|---|"):
        mae_rmse_lines.append(line.strip())

# The last 4 matches will be the final table
report.extend(mae_rmse_lines[-4:])

report.append("\n## Loss Curves (Seed 42)")

cgcnn_epochs = []
alignn_epochs = []

current_model = None
current_seed = None

for line in lines:
    if "=== Seed" in line:
        current_seed = int(re.search(r"Seed (\d+)", line).group(1))
    if "Training CGCNN" in line:
        current_model = "CGCNN"
    if "Training ALIGNN" in line:
        current_model = "ALIGNN"
        
    if current_seed == 42 and "Epoch" in line and "Train Loss" in line:
        m = re.search(r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)", line)
        if m:
            ep = int(m.group(1))
            t_loss = m.group(2)
            v_loss = m.group(3)
            if current_model == "CGCNN":
                cgcnn_epochs.append((ep, t_loss, v_loss))
            elif current_model == "ALIGNN":
                alignn_epochs.append((ep, t_loss, v_loss))

report.append("\n### CGCNN (Seed 42)")
report.append("| Epoch | Train Loss | Val Loss |")
report.append("|---|---|---|")
for ep, t, v in cgcnn_epochs:
    if ep % 5 == 0 or ep == cgcnn_epochs[-1][0]:
        report.append(f"| {ep} | {t} | {v} |")

report.append("\n### ALIGNN (Seed 42)")
report.append("| Epoch | Train Loss | Val Loss |")
report.append("|---|---|---|")
for ep, t, v in alignn_epochs:
    if ep % 5 == 0 or ep == alignn_epochs[-1][0]:
        report.append(f"| {ep} | {t} | {v} |")

report_str = "\n".join(report)
print(report_str)
with open("alignn_benchmark_report.md", "w", encoding="utf-8") as f:
    f.write(report_str)
