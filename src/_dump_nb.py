import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    if i < 30:
        continue
    print(f"\n===== [{i}] {c['cell_type']} =====")
    print("".join(c["source"]))
