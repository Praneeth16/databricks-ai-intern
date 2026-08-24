#!/usr/bin/env python3
"""Assemble the percent-format sources into the Kaggle notebook."""
import json, pathlib, uuid

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "lookup-transformer.src.py").read_text()

cells, cur, kind = [], [], None
def flush():
    if kind is None: return
    body = "\n".join(cur).strip("\n")
    if not body.strip(): return
    cid = uuid.uuid4().hex[:8]
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l)
                          for l in body.split("\n"))
        cells.append({"cell_type": "markdown", "id": cid, "metadata": {},
                      "source": text.split("\n")})
    else:
        cells.append({"cell_type": "code", "id": cid, "metadata": {},
                      "execution_count": None, "outputs": [], "source": body.split("\n")})

for line in src.split("\n"):
    if line.startswith("# %% [markdown]"): flush(); cur, kind = [], "markdown"; continue
    if line.startswith("# %%"): flush(); cur, kind = [], "code"; continue
    cur.append(line)
flush()
for c in cells:
    b = c["source"]; c["source"] = [l + "\n" for l in b[:-1]] + [b[-1]]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11.0"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "lookup-transformer.ipynb"
json.dump(nb, out.open("w"), indent=1)

text = "\n".join("".join(c["source"]) for c in cells)
bad = sorted({ch for ch in text if ord(ch) > 127})
md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"cells={len(cells)} md={md} code={len(cells) - md} non_ascii={bad or 'none'}")
assert not bad, f"non-ascii characters present: {bad}"
