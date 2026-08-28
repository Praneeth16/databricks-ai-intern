#!/usr/bin/env python3
"""Assemble a percent-format source into the Kaggle notebook.

    python build.py [stem]

`stem` defaults to eda-baseline, and names both <stem>.src.py and <stem>.ipynb.
"""
import ast, builtins, json, pathlib, sys, uuid

HERE = pathlib.Path(__file__).resolve().parent
STEM = sys.argv[1] if len(sys.argv) > 1 else "eda-baseline"
src = (HERE / f"{STEM}.src.py").read_text()

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
out = HERE / f"{STEM}.ipynb"
json.dump(nb, out.open("w"), indent=1)

text = "\n".join("".join(c["source"]) for c in cells)
bad = sorted({ch for ch in text if ord(ch) > 127})
md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"cells={len(cells)} md={md} code={len(cells) - md} non_ascii={bad or 'none'}")
if bad:
    print(f"note: non-ascii kept (emoji/typography allowed): {bad}")

# A later cell that assigns a name a helper already uses turns the helper into a value, and
# the failure appears only when the helper is next called, which may be an hour into the run.
# `show = gold_auc.idxmax()` cost one full GPU re-run before this check existed.
HELPERS = {
    "show", "facts", "note", "quote", "short_uid", "_cell", "_esc", "display", "Markdown",
    "HTML", "log", "predict", "predict_top", "to_gpu", "batches", "safe_auc", "mean_auc",
    "fold", "rx", "find_file", "read_pixels", "read_header", "series_order", "slice_geometry",
    "build_study", "build_set", "write_submission", "score_against_gold", "bootstrap_auc",
    "pick_series", "augment", "build_model", "detect_language", "read_report",
    "contains_the_answers", "class_activation_map", "bucket",
}
seen, shadowed = set(), []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    for node in ast.parse("".join(c["source"])).body:      # module level only
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            names = []
        for n in names:
            if n in HELPERS and n in seen:
                shadowed.append(f"cell {i} rebinds {n}")
            seen.add(n)
assert not shadowed, "a later cell shadows a helper: " + "; ".join(shadowed)
print(f"no cell shadows any of the {len(HELPERS)} helper names")

# A name that is read but never bound anywhere is a NameError waiting in whichever cell runs
# last. The notebook cannot be executed off Kaggle, because the DICOM corpus is only mounted
# there, so this stands in for running it.
class _Names(ast.NodeVisitor):
    def __init__(self):
        self.bound, self.loaded = set(), set()

    def _bind(self, t):
        if isinstance(t, ast.Name):
            self.bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._bind(e)
        elif isinstance(t, ast.Starred):
            self._bind(t.value)

    def visit_Name(self, n):
        (self.bound if isinstance(n.ctx, (ast.Store, ast.Del)) else self.loaded).add(n.id)

    def visit_arg(self, n):
        self.bound.add(n.arg)
        self.generic_visit(n)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.bound.add(n.name)
        self.generic_visit(n)

    def visit_Global(self, n):
        self.bound.update(n.names)

    def visit_Nonlocal(self, n):
        self.bound.update(n.names)

    def visit_alias(self, n):
        self.bound.add((n.asname or n.name).split(".")[0])

    def visit_FunctionDef(self, n):
        self.bound.add(n.name)
        self.generic_visit(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.bound.add(n.name)
        self.generic_visit(n)

v = _Names()
for c in cells:
    if c["cell_type"] == "code":
        v.visit(ast.parse("".join(c["source"])))
unknown = sorted(v.loaded - v.bound - set(dir(builtins)))
assert not unknown, f"names read but never bound: {unknown}"
print(f"every one of the {len(v.loaded)} names read is bound somewhere or a builtin")

# A later cell that assigns a name a helper already uses turns the helper into a value, and
meta = json.loads((HERE / "kernel-metadata.json").read_text())
assert len(meta["title"]) <= 50, f"title is {len(meta['title'])} chars, Kaggle caps at 50"
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print(f"title={len(meta['title'])}/50 chars; all code cells parse")
