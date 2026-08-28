#!/usr/bin/env python3
"""Assemble a percent-format source into the Kaggle notebook.

    python build.py [stem]

`stem` defaults to model-v2, and names both <stem>.src.py and <stem>.ipynb. This is a copy of
the parent build.py: same assertions, different default stem and a helper set that names this
notebook's helpers. `HERE` resolves to this copy's directory, so it reads the metadata and
writes the ipynb beside it and the parent notebook's workflow is untouched.
"""
import ast, builtins, json, pathlib, sys, uuid

HERE = pathlib.Path(__file__).resolve().parent
STEM = sys.argv[1] if len(sys.argv) > 1 else "model-v2"
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
    "show", "facts", "note", "short_uid", "_cell", "_esc", "display", "Markdown",
    "HTML", "log", "safe_auc", "mean_auc", "fold", "find_file", "write_submission",
    "score_against_gold", "bootstrap_auc", "contains_the_answers", "bucket",
    # this notebook's own helpers, on top of the shared baseline set
    "build_backbone", "RaptorClassifier", "load_model", "find_weight_file", "find_test_root",
    "eval_windows", "window_centers", "infer_study", "infer_probs", "rankpct", "get_stack",
    "gate_auc", "blend_ranks", "_eval_centers",
    "_pick_series_for_slot", "_make_reader",
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

# A name being bound is not the same as the call matching the signature. `infer_probs(model,
# x, dev)` was defined with three parameters and called with two; the NameError check above
# is happy, the notebook runs for 51 minutes of decoding, and the crash lands in the cell
# after. Positional counts are checked exactly; keyword and default handling is left to Python.
sigs = {}
for c in cells:
    if c["cell_type"] != "code":
        continue
    tree = ast.parse("".join(c["source"]))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_pos = len(node.args.posonlyargs) + len(node.args.args) - len(node.args.defaults)
            n_max = len(node.args.posonlyargs) + len(node.args.args) + (
                1 if node.args.vararg else 0)
            params = {a.arg for a in node.args.posonlyargs + node.args.args +
                      node.args.kwonlyargs}
            sigs[node.name] = (n_pos, n_max, params, bool(node.args.kwarg))
bad_calls = []
for c in cells:
    if c["cell_type"] != "code":
        continue
    tree = ast.parse("".join(c["source"]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in sigs:
            lo, hi, params, has_kwargs = sigs[node.func.id]
            n_pos = len(node.args) - sum(1 for a in node.args if isinstance(a, ast.Starred))
            if n_pos < lo and not node.keywords:
                bad_calls.append(f"{node.func.id} called with {n_pos} positional args, "
                                 f"signature needs at least {lo}")
            if n_pos > hi:
                bad_calls.append(f"{node.func.id} called with {n_pos} positional args, "
                                 f"signature takes at most {hi}")
            if not has_kwargs:
                unknown_kw = [kw.arg for kw in node.keywords
                              if kw.arg is not None and kw.arg not in params]
                if unknown_kw:
                    bad_calls.append(f"{node.func.id} called with unknown keyword args "
                                     f"{unknown_kw}")
assert not bad_calls, "signature mismatches: " + "; ".join(bad_calls)
print(f"every call to a notebook-defined function matches its signature")

# A later cell that assigns a name a helper already uses turns the helper into a value, and
meta = json.loads((HERE / "kernel-metadata.json").read_text())
assert len(meta["title"]) <= 50, f"title is {len(meta['title'])} chars, Kaggle caps at 50"
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print(f"title={len(meta['title'])}/50 chars; all code cells parse")
