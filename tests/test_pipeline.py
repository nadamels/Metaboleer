import json
import os

from metaboleer import Config, run_pipeline
from metaboleer.example import make_example


def test_end_to_end(tmp_path):
    d = make_example(str(tmp_path / "data"), n=200)
    cfg = Config(data=f"{d}/data.csv", metadata=f"{d}/metadata.csv", annotation=f"{d}/annotation.csv",
                 outdir=str(tmp_path / "out"), imputer_trees=10, imputer_iter=1)
    res = run_pipeline(cfg, log=lambda *_: None)
    s = res["summary"]
    assert s["ggm_edges"] > 0 and s["pgain_strong_edges"] >= 1
    html = open(res["html"]).read()
    assert "__GRAPH_JSON__" not in html and "const GRAPH = {" in html
    assert os.path.exists(tmp_path / "out" / "09_export" / "cytoscape" / "network.cyjs")
    assert json.load(open(tmp_path / "out" / "summary.json"))["modules"] >= 4
