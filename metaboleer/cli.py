from __future__ import annotations

import argparse
import sys

from .config import Config


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="metaboleer", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--data", required=True, help="CSV/TSV: rows = samples, columns = metabolites (first column = sample id)")
    r.add_argument("--metadata", required=True, help="CSV/TSV: sample id + MetS, age, sex, fasting hours")
    r.add_argument("--outdir", default="metaboleer_out")
    r.add_argument("--annotation", help="optional feature table: feature, super_pathway (Xenobiotics), label")
    r.add_argument("--xenobiotics-file", help="optional text file with one xenobiotic feature per line")
    r.add_argument("--id-col")
    r.add_argument("--mets-col", default="MetS")
    r.add_argument("--age-col", default="age")
    r.add_argument("--sex-col", default="sex")
    r.add_argument("--fasting-col", default="fasting_hours")
    r.add_argument("--min-fasting-hours", type=float)
    r.add_argument("--adjust-fasting", action="store_true", help="also regress out fasting hours")
    r.add_argument("--max-missing", type=float, default=0.80)
    r.add_argument("--protect-trait", action="store_true")
    r.add_argument("--vip-cutoff", type=float, default=1.0)
    r.add_argument("--fdr-cutoff", type=float, default=0.05)
    r.add_argument("--network-type", default="signed", choices=["signed", "unsigned", "signed hybrid"])
    r.add_argument("--soft-power", type=int)
    r.add_argument("--min-module-size", type=int, default=10)
    r.add_argument("--deep-split", type=int, default=2, choices=range(5))
    r.add_argument("--merge-cut-height", type=float, default=0.25)
    r.add_argument("--module-alpha", type=float, default=0.05)
    r.add_argument("--modules", help="comma-separated modules to use for the GGM (skips auto selection)")
    r.add_argument("--ggm-alpha", type=float, default=0.01)
    r.add_argument("--pgain-alpha", type=float, default=0.05)
    r.add_argument("--pgain-scope", default="all", choices=["all", "ggm"])
    r.add_argument("--keep-all-pairs", action="store_true", help="write every pair in step 8b, not only p-gain >= 10")
    r.add_argument("--imputer-trees", type=int, default=100)
    r.add_argument("--imputer-iter", type=int, default=5)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--n-jobs", type=int, default=-1)
    r.add_argument("--title", default="WGCNA + GGM network")

    e = sub.add_parser("example", help="write a synthetic example dataset")
    e.add_argument("outdir", nargs="?", default="example_data")
    e.add_argument("--n", type=int, default=300)
    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "example":
        from .example import make_example
        make_example(args.outdir, n=args.n)
        print(f"wrote {args.outdir}/data.csv, metadata.csv, annotation.csv")
        return 0
    from .pipeline import run_pipeline
    kw = {k: v for k, v in vars(args).items() if k != "cmd"}
    if kw.get("modules"):
        kw["modules"] = [m.strip() for m in kw["modules"].split(",") if m.strip()]
    run_pipeline(Config(**kw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
