#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reporting.sequential_basil_plots import generate_comparison,generate_run_plots
from gui.research_protocol import PLOT_ROOT, RESULT_ROOT

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir",nargs="?"); parser.add_argument("--comparison",action="store_true")
    parser.add_argument("--plot-root",default=str(PLOT_ROOT));parser.add_argument("--result-root",default=str(RESULT_ROOT));args=parser.parse_args()
    if not args.comparison and not args.run_dir:parser.error("Supply a run directory or --comparison")
    print(json.dumps(generate_comparison(args.result_root,args.plot_root) if args.comparison else generate_run_plots(args.run_dir,args.plot_root),indent=2))
