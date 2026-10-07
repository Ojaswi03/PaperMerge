#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.research_protocol import CONFIG_ROOT, write_config_library

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Prepare paired configs without launching training")
    parser.add_argument("--output", type=Path, default=CONFIG_ROOT)
    args=parser.parse_args()
    production,smoke=write_config_library(args.output)
    print(f"Wrote {production} production and {smoke} smoke configurations.")
