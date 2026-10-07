"""Compatibility import for the pre-workspace GUI module."""

from gui.app import PaperMergeApp, main

ExperimentGUI = PaperMergeApp

__all__ = ["ExperimentGUI", "PaperMergeApp", "main"]


if __name__ == "__main__":
    main()
