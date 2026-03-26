
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure GPU memory growth before any TF-importing modules load
from scripts.common import setupGpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
setupGpu()

from gui.experimentGui import main

if __name__ == "__main__":
    print("""
====================================================================
          BASIL + Noisy Channel Experiment GUI                               
                                                                           
   A graphical interface for configuring and running experiments             
====================================================================
Starting GUI...
    """)

    main()
