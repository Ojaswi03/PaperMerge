
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
