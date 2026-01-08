#!/usr/bin/env python3
import sys
import os

# Ensure src is in path so we can import the package
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ai_subtitle_generator.cli import main

if __name__ == "__main__":
    main()
