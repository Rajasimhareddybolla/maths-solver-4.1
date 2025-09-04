#!/usr/bin/env python3
"""Main entry point for the Math Solver package.

This allows the package to be run with: python -m math_solver
"""

import sys
from pathlib import Path

# Add the package to the path
package_root = Path(__file__).parent
sys.path.insert(0, str(package_root))

if __name__ == '__main__':
    from src.math_solver.cli import main
    sys.exit(main())
