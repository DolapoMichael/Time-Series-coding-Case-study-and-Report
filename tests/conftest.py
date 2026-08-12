"""
Shared pytest configuration.

The pipeline scripts (part1_data_prep_eda.py, part2_problem_definition.py,
etc.) live at the repo root, not in an installable package, since the
project deliberately uses one consolidated script per assignment part
rather than the supplementary README's suggested nested src/ package (see
README.md for why). This conftest adds the repo root to sys.path so tests
can `import part2_problem_definition` etc. regardless of where pytest is
invoked from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
