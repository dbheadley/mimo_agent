"""
Entry point for the mimo_pack.util.scaffold module.

This allows the scaffold utility to be run as a module:
    python -m mimo_pack.util.scaffold /path/to/session
"""

from .scaffold import main

if __name__ == '__main__':
    main()