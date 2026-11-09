"""Entry point for ``python -m stratum``."""
import sys

from .server._cli import main


if __name__ == "__main__":
    sys.exit(main())
