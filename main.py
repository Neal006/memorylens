"""Backward-compatible entry point. Prefer the `memorylens` CLI after `pip install memorylens-bench`."""

from memorylens.cli import main

if __name__ == "__main__":
    main()
