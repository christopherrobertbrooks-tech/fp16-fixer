"""Model locations come from environment variables; there are no default paths."""
import os
import sys


def need(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"set {name} to the model's directory (see the README's drivers note)")
    return v
