"""
pytest bootstrap: make `software/` importable before any test module loads.

The source modules do `sys.path.insert(0, parents[1])` at import time, which
works when they are run as scripts but not when pytest imports them. Importing
`_harness` here does the same thing once, for the whole session.
"""

import _harness  # noqa: F401  (imported for its sys.path side effect)
