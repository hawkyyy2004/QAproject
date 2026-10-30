"""Local, evidence-grounded QA prototype."""

from .console import repair_streams

# Importing the package is enough to make printing safe on a Chinese Windows
# console: scientific symbols (NO3-, umol/kg) are not in GBK, and a strict
# stdout turns such a print into UnicodeEncodeError instead of a warning.
repair_streams()
