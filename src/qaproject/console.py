"""Keep child processes and console writes on one encoding.

The service runs on Chinese Windows machines where the ANSI code page is 936.
Without the "Beta: UTF-8" system option ``locale.getpreferredencoding()``
returns GBK, so any child-process pipe, console print or ``read_text()`` without
an explicit encoding decodes UTF-8 bytes as GBK: mojibake at best, and
``UnicodeDecodeError`` whenever the byte count is odd or a trail byte is
invalid. Every boundary that crosses this module uses UTF-8 explicitly.
"""
import os
import sys

ENV = {'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}


def child_env(base=None):
    """Environment for a child whose stdout/stderr this process decodes as UTF-8."""
    env = dict(os.environ if base is None else base)
    env.update(ENV)
    return env


def repair_streams():
    """Printing must never abort a run because the console cannot encode a glyph."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError, OSError):
            # Pytest capture objects and redirected files vary; never fail here.
            pass
