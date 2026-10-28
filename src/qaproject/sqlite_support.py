"""Platform-specific SQLite journal selection, applied before serving work."""
import os
import re
import sqlite3
import sys


WINDOWS = sys.platform == 'win32'


def journal_mode():
    # Per-machine override; never read configuration files on a DB hot path.
    value = os.environ.get('QA_SQLITE_JOURNAL_MODE', 'AUTO').strip().upper()
    if value == 'AUTO':
        return 'DELETE' if WINDOWS else 'WAL'
    if value not in {'WAL', 'DELETE'}:
        raise ValueError('QA_SQLITE_JOURNAL_MODE 仅接受 auto、WAL 或 DELETE，修改后重启。')
    return value


def initialize_journal(db, mode=None):
    """Select once outside transactions; never remove journal/SHM files by hand."""
    mode = journal_mode() if mode is None else mode
    if mode not in {'WAL', 'DELETE'}:
        raise ValueError('SQLite 日志模式无效。')
    actual = db.execute('PRAGMA journal_mode').fetchone()[0].upper()
    if actual != mode:
        actual = db.execute(f'PRAGMA journal_mode={mode}').fetchone()[0].upper()
    if actual != mode:
        raise ValueError('数据库日志模式切换未完成，请停止其他连接后重启。')
    return actual


def sqlite_diagnostics(error):
    """Keep native codes without copying SQL, paths or arbitrary error messages."""
    if not isinstance(error, sqlite3.Error):
        return {}
    result = {}
    code = getattr(error, 'sqlite_errorcode', None)
    name = getattr(error, 'sqlite_errorname', '')
    if type(code) is int:
        result['sqlite_errorcode'] = code
    if isinstance(name, str) and re.fullmatch(r'SQLITE_[A-Z0-9_]+', name):
        result['sqlite_errorname'] = name
    return result
