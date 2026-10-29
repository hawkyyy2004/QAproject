"""A Chinese Windows console is GBK; no pipeline boundary may assume otherwise.

The colleague's report: a UTF-8 file read as GBK aborts the parse, the MinerU
probe dies before it starts, and every retry fails in the same few milliseconds
because the failure is cached.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qaproject import layout
from qaproject.console import child_env
from qaproject.store import load_json

ROOT = Path(__file__).resolve().parents[1]
# tests/conftest.py stubs layout.runner for the whole suite; keep the real one.
REAL_RUNNER = layout.runner


def probe_result(code):
    return subprocess.CompletedProcess([], code, '', '')


def test_child_env_pins_both_interpreter_switches():
    env = child_env({'PATH': 'x'})
    assert env['PYTHONIOENCODING'] == 'utf-8'
    assert env['PYTHONUTF8'] == '1'
    assert env['PATH'] == 'x'


def test_cached_answers_are_read_as_utf8_not_the_console_code_page(tmp_path):
    path = tmp_path / 'signature.json'
    path.write_text(json.dumps({'result': '已解析，质量合格℃'}, ensure_ascii=False), encoding='utf-8')
    # The exact failure: an odd byte tail that a GBK decoder cannot consume.
    with pytest.raises(UnicodeDecodeError):
        path.read_bytes().decode('gbk')
    assert load_json(path)['result'] == '已解析，质量合格℃'


def test_probe_pins_utf8_decoding_and_a_utf8_child_env(monkeypatch):
    seen = {}

    def fake(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        return probe_result(0)

    monkeypatch.setattr(layout.subprocess, 'run', fake)
    layout._RUNNER.clear()
    assert REAL_RUNNER({'mineru': {'python': 'mineru-python'}}) == 'mineru-python'
    assert seen['encoding'] == 'utf-8' and seen['errors'] == 'replace'
    assert seen['env']['PYTHONIOENCODING'] == 'utf-8' and seen['env']['PYTHONUTF8'] == '1'


def test_a_failed_probe_is_cached_and_reprobed_only_after_the_ttl(monkeypatch):
    calls = []

    def failing(cmd, **kwargs):
        calls.append(cmd)
        return probe_result(1)

    monkeypatch.setattr(layout.subprocess, 'run', failing)
    layout._RUNNER.clear()
    config = {'mineru': {'python': 'mineru-python'}}
    assert REAL_RUNNER(config) == ''
    assert REAL_RUNNER(config) == ''
    assert len(calls) == 1, 'a retry must not re-probe inside the TTL'
    # Installing MinerU must take effect without a service restart.
    layout._RUNNER['mineru-python'] = (False, time.monotonic() - layout.PROBE_RETRY_SECONDS - 1)
    assert REAL_RUNNER(config) == ''
    assert len(calls) == 2


def test_a_missing_interpreter_is_unavailable_instead_of_fatal(tmp_path, monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(layout.subprocess, 'run', missing)
    layout._RUNNER.clear()
    assert REAL_RUNNER({'mineru': {'python': str(tmp_path / 'nope.exe')}}) == ''


def test_undecodable_probe_output_cannot_abort_the_parse(monkeypatch):
    # GBK cannot decode this text; before the fix the probe raised here and the
    # document failed instantly on every retry.
    def garble(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, '已解析'.encode('utf-8').decode('utf-8'), '')

    monkeypatch.setattr(layout.subprocess, 'run', garble)
    layout._RUNNER.clear()
    assert REAL_RUNNER({'mineru': {'python': 'mineru-python'}}) == ''


def test_script_progress_survives_a_gbk_console():
    code = (
        'import json,sys\n'
        f'sys.path.insert(0, {str(ROOT / "src")!r})\n'
        'from qaproject.console import repair_streams\n'
        'repair_streams()\n'
        "print(json.dumps({'caption': 'NO\\u2083\\u207b \\u00b5mol kg\\u207b\\u00b9'}, ensure_ascii=False), flush=True)\n"
    )
    done = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          env=dict(child_env({'PATH': str(Path(sys.executable).parent)}), PYTHONIOENCODING='gbk'))
    assert done.returncode == 0, done.stderr[-400:]
