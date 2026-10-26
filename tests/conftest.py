"""Keep the suite offline: no test may shell out to the real MinerU install."""
import pytest


@pytest.fixture(autouse=True)
def no_mineru(monkeypatch):
    # `pdf_layout=auto` would otherwise find a local mineru venv and spend
    # minutes per PDF inside the suite. Layout tests stub this explicitly.
    monkeypatch.setattr('qaproject.layout.runner', lambda config: '')
