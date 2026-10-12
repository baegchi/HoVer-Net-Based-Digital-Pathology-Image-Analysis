"""End-to-end smoke test of the Streamlit dashboard.

Uses Streamlit's own AppTest runner, so the real script is executed (including
model loading and analysis) without a browser.

    python -m pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = ROOT / "app" / "dashboard.py"
SAMPLE = ROOT / "samples" / "images" / "pannuke_00003_256x256.png"
CHECKPOINT = ROOT / "assets" / "checkpoints" / "hovernet_original_pannuke.pth"


def test_dashboard_renders_without_input():
    """With nothing uploaded the app should show its empty state, not crash."""
    at = AppTest.from_file(str(DASHBOARD), default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.title[0].value.startswith("Digital Pathology")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not downloaded")
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not downloaded")
def test_dashboard_analyses_a_sample_image():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=900)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # point the app at a bundled sample through the sidebar radio
    at.sidebar.radio[0].set_value("Sample images")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # `demo_placeholder.png` sorts first and has no nuclei, so choose a real
    # PanNuke patch explicitly.
    at.sidebar.selectbox[1].set_value(SAMPLE.name)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    metrics = [m.value for m in at.metric]
    assert any(m.isdigit() and int(m) > 0 for m in metrics), metrics
    # the report preview must be rendered from the live result
    assert any("Nuclei detected" in m.value for m in at.markdown), "report preview missing"