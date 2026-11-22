"""The dashboard must build, embed its data, and be syntactically valid JS."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gcsim.config import get_scenario  # noqa: E402
from gcsim.acceptance import run_acceptance  # noqa: E402
from gcsim.web.build import build_dashboard, PLACEHOLDER  # noqa: E402


@pytest.fixture(scope="module")
def acceptance_doc():
    return run_acceptance(["order-lifecycle"], do_tune=False, record_cap=16)


def test_dashboard_embeds_data_and_has_no_placeholder(acceptance_doc, tmp_path):
    out = tmp_path / "dashboard.html"
    build_dashboard(acceptance_doc, str(out))
    html = out.read_text(encoding="utf-8")
    assert PLACEHOLDER not in html
    assert "分代 GC" in html
    # the data document is embedded and parseable
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    data = json.loads(html[start:end].replace("<\\/", "</"))
    assert data["scenarios"][0]["name"] == "order-lifecycle"


def test_dashboard_js_is_syntactically_valid(acceptance_doc, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    out = tmp_path / "dashboard.html"
    build_dashboard(acceptance_doc, str(out))
    html = out.read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    js = tmp_path / "app.js"
    js.write_text(script, encoding="utf-8")
    proc = subprocess.run([node, "--check", str(js)], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr
