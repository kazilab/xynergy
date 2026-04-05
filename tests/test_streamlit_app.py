import subprocess
import sys
from pathlib import Path


def test_streamlit_app_prints_raw_mode_hint():
    root = Path(__file__).resolve().parents[1]
    app = root / "streamlit_app.py"

    result = subprocess.run(
        [sys.executable, str(app)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "streamlit run streamlit_app.py" in result.stdout
    assert "missing ScriptRunContext" not in result.stderr
