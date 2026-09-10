import subprocess
import sys
from pathlib import Path

from indexa.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_check_ok(capsys):
    assert main(["check", str(EXAMPLES / "mlp.a")]) == 0
    assert "ok: module `MLP` type-checks" in capsys.readouterr().out


def test_compile_to_file(tmp_path, capsys):
    out = tmp_path / "mlp.py"
    assert main(["compile", str(EXAMPLES / "mlp.a"), "-o", str(out), "--dim", "B=2"]) == 0
    text = out.read_text()
    assert "def forward(" in text and "'B': 2" in text


def test_ir_dump(capsys):
    assert main(["ir", str(EXAMPLES / "mlp.a")]) == 0
    out = capsys.readouterr().out
    assert "sum[i#" in out and "def #" in out and "output probability;" in out


def test_errors_are_reported(tmp_path, capsys):
    bad = tmp_path / "bad.a"
    bad.write_text("module Bad { dim N = 2; space S = Fin(N); input x : Tensor<Real>[S]; def y[i : S] = x[j]; output y; }")
    assert main(["check", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "error[E0201]" in err and "1 previous error" in err
    assert main(["check", str(tmp_path / "missing.a")]) == 2


def test_module_entry_point():
    proc = subprocess.run([sys.executable, "-m", "indexa", "check", str(EXAMPLES / "attention.a")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Attention" in proc.stdout
