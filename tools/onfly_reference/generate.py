"""Build independent C response fixtures; no production Python code is imported.

Run ``python tools/onfly_reference/generate.py`` to rebuild from vendored C.
Pass ``--extract /path/to/ldasoft`` to re-extract the pinned original routines.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PIN = "079c6a3a6fcf380a7c107f39640fff7fc720d0f1"
FUNCTIONS = {
    "utils/src/glass_lisa.c": [
        "analytic_orbits",
        "initialize_analytic_orbit",
        "LISA_polarization_tensor",
        "hplus_and_hcross",
        "LISA_TDI_spline",
        "LISA_spline_response",
    ],
    "utils/src/glass_math.c": ["extract_amplitude_and_phase", "unwrap_phase"],
}


def extract(repository):
    """Extract original functions and record their pinned source hashes."""
    parts, manifest = [], {"commit": PIN, "files": [], "functions": []}
    for path, names in FUNCTIONS.items():
        source = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{PIN}:{path}"]
        )
        manifest["files"].append(
            {"path": path, "sha256": hashlib.sha256(source).hexdigest()}
        )
        source = source.decode()
        parts.append(source[: source.index("*/") + 2])
        for name in names:
            match = re.search(
                r"^(?:static )?void " + name + r"\([^\n]*\)\n\{", source, re.M
            )
            start = match.start()
            end, depth = source.index("{", start) + 1, 1
            while depth:
                depth += (source[end] == "{") - (source[end] == "}")
                end += 1
            block = source[start:end]
            parts.append(block)
            manifest["functions"].append(
                {
                    "name": name,
                    "path": path,
                    "line": source[:start].count("\n") + 1,
                    "sha256": hashlib.sha256(block.encode()).hexdigest(),
                }
            )
    (ROOT / "upstream.c").write_text("\n\n".join(parts) + "\n")
    (ROOT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def generate():
    """Compile the C harness and regenerate the varied source corpus."""
    cases = []
    with tempfile.TemporaryDirectory() as work:
        binary = Path(work) / "reference"
        subprocess.run(
            ["cc", "-O2", "-std=c99", str(ROOT / "driver.c"), "-lm", "-o", str(binary)],
            check=True,
        )
        sources = (
            (1e-4, -0.8, 0.2, 1e-22, 0.0, 0.1, 0.3, 2.0, 0.1),
            (0.001, 0.3, 2.1, 1.0, 0.4, 0.7, 0.9, 40.0, 4.0),
            (0.005, 0.96, 5.1, 1e-22, -0.9, 1.9, 2.5, 1300.0, 90.0),
            (0.03, -0.15, 3.8, 1.0, 1.0, 2.7, 5.7, -200.0, -20.0),
        )
        for source, n in ((source, n) for source in sources for n in (7, 8, 9)):
            duration = 31457280.0
            frequency, costh, lon, amplitude, cosi, psi, phi0, fdot, fddot = source
            params = [
                duration * frequency,
                costh,
                lon,
                float(np.log(amplitude)),
                cosi,
                psi,
                phi0,
                fdot,
                fddot,
            ][:n]
            times = np.linspace(0.03 * duration, 0.97 * duration, 12)
            text = (
                f"{duration:.17g} {n} {len(times)}\n"
                + " ".join(map(str, params))
                + "\n"
                + " ".join(map(str, times))
            )
            rows = np.array(
                [
                    list(map(float, line.split()))
                    for line in subprocess.check_output(
                        [str(binary)], input=text.encode()
                    )
                    .decode()
                    .splitlines()
                ]
            )
            cases.append(
                {
                    "T": duration,
                    "params": params,
                    "times": times.tolist(),
                    "positions": rows[:, :9]
                    .reshape(-1, 3, 3)
                    .transpose(1, 0, 2)
                    .tolist(),
                    "real": rows[:, 9::2].T.tolist(),
                    "imag": rows[:, 10::2].T.tolist(),
                }
            )
    target = ROOT.parents[1] / "tests/data/onfly/reference.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"commit": PIN, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.extract:
        extract(args.extract)
    generate()
