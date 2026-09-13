"""Extract the unmodified reference routines from a pinned GLASS checkout."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIN = "079c6a3a6fcf380a7c107f39640fff7fc720d0f1"
FUNCTIONS = {
    "utils/src/glass_lisa.c": [
        "analytic_orbits",
        "initialize_analytic_orbit",
        "recursive_phase_evolution",
        "double_angle",
        "triple_angle",
        "XYZ2AE",
        "LISA_tdi",
        "LISA_tdi_FF",
        "LISA_tdi_Sangria",
        "LISA_detector_tensor",
        "LISA_polarization_tensor",
    ],
    "utils/src/glass_math.c": ["glass_forward_complex_fft"],
    "ucb/src/glass_ucb_waveform.c": ["ucb_waveform"],
}


def extract(source):
    """Verify upstream file hashes, then extract complete functions verbatim."""
    manifest = json.loads((ROOT / "sources.json").read_text())
    for entry in manifest["upstream_files"]:
        raw = (source / entry["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Upstream hash mismatch: {entry['path']}")
    parts = [(ROOT / "preamble.h").read_text()]
    blocks = []
    for relative, names in FUNCTIONS.items():
        text = (source / relative).read_text()
        # Preserve each source's original copyright and Apache license notice.
        parts.append(text[: text.index("*/") + 2])
        for name in names:
            match = re.search(
                r"^(?:static )?void " + name + r"\([^\n]*\)\n\{", text, re.M
            )
            if match is None:
                raise ValueError(f"Missing function: {name}")
            start = match.start()
            end = text.index("{", start) + 1
            depth = 1
            while depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            block = text[start:end]
            parts.append(block)
            blocks.append(
                {
                    "file": relative,
                    "function": name,
                    "start_line": text[:start].count("\n") + 1,
                    "end_line": text[:end].count("\n") + 1,
                    "sha256": hashlib.sha256(block.encode()).hexdigest(),
                }
            )
    if blocks != manifest["extracted_functions"]:
        raise ValueError("Extracted routine provenance changed")
    (ROOT / "reference.c").write_text("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help=f"GLASS checkout at {PIN}")
    extract(parser.parse_args().upstream)
