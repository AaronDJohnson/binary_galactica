"""Compile vendored GLASS C and regenerate the independent reference fixtures."""

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parents[1] / "tests" / "data" / "glass"
CHANNELS = ("X", "Y", "Z", "A", "E")


def cases():
    """Cover formats, dimensions, chirps, epochs, bandwidths and carrier edges."""
    result = []

    def add(
        name,
        f0,
        duration,
        bandwidth,
        fmt,
        nparams=8,
        epoch=0.0,
        fdot=0.0,
        fddot=0.0,
        carrier=None,
        source=(0.4, 2.1, 1e-21, 0.5, 0.7, 0.3),
    ):
        sin_beta, longitude, amplitude, cos_iota, polarization, phase = source
        params = [
            f0 * duration if carrier is None else carrier,
            sin_beta,
            longitude,
            float(np.log(amplitude)),
            cos_iota,
            polarization,
            phase,
            fdot * duration**2,
            fddot * duration**3,
        ][:nparams]
        result.append(
            {
                "id": name,
                "params": params,
                "T": duration,
                "t0": epoch,
                "BW": bandwidth,
                "fmt": fmt,
            }
        )

    for bandwidth in (32, 512):
        for fmt in ("sangria", "phase", "frequency"):
            add(f"1mhz_8p_bw{bandwidth}_{fmt}", 0.001, 7884000.0, bandwidth, fmt)
        for f0, fdot in ((0.005, 1e-17), (0.01, 6.5e-16)):
            add(
                f"{round(f0 * 1000)}mhz_8p_bw{bandwidth}_sangria",
                f0,
                31536000.0,
                bandwidth,
                "sangria",
                epoch=12345.0,
                fdot=fdot,
            )
    for fmt in ("sangria", "phase", "frequency"):
        add(f"5mhz_7p_bw32_{fmt}", 0.005, 31536000.0, 32, fmt, nparams=7, epoch=34567.0)
        add(
            f"10mhz_9p_bw512_{fmt}",
            0.01,
            31536000.0,
            512,
            fmt,
            nparams=9,
            epoch=12345.0,
            fdot=6.5e-16,
            fddot=1e-24,
        )
    # 31004/T*T = 31003.999999999996 in binary64: truncating params[0]
    # directly would incorrectly place the returned spectrum one bin higher.
    for name, carrier in (
        ("roundtrip_below_integer", 31004.0),
        ("just_above_integer", float(np.nextafter(31004.0, np.inf))),
    ):
        add(
            f"carrier_{name}",
            0.001,
            31536000.0,
            32,
            "sangria",
            epoch=23456.0,
            fdot=1e-18,
            carrier=carrier,
        )

    # Append to preserve the first 18 regression cases and their stable IDs.
    # These six fixed sources vary every orientation/amplitude/phase coordinate.
    # Each is run independently through C in every format, including the
    # non-power-of-two FFT and four-year cases absent from the original corpus.
    varied_sources = [
        {
            "name": "south_7p_quarter_year_bw32",
            "f0": 0.00173,
            "duration": 7_884_000.0,
            "bandwidth": 32,
            "nparams": 7,
            "source": (-0.72, 0.23, 2e-23, 0.86, 0.13, 5.61),
        },
        {
            "name": "north_8p_negative_chirp_year_bw96",
            "f0": 0.00421,
            "duration": 31_536_000.0,
            "bandwidth": 96,
            "nparams": 8,
            "epoch": 456_789.0,
            "fdot": -3e-17,
            "source": (0.61, 4.75, 3e-20, -0.08, 2.41, 1.82),
        },
        {
            "name": "equatorial_9p_four_years_bw512",
            "f0": 0.00827,
            "duration": 126_144_000.0,
            "bandwidth": 512,
            "nparams": 9,
            "epoch": 1_234_567.0,
            "fdot": 1.8e-16,
            "fddot": -2e-25,
            "source": (0.09, 3.21, 8e-22, -0.83, 1.77, 4.39),
        },
        {
            "name": "north_7p_four_years_bw96",
            "f0": 0.00237,
            "duration": 126_144_000.0,
            "bandwidth": 96,
            "nparams": 7,
            "epoch": 88_888.0,
            "source": (0.88, 5.94, 5e-23, -0.94, 2.88, 0.96),
        },
        {
            "name": "south_8p_positive_chirp_quarter_year_bw512",
            "f0": 0.00961,
            "duration": 7_884_000.0,
            "bandwidth": 512,
            "nparams": 8,
            "epoch": 34_567.0,
            "fdot": 4e-16,
            "source": (-0.44, 1.57, 1.2e-21, 0.04, 0.91, 3.17),
        },
        {
            "name": "oblique_9p_negative_chirp_year_bw32",
            "f0": 0.00343,
            "duration": 31_536_000.0,
            "bandwidth": 32,
            "nparams": 9,
            "epoch": 765_432.0,
            "fdot": -1e-17,
            "fddot": 6e-26,
            "source": (0.32, 4.13, 4e-20, 0.62, 0.37, 2.64),
        },
    ]
    for spec in varied_sources:
        name = spec["name"]
        values = {key: value for key, value in spec.items() if key != "name"}
        for fmt in ("sangria", "phase", "frequency"):
            add(f"varied_{name}_{fmt}", fmt=fmt, **values)
    return result


def generate(output, compiler):
    """Run C for every case; no Python waveform implementation is imported."""
    provenance = json.loads((ROOT / "sources.json").read_text())
    for item in provenance["upstream_files"]:
        if "vendored_path" in item:
            raw = (ROOT / item["vendored_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(f"Vendored source hash mismatch: {item['path']}")
    spectra = {}
    metadata = {
        "schema_version": 1,
        "repository": provenance["repository"],
        "commit": provenance["commit"],
        "channels": list(CHANNELS),
        "dtype": "complex128",
        "units": (
            "raw GLASS Fourier coefficients with the b0751bd half polarization "
            "factors, i.e. half the physical response for A_G = LDC Amplitude; "
            "continuous coefficients = raw * sqrt(T)"
        ),
        "normalization": "unchanged upstream invBW2 = 1/(2*BW)",
        "native_parameters": [
            "f0*T",
            "sin(beta)",
            "lambda",
            "log(A_G)",
            "cos(iota)",
            "psi",
            "phi0_G",
            "fdot*T^2",
            "fddot*T^3",
        ],
        "orbit": "GLASS analytic orbit; L=2.5e9 m; NI=3",
        "cases": cases(),
    }
    flags = ["-std=c99", "-D_DEFAULT_SOURCE", "-O2", "-ffp-contract=off"]
    metadata["compiler"] = subprocess.check_output(
        [*compiler, "--version"], text=True
    ).splitlines()[0]
    metadata["compiler_flags"] = flags
    source_paths = [
        "driver.c",
        "reference.c",
        "glass_constants.h",
        "vendor/kissfft/kiss_fft.c",
        "vendor/kissfft/kiss_fft.h",
        "vendor/kissfft/_kiss_fft_guts.h",
        "vendor/kissfft/kiss_fft_log.h",
    ]
    metadata["compiled_source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in source_paths
    }
    with tempfile.TemporaryDirectory(prefix="glass-reference-") as build:
        binary = Path(build) / "reference"
        subprocess.run(
            [
                *compiler,
                *flags,
                "-I",
                str(ROOT / "vendor" / "kissfft"),
                str(ROOT / "driver.c"),
                str(ROOT / "vendor" / "kissfft" / "kiss_fft.c"),
                "-lm",
                "-o",
                str(binary),
            ],
            check=True,
        )
        for case in metadata["cases"]:
            request = (
                f"{case['T']:.17g} {case['t0']:.17g} {case['BW']} "
                f"{len(case['params'])} {case['fmt']}\n"
                + " ".join(f"{value:.17g}" for value in case["params"])
                + "\n"
            )
            response = subprocess.check_output([str(binary)], input=request, text=True)
            header, body = response.split("\n", 1)
            case["q"], case["qmin"] = map(int, header.split())
            array = np.loadtxt(io.StringIO(body))
            if array.shape != (case["BW"], 11) or not np.isfinite(array).all():
                raise ValueError(f"Invalid C output: {case['id']}")
            spectra[f"{case['id']}_bins"] = array[:, 0].astype(np.int64)
            for index, channel in enumerate(CHANNELS):
                spectra[f"{case['id']}_{channel}"] = (
                    array[:, 1 + 2 * index] + 1j * array[:, 2 + 2 * index]
                )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reference.npz", **spectra)
    metadata["npz_sha256"] = hashlib.sha256(
        (output / "reference.npz").read_bytes()
    ).hexdigest()
    (output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Generated {len(metadata['cases'])} cases in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    generate(args.output, shlex.split(args.cc))
