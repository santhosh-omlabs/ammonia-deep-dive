"""NH3 geometries and runs (STO-3G). N at the origin, coordinates in Angstrom.

  python main.py all      geometries x spins x CASCI keys x VQD k=2 (CAS dataset)
  python main.py sweep    same grid, CASCI + HF only (fast)
  python main.py vqe      one pyramidal singlet, ground VQE
  python main.py vqd      one pyramidal singlet, VQD k=2
  python main.py doctor   Aer devices / GPU probe (no chemistry)
  python main.py bench    short CPU then GPU VQE on one geometry
"""

from __future__ import annotations

import math
import os
import sys
import traceback

from gpu_simulator import diagnose, print_runpod_needed
from mol_simulate import simulate_molecule

Atom = tuple[str, tuple[float, float, float]]

R_EQ = 1.012
ANGLE_EQ = 106.7
PROJECT = "nh3-sto3g-cas"
# CPU unless AMMONIA_USE_GPU=1. The published records were produced with that
# flag on a RunPod GPU. A clone should be able to run `sweep` without a GPU.
USE_GPU = os.environ.get("AMMONIA_USE_GPU", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def pyramidal(r_nh: float, angle_hnh_deg: float) -> list[Atom]:
    alpha = math.radians(angle_hnh_deg)
    cos_phi_sq = (1.0 + 2.0 * math.cos(alpha)) / 3.0
    if cos_phi_sq <= 0.0 or cos_phi_sq >= 1.0:
        raise RuntimeError("H-N-H angle does not give a valid pyramid.")
    cos_phi = math.sqrt(cos_phi_sq)
    sin_phi = math.sqrt(1.0 - cos_phi_sq)
    hydrogens = []
    for k in range(3):
        az = 2.0 * math.pi * k / 3.0
        hydrogens.append(
            (
                r_nh * sin_phi * math.cos(az),
                r_nh * sin_phi * math.sin(az),
                r_nh * cos_phi,
            )
        )
    return [
        ("N", (0.0, 0.0, 0.0)),
        ("H", hydrogens[0]),
        ("H", hydrogens[1]),
        ("H", hydrogens[2]),
    ]


def planar(r_nh: float) -> list[Atom]:
    hydrogens = []
    for k in range(3):
        az = 2.0 * math.pi * k / 3.0
        hydrogens.append((r_nh * math.cos(az), r_nh * math.sin(az), 0.0))
    return [
        ("N", (0.0, 0.0, 0.0)),
        ("H", hydrogens[0]),
        ("H", hydrogens[1]),
        ("H", hydrogens[2]),
    ]


NH3 = pyramidal(R_EQ, ANGLE_EQ)


def jobs() -> list[dict]:
    out = []
    for r_nh in (0.90, R_EQ, 1.12):
        out.append(
            {
                "tag": f"pyramidal_R{r_nh:.3f}",
                "atoms": pyramidal(r_nh, ANGLE_EQ),
            }
        )
    out.append({"tag": f"planar_R{R_EQ:.3f}", "atoms": planar(R_EQ)})
    return out


def _run_point(
    atoms,
    *,
    multiplicity: int,
    geometry_tag: str,
    ansatz: str,
    quantum_roots: int,
    shots: int,
    use_gpu: bool = USE_GPU,
    cap_evals: bool = False,
    write_record: bool = True,
    maxiter: int = 40,
    active: str = "auto",
    max_spatial: int = 6,
    project: str = PROJECT,
    vqd_beta: float = 5.0,
    optimizer: str = "COBYLA",
    n_roots: int = 6,
) -> dict:
    return simulate_molecule(
        atoms,
        name="NH3",
        charge=0,
        multiplicity=multiplicity,
        basis="sto3g",
        unit="Angstrom",
        mapper="bk",
        ansatz=ansatz,
        optimizer=optimizer,
        maxiter=maxiter,
        shots=shots,
        active=active,
        max_spatial=max_spatial,
        n_roots=n_roots,
        quantum_roots=quantum_roots,
        vqd_beta=vqd_beta,
        write_record=write_record,
        project=project,
        geometry_tag=geometry_tag,
        use_gpu=use_gpu,
        cap_evals=cap_evals,
    )


def run_sweep() -> None:
    for geom in jobs():
        for multiplicity in (1, 3):
            print(
                f"\n==== {geom['tag']}  multiplicity={multiplicity} ====",
                flush=True,
            )
            _run_point(
                geom["atoms"],
                multiplicity=multiplicity,
                geometry_tag=geom["tag"],
                ansatz="hf",
                quantum_roots=1,
                shots=0,
                use_gpu=False,
            )


def run_all() -> None:
    """Full stated dataset: each geometry and spin gets CASCI keys + VQD k=2."""
    print(
        "Dataset: 4 geometries x 2 spins x CASCI n_roots=6 x VQD k=2 (UCC-D). "
        "Each point is two long COBYLA hunts. This can take many hours.",
        flush=True,
    )
    for geom in jobs():
        for multiplicity in (1, 3):
            print(
                f"\n==== VQD  {geom['tag']}  multiplicity={multiplicity} ====",
                flush=True,
            )
            try:
                _run_point(
                    geom["atoms"],
                    multiplicity=multiplicity,
                    geometry_tag=geom["tag"],
                    ansatz="ucc-d",
                    quantum_roots=2,
                    shots=4096,
                )
            except Exception as exc:
                print(
                    f"     POINT FAILED {geom['tag']} m={multiplicity}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                continue


def run_vqe() -> None:
    _run_point(
        NH3,
        multiplicity=1,
        geometry_tag="pyramidal_R1.012",
        ansatz="ucc-d",
        quantum_roots=1,
        shots=4096,
    )


def run_vqd() -> None:
    _run_point(
        NH3,
        multiplicity=1,
        geometry_tag="pyramidal_R1.012",
        ansatz="ucc-d",
        quantum_roots=2,
        shots=4096,
    )


def run_bench() -> None:
    """Short same-Hamiltonian VQE on CPU then GPU. Not the full COBYLA budget."""
    print(
        "Bench: pyramidal R=1.012 singlet, UCC-D, 25 COBYLA evals, shots=0. "
        "Same CAS/mapper/ansatz as the dataset; fewer optimizer steps.",
        flush=True,
    )
    rows: list[tuple[str, dict | None, str | None]] = []
    for gpu in (False, True):
        label = "GPU" if gpu else "CPU"
        print(f"\n==== bench {label} ====", flush=True)
        try:
            out = _run_point(
                NH3,
                multiplicity=1,
                geometry_tag="bench_pyramidal_R1.012",
                ansatz="ucc-d",
                quantum_roots=1,
                shots=0,
                use_gpu=gpu,
                cap_evals=True,
                write_record=False,
                maxiter=25,
            )
            rows.append((label, out, None))
        except Exception as exc:
            print(
                f"     bench {label} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            rows.append((label, None, f"{type(exc).__name__}: {exc}"))
    print("\n==== bench summary ====", flush=True)
    by = {name: (data, err) for name, data, err in rows}
    cpu, cpu_err = by.get("CPU", (None, "missing"))
    gpu, gpu_err = by.get("GPU", (None, "missing"))
    if cpu:
        print(
            f"     CPU  qubits={cpu['n_qubits']}  nfev={cpu['nfev']}  "
            f"E={cpu['e_vqe']:+.8f}  wall={cpu['runtime_s']:.2f}s"
        )
    else:
        print(f"     CPU  failed  {cpu_err}")
    if gpu:
        print(
            f"     GPU  qubits={gpu['n_qubits']}  nfev={gpu['nfev']}  "
            f"E={gpu['e_vqe']:+.8f}  wall={gpu['runtime_s']:.2f}s"
        )
    else:
        print(f"     GPU  failed  {gpu_err}")
    if cpu and gpu and cpu.get("e_vqe") is not None and gpu.get("e_vqe") is not None:
        de = abs(float(gpu["e_vqe"]) - float(cpu["e_vqe"]))
        print(f"     |E_GPU - E_CPU| = {de:.3e} hartree")
        if cpu["runtime_s"] > 0:
            print(f"     wall GPU/CPU     = {gpu['runtime_s'] / cpu['runtime_s']:.3f}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode in ("all", "dataset"):
        run_all()
    elif mode in ("vqe", "ucc", "ucc-d"):
        run_vqe()
    elif mode in ("vqd",):
        run_vqd()
    elif mode in ("sweep",):
        run_sweep()
    elif mode in ("doctor", "gpu"):
        diagnose(probe_gpu=True)
        print_runpod_needed()
    elif mode in ("bench",):
        run_bench()
    else:
        raise SystemExit(
            "usage: python main.py [all|sweep|vqe|vqd|doctor|bench]"
        )
