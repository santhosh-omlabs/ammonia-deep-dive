"""The published CAS grid: eight records, ground near CASCI, excited above it.

No PySCF and no Qiskit. Reads data/projects/nh3-sto3g-cas/*.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAS = ROOT / "data" / "projects" / "nh3-sto3g-cas"

EXPECTED = {
    ("pyramidal_R0.900", 1),
    ("pyramidal_R1.012", 1),
    ("pyramidal_R1.120", 1),
    ("planar_R1.012", 1),
    ("pyramidal_R0.900", 3),
    ("pyramidal_R1.012", 3),
    ("pyramidal_R1.120", 3),
    ("planar_R1.012", 3),
}


def _records() -> list[dict]:
    paths = sorted(CAS.glob("NH3_*.json"))
    assert paths, f"missing records in {CAS}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_eight_cas_points_cover_the_grid() -> None:
    rows = _records()
    got = {
        (r["identity"]["geometry_tag"], int(r["identity"]["multiplicity"]))
        for r in rows
    }
    assert got == EXPECTED
    assert len(rows) == 8


def test_ground_vqd_sits_on_casci_and_excited_does_not() -> None:
    for rec in _records():
        energies = rec["energies"]
        vqd = energies["e_vqd"]
        roots = energies["e_casci_roots"]
        assert len(vqd) >= 2
        assert len(roots) >= 2
        assert abs(float(vqd[0]) - float(energies["e_casci"])) < 2e-3
        assert float(vqd[1]) > float(roots[1]) + 0.2
        assert rec["quantum"]["n_qubits"] == 12
        assert rec["quantum"]["device"] == "GPU"
