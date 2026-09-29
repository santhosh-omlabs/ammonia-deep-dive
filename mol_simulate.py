"""Electronic structure for one geometry: SCF, automatic CAS, qubit H, UCC-D VQE, Aer shots.

Core MOs (atomic 1s on Z>=3, etc.) are frozen. The active space is the valence
occupied set plus the lowest virtuals ('higher' MOs), capped so the qubit
problem stays finite. VQE and Aer target CASCI in that space, not full FCI.

PySCF: WSL micromamba env `flibe` on this lab PC. No sibling Quantum imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQD, VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

from gpu_simulator import make_session

Atom = tuple[str, tuple[float, float, float]]

_UCC = {"ucc-s": "s", "ucc-d": "d", "ucc-sd": "sd"}
_SCHEMA = "ammonia-deep-dive.record.v1"
_REPO = Path(__file__).resolve().parent
_PROJECT_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _reexec_under_wsl() -> int:
    repo = Path(sys.argv[0]).resolve().parent
    script = Path(sys.argv[0]).name
    inner = f"$HOME/.local/bin/micromamba run -n flibe python {script}"
    cmd = ["wsl", "--cd", str(repo), "-e", "bash", "-lc", inner]
    print("PySCF missing in this Python. Re-running under WSL env flibe.")
    try:
        return int(subprocess.call(cmd))
    except FileNotFoundError:
        print("wsl.exe not found. Install pyscf (conda-forge) and retry.")
        return 1


def _load_pyscf():
    try:
        from pyscf import ao2mo, fci, gto, mcscf, scf
    except ImportError:
        if os.name == "nt":
            raise SystemExit(_reexec_under_wsl())
        print("PySCF is not installed in this Python.")
        raise SystemExit(1)
    return ao2mo, fci, gto, mcscf, scf


def _step(msg: str) -> None:
    print(f"[ok] {msg}", flush=True)


def _mapper(name: str):
    key = name.lower().strip()
    if key in ("jw", "jordan-wigner", "jordan_wigner"):
        return JordanWignerMapper()
    if key in ("bk", "bravyi-kitaev", "bravyi_kitaev"):
        return BravyiKitaevMapper()
    raise ValueError(f"Unknown mapper {name!r}. Use 'jw' or 'bk'.")


def _optimizer(
    name: str,
    maxiter: int,
    n_par: int,
    ansatz_id: str,
    *,
    cap_evals: bool = False,
):
    """Budget and step size follow the ansatz, not a fixed laptop guess.

    UCC amplitudes are O(0.01–0.1). COBYLA's default rhobeg=1 jumps off
    Hartree–Fock into a high-energy region, then n_par+2 evals cannot recover.
    cap_evals: honor maxiter as a hard cap (CPU/GPU bench only).
    """
    key = name.upper().replace("-", "_").strip()
    n_par = max(int(n_par), 0)
    floor = max(int(maxiter), n_par + 2)
    # Derivative-free search needs several probes per angle.
    if n_par >= 8 and not cap_evals:
        floor = max(floor, 8 * n_par)
    if cap_evals:
        floor = max(int(maxiter), 1)
    cap = 8000 if n_par >= 80 else 4000
    floor = min(floor, cap)
    rhobeg = 0.05 if ansatz_id.startswith("ucc") else 0.25

    if key == "COBYLA":
        print(f"     COBYLA  maxfun={floor}  rhobeg={rhobeg}")
        return COBYLA(maxiter=floor, rhobeg=rhobeg, tol=1e-8)
    if key == "SLSQP":
        it = max(int(maxiter), min(200, max(25, 3 * n_par)))
        print(f"     SLSQP   maxiter={it}")
        return SLSQP(maxiter=it, ftol=1e-10)
    if key in ("L_BFGS_B", "LBFGSB"):
        fun = max(int(maxiter), 15 * n_par + 30)
        fun = min(fun, 4000)
        print(f"     L_BFGS_B  maxfun={fun}")
        return L_BFGS_B(maxfun=fun, maxiter=max(80, 2 * n_par), ftol=1e-10)
    raise ValueError(
        f"Unknown optimizer {name!r}. Use COBYLA, SLSQP, or L_BFGS_B."
    )


def _vqd_callback():
    best: dict[int, float] = {}

    def _cb(*args) -> None:
        n_eval = int(args[0])
        energy = float(np.real(args[2]))
        step = int(args[4]) if len(args) > 4 else 1
        prev = best.get(step)
        improved = prev is None or energy < prev - 1e-6
        if improved:
            best[step] = energy
        if n_eval == 1 or n_eval % 20 == 0 or improved:
            print(
                f"     k={step}  nfev={n_eval:4d}  E={energy:+.8f}",
                flush=True,
            )

    return _cb


def _atom_string(atoms: Sequence[Atom]) -> str:
    parts = []
    for symbol, xyz in atoms:
        x, y, z = xyz
        parts.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f}")
    return "; ".join(parts)


def _n_core_spatial(mol) -> int:
    """Closed-shell cores in a minimal picture: 1s for Li-Ne, 1s2s2p for Na-Ar."""
    n_core = 0
    for z in mol.atom_charges():
        z = int(z)
        if z >= 11:
            n_core += 5
        elif z >= 3:
            n_core += 1
    return n_core


def _choose_cas(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
    n_core: int,
    max_spatial: int,
) -> tuple[int, int, int, int]:
    """Energy-ordered window: freeze core, keep HOMO side + lowest virtuals."""
    n_occ = max(n_alpha, n_beta)
    n_docc = min(n_alpha, n_beta)
    n_core = min(max(n_core, 0), n_docc)
    n_occ_cas = n_occ - n_core
    n_virt_cas = n_spatial - n_occ
    ncas = n_occ_cas + n_virt_cas
    if ncas > max_spatial:
        extra = ncas - max_spatial
        drop_v = min(extra, n_virt_cas)
        n_virt_cas -= drop_v
        extra -= drop_v
        if extra > 0:
            n_occ_cas -= extra
            n_core += extra
        ncas = n_occ_cas + n_virt_cas
    if ncas < 1 or n_occ_cas < 0:
        raise ValueError("Could not build a non-empty active space.")
    n_alpha_cas = n_alpha - n_core
    n_beta_cas = n_beta - n_core
    if n_alpha_cas < 0 or n_beta_cas < 0 or n_alpha_cas + n_beta_cas < 1:
        raise ValueError("Active electron count is invalid.")
    return n_core, ncas, n_alpha_cas, n_beta_cas


def _print_mo_partition(mo_energy, n_core: int, ncas: int) -> None:
    n_spatial = len(mo_energy)
    print("     MO   energy/h     role")
    for i, e in enumerate(mo_energy):
        if i < n_core:
            role = "frozen core"
        elif i < n_core + ncas:
            role = "active"
        else:
            role = "discarded virt"
        print(f"     {i:3d}  {float(e):+10.4f}     {role}")
    print(f"     (spatial {n_spatial}; active {ncas} -> {2 * ncas} qubits)")


def _statevector_energy(circuit, hamiltonian: SparsePauliOp) -> float:
    psi = Statevector.from_instruction(circuit.decompose(reps=8))
    return float(np.real(psi.expectation_value(hamiltonian)))


def _as_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        arr = np.asarray(val).reshape(-1)
        if arr.size == 0:
            return default
        return int(arr[-1])
    except (TypeError, ValueError):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default


def _as_float(val, default: float = float("nan")) -> float:
    if val is None:
        return default
    try:
        arr = np.asarray(val, dtype=float).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[-1])
    except (TypeError, ValueError):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default


def _unpack_thetas(result, n_par: int, k: int) -> list[np.ndarray]:
    raw = getattr(result, "optimal_points", None)
    if raw is None:
        raw = getattr(result, "optimal_point", None)
    if raw is None:
        return []
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        if k > 1 and n_par > 0 and arr.size == k * n_par:
            arr = arr.reshape(k, n_par)
        else:
            return [arr]
    return [np.asarray(row, dtype=float).reshape(-1) for row in arr]


def _opt_status(result) -> dict[str, Any]:
    out: dict[str, Any] = {}
    nfev = getattr(result, "cost_function_evals", None)
    if nfev is not None:
        out["cost_function_evals"] = _as_int(nfev)
    ot = getattr(result, "optimizer_time", None)
    if ot is not None:
        out["optimizer_time"] = _as_float(ot)
    opt_res = getattr(result, "optimizer_result", None)
    if opt_res is not None:
        msg = getattr(opt_res, "message", None)
        if msg is None:
            msg = getattr(opt_res, "status", None)
        if msg is not None:
            out["message"] = str(msg)
        nfev2 = getattr(opt_res, "nfev", None)
        if nfev2 is not None:
            out["nfev"] = _as_int(nfev2)
    return out


def _make_ansatz(
    kind: str,
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
    mapper,
    n_qubits: int,
):
    hf = HartreeFock(
        num_spatial_orbitals=n_spatial,
        num_particles=(n_alpha, n_beta),
        qubit_mapper=mapper,
    )
    key = kind.lower().strip()
    if key == "hf":
        return hf, "hf"
    if key in _UCC:
        ucc = UCC(
            num_spatial_orbitals=n_spatial,
            num_particles=(n_alpha, n_beta),
            excitations=_UCC[key],
            qubit_mapper=mapper,
            initial_state=hf,
        )
        return ucc, key
    if key == "hea":
        body = efficient_su2(
            n_qubits, reps=1, su2_gates=["ry"], entanglement="linear"
        )
        return hf.compose(body), "hea"
    raise ValueError(
        f"Unknown ansatz {kind!r}. Use hf, ucc-s, ucc-d, ucc-sd, or hea."
    )


def _pkg_ver(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _project_dir(project: str) -> Path:
    name = project.strip()
    if not _PROJECT_OK.match(name):
        raise ValueError(
            "project must be a folder name (letters, digits, . _ -), not a path."
        )
    path = _REPO / "data" / "projects" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mo_roles(n_spatial: int, n_core: int, ncas: int) -> list[str]:
    roles = []
    for i in range(n_spatial):
        if i < n_core:
            roles.append("frozen core")
        elif i < n_core + ncas:
            roles.append("active")
        else:
            roles.append("discarded virt")
    return roles


def _cas_rdm1(fcisolver, ci, ncas: int, n_elec_cas) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    rdm1 = np.asarray(fcisolver.make_rdm1(ci, ncas, n_elec_cas), dtype=float)
    rdm1_a = rdm1_b = None
    make_s = getattr(fcisolver, "make_rdm1s", None)
    if make_s is not None:
        pair = make_s(ci, ncas, n_elec_cas)
        rdm1_a = np.asarray(pair[0], dtype=float)
        rdm1_b = np.asarray(pair[1], dtype=float)
    return rdm1, rdm1_a, rdm1_b


_AU_TO_DEBYE = 2.541746285


def _ci_list(ci, n_roots: int) -> list:
    if n_roots <= 1:
        return [ci]
    if isinstance(ci, (list, tuple)):
        return list(ci)[:n_roots]
    return [ci]


def _total_energies(e_tot, e_cas) -> list[float]:
    tot = np.atleast_1d(np.asarray(e_tot, dtype=float)).ravel()
    cas = np.atleast_1d(np.asarray(e_cas, dtype=float)).ravel()
    if tot.size == cas.size:
        return [float(x) for x in tot]
    e0 = float(tot[0])
    e0_cas = float(cas[0])
    return [e0 + float(ek) - e0_cas for ek in cas]


def _dipole_au(mol, dm_ao: np.ndarray) -> np.ndarray:
    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    mu_n = charges @ coords
    dip_ao = np.asarray(mol.intor("int1e_r", comp=3), dtype=float)
    mu_e = -np.einsum("xij,ij->x", dip_ao, dm_ao)
    return np.asarray(mu_n + mu_e, dtype=float)


def _ao_rdm1(C, cas_obj, fcisolver, ci, ncas: int, nelecas) -> np.ndarray:
    if cas_obj is not None:
        return np.asarray(cas_obj.make_rdm1(ci=ci), dtype=float)
    dm_mo = np.asarray(fcisolver.make_rdm1(ci, ncas, nelecas), dtype=float)
    return np.asarray(C @ dm_mo @ C.T, dtype=float)


def _spin_square(fcisolver, ci, ncas: int, nelecas) -> tuple[float, float]:
    try:
        ss, s = fcisolver.spin_square(ci, ncas, nelecas)
        return float(np.asarray(ss).ravel()[0]), float(np.asarray(s).ravel()[0])
    except Exception:
        return float("nan"), float("nan")


def _ci_array(ci) -> np.ndarray:
    arr = np.asarray(ci)
    if arr.dtype == object:
        arr = np.stack([np.asarray(part, dtype=float) for part in ci])
    return np.asarray(arr, dtype=float)


def _write_record(
    *,
    project: str,
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    folder = _project_dir(project)
    digest = _digest_identity(payload["identity"])
    stem = f"{payload['system']['name']}_m{payload['system']['multiplicity']}_{digest}"
    npz_name = f"{stem}.npz"
    json_path = folder / f"{stem}.json"
    npz_path = folder / npz_name
    np.savez_compressed(npz_path, **arrays)
    payload["arrays"] = npz_name
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(json_path)
    _step(f"record       {json_path.relative_to(_REPO)}")
    sys.stdout.flush()
    return json_path


def _digest_identity(identity: dict[str, Any]) -> str:
    ident = json.dumps(identity, sort_keys=True, default=str)
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:12]


def _record_has_vqd_thetas(project: str | None, prev: dict[str, Any] | None) -> bool:
    if not project or not prev:
        return False
    arr_name = prev.get("arrays")
    if not arr_name:
        return False
    npz_path = _REPO / "data" / "projects" / str(project) / str(arr_name)
    if not npz_path.is_file():
        return False
    try:
        with np.load(npz_path) as packed:
            names = set(packed.files)
    except Exception:
        return False
    return any(name.startswith("vqd_theta_") for name in names)


def _load_record(project: str | None, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not project:
        return None
    digest = _digest_identity(identity)
    folder = _REPO / "data" / "projects" / project
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob(f"*_{digest}.json")):
        if path.name == "corpus.json":
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return None


def simulate_molecule(
    atoms: Sequence[Atom],
    *,
    name: str = "molecule",
    charge: int = 0,
    multiplicity: int = 1,
    basis: str = "sto3g",
    unit: str = "Angstrom",
    mapper: str = "jw",
    ansatz: str = "ucc-d",
    optimizer: str = "COBYLA",
    maxiter: int = 40,
    shots: int = 4096,
    active: str = "auto",
    max_spatial: int = 6,
    n_roots: int = 1,
    quantum_roots: int = 1,
    vqd_beta: float = 5.0,
    write_record: bool = False,
    project: str | None = None,
    geometry_tag: str = "",
    use_gpu: bool = False,
    cap_evals: bool = False,
    seed: int | None = 42,
) -> dict[str, Any]:
    """SCF, CAS (auto or full), mapped H, VQE, Aer shots at the VQE state.

    active='auto': freeze cores, CAS = valence occupied + lowest virtuals,
    truncated to max_spatial. Baseline is CASCI in that space.
    active='full': all MOs (only for tiny bases).

    maxiter is a floor. COBYLA maxfun is raised to about 8 per UCC angle
    (capped at 4000) and rhobeg is 0.05 so the first step stays near HF.
    Score E_VQE against E_CASCI, not full-space FCI.

    n_roots: CASCI/FCI how many lowest states to store as classical keys
    (CI, 1-RDM, dipole, S^2).

    quantum_roots: 1 = ground VQE. 2+ = VQD with k states on the same
    mapped H (same ansatz). Score vs CASCI roots. vqd_beta is the overlap
    penalty (length k-1).

    write_record: JSON + npz under data/projects/<project>/. project is
    required when write_record is True. CASCI states are the chemistry
    keys; VQE/VQD theta* / psi are the quantum stamp.

    use_gpu: AerSimulator(device=GPU) via gpu_simulator (RunPod NVIDIA).
    False keeps the original CPU StatevectorEstimator path.
    cap_evals: do not inflate COBYLA maxfun (bench only).
    seed: Aer shot/GPU seed; CPU exact estimator ignores it.
    """
    ao2mo, fci, gto, mcscf, scf = _load_pyscf()
    t_all = time.perf_counter()
    t_chem0 = time.perf_counter()

    if multiplicity < 1:
        raise ValueError("multiplicity must be 2S+1 >= 1.")
    if not atoms:
        raise ValueError("atoms is empty.")
    if max_spatial < 1:
        raise ValueError("max_spatial must be >= 1.")
    if write_record and not project:
        raise ValueError("project is required when write_record=True.")
    n_roots = int(n_roots)
    if n_roots < 1:
        raise ValueError("n_roots must be >= 1.")
    quantum_roots = int(quantum_roots)
    if quantum_roots < 1:
        raise ValueError("quantum_roots must be >= 1.")
    n_roots = max(n_roots, quantum_roots)

    print(f"== {name} ==", flush=True)

    mol = gto.Mole()
    mol.atom = _atom_string(atoms)
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.unit = unit
    mol.verbose = 0
    mol.build()
    n_spatial = int(mol.nao)
    n_alpha, n_beta = mol.nelec
    _step(
        f"molecule     {mol.natm} atoms  N={mol.nelectron}  "
        f"Q={charge}  multiplicity={multiplicity}  basis={basis}  "
        f"spatial={n_spatial}"
    )

    e_nn = float(mol.energy_nuc())
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    scf_name = "RHF" if mol.spin == 0 else "ROHF"
    e_hf = float(mf.kernel())
    C = np.asarray(mf.mo_coeff, dtype=float)
    mo_e = np.asarray(mf.mo_energy, dtype=float)
    _step(f"SCF          {scf_name}")
    print(f"     E_nn  = {e_nn:+.8f} hartree")
    print(f"     E_HF  = {e_hf:+.8f} hartree")

    key_active = active.lower().strip()
    e_fci_full = None
    cas_obj = None
    fcisolver = None
    nelecas = None
    ci_list: list = []
    e_casci_roots: list[float] = []
    state_pack: list[dict[str, Any]] = []
    if key_active == "full":
        n_core, ncas = 0, n_spatial
        n_alpha_cas, n_beta_cas = int(n_alpha), int(n_beta)
        nelecas = (n_alpha_cas, n_beta_cas)
        h1 = C.T @ np.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ C
        h2 = np.asarray(ao2mo.restore(1, ao2mo.kernel(mol, C), ncas), dtype=float)
        e_const = e_nn
        fcisolver = fci.FCI(mol, C)
        n_fci_probe = max(int(n_roots) * 4, 12)
        fcisolver.nroots = n_fci_probe
        print(
            f"     FCI kernel    nroots={n_fci_probe}  (filter to same spin after)",
            flush=True,
        )
        e_out, ci_out = fcisolver.kernel(nroots=n_fci_probe)
        e_casci_roots = [float(x) for x in np.atleast_1d(e_out).ravel()]
        ci_list = _ci_list(ci_out, len(e_casci_roots))
        s_val = 0.5 * float(mol.spin)
        ss_target = s_val * (s_val + 1.0)
        kept_e: list[float] = []
        kept_ci: list = []
        print(
            f"     FCI spin filter  2S={int(mol.spin)}  S^2 target={ss_target:.4f}",
            flush=True,
        )
        for ek, cik in zip(e_casci_roots, ci_list):
            ss, _ = _spin_square(fcisolver, cik, ncas, nelecas)
            keep = abs(float(ss) - ss_target) <= 0.75
            print(
                f"     FCI raw       E={ek:+.8f}  S^2={float(ss):.4f}  "
                f"{'keep' if keep else 'skip'}",
                flush=True,
            )
            if keep:
                kept_e.append(float(ek))
                kept_ci.append(cik)
            if len(kept_e) >= int(n_roots):
                break
        if len(kept_e) >= min(2, int(n_roots)):
            e_casci_roots, ci_list = kept_e, kept_ci
        else:
            print(
                "     FCI spin filter kept too few roots; using unfiltered list",
                flush=True,
            )
            e_casci_roots = e_casci_roots[: int(n_roots)]
            ci_list = ci_list[: int(n_roots)]
        e_casci = e_casci_roots[0]
        e_fci_full = e_casci
        _print_mo_partition(mo_e, 0, ncas)
        _step(f"active space  full FCI  {ncas} spatial -> {2 * ncas} qubits")
    elif key_active == "auto":
        n_core_atom = _n_core_spatial(mol)
        n_core, ncas, n_alpha_cas, n_beta_cas = _choose_cas(
            n_spatial, int(n_alpha), int(n_beta), n_core_atom, max_spatial
        )
        cas_obj = mcscf.CASCI(mf, ncas, (n_alpha_cas, n_beta_cas))
        cas_obj.frozen = n_core
        cas_obj.fcisolver.nroots = n_roots
        cas_obj.kernel()
        fcisolver = cas_obj.fcisolver
        nelecas = cas_obj.nelecas
        e_casci_roots = _total_energies(cas_obj.e_tot, cas_obj.e_cas)
        ci_list = _ci_list(cas_obj.ci, len(e_casci_roots))
        e_casci = e_casci_roots[0]
        h1, e_const = cas_obj.get_h1eff()
        h1 = np.asarray(h1, dtype=float)
        e_const = float(e_const)
        h2 = np.asarray(ao2mo.restore(1, cas_obj.get_h2eff(), ncas), dtype=float)
        _print_mo_partition(mo_e, n_core, ncas)
        _step(
            f"active space  CAS({n_alpha_cas + n_beta_cas}e, {ncas}o)  "
            f"frozen={n_core}"
        )
        if n_spatial <= 10:
            e_fci_full, _ci_full = fci.FCI(mol, C).kernel()
            e_fci_full = float(e_fci_full)
            print(f"     E_FCI_full = {e_fci_full:+.8f} hartree  (all MOs)")
    else:
        raise ValueError("active must be 'auto' or 'full'.")

    n_keep = min(len(ci_list), len(e_casci_roots), n_roots)
    ci_list = ci_list[:n_keep]
    e_casci_roots = e_casci_roots[:n_keep]
    for k, (ek, cik) in enumerate(zip(e_casci_roots, ci_list)):
        rdm1, ra, rb = _cas_rdm1(fcisolver, cik, ncas, nelecas)
        dm_ao = _ao_rdm1(C, cas_obj, fcisolver, cik, ncas, nelecas)
        dip = _dipole_au(mol, dm_ao)
        ss, s_val = _spin_square(fcisolver, cik, ncas, nelecas)
        state_pack.append(
            {
                "root": k,
                "energy": float(ek),
                "ci": cik,
                "rdm1": rdm1,
                "rdm1_a": ra,
                "rdm1_b": rb,
                "dipole_au": dip,
                "s2": ss,
                "spin": s_val,
            }
        )
        label = "ground" if k == 0 else f"{'FCI' if key_active == 'full' else 'CASCI'} root {k}"
        print(
            f"     {label:14s}  E={ek:+.8f}  "
            f"|mu|={float(np.linalg.norm(dip)):.4f} au  "
            f"S^2={ss:.4f}",
            flush=True,
        )

    target_name = "FCI" if key_active == "full" else "CASCI"
    print(
        f"     E_{target_name} = {e_casci:+.8f} hartree  "
        f"(VQE target = same-spin root 0)",
        flush=True,
    )
    if len(e_casci_roots) > 1:
        print(
            f"     E_{target_name}[1] = {e_casci_roots[1]:+.8f} hartree  "
            f"(VQD k=2 target if spin-locked)",
            flush=True,
        )

    fermionic = ElectronicEnergy.from_raw_integrals(h1_a=h1, h2_aa=h2)
    qmap = _mapper(mapper)
    h_el = qmap.map(fermionic.second_q_op())
    h_tot = h_el + SparsePauliOp("I" * h_el.num_qubits, coeffs=[e_const])
    _step(
        f"qubit H      {mapper.upper()}  "
        f"{h_tot.num_qubits} qubits  {len(h_tot)} Pauli terms"
    )

    print(
        f"     building {ansatz} (CPU circuit synthesis, GPU not used yet)...",
        flush=True,
    )
    t_ansatz = time.perf_counter()
    circuit, ansatz_id = _make_ansatz(
        ansatz, ncas, n_alpha_cas, n_beta_cas, qmap, h_tot.num_qubits
    )
    n_par = int(getattr(circuit, "num_parameters", 0) or 0)
    print(
        f"     ansatz ready  {n_par} parameters  "
        f"wall={time.perf_counter() - t_ansatz:.1f}s",
        flush=True,
    )

    e_vqe = None
    e_vqd: list[float] | None = None
    nfev = 0
    vqe_theta = None
    vqd_thetas: list[np.ndarray] | None = None
    e_shots = None
    e_shots_vqd: list[float] | None = None
    record_path = None
    e_ref = None
    opt_circuit = None
    extra_frozen_occ = n_core > _n_core_spatial(mol) if key_active == "auto" else False
    atoms_json = [
        [str(sym), [float(x), float(y), float(z)]] for sym, (x, y, z) in atoms
    ]
    chem_s = time.perf_counter() - t_chem0
    session = None
    stages: list[dict[str, Any]] = []
    vqe_status: dict[str, Any] | None = None
    vqd_status: dict[str, Any] | None = None
    t_vqe_s = 0.0
    t_vqd_s = 0.0

    def emit_record() -> None:
        nonlocal record_path
        if not write_record or opt_circuit is None or e_ref is None:
            return
        n_amp = 2 ** int(h_tot.num_qubits)
        try:
            if int(h_tot.num_qubits) >= 14:
                raise RuntimeError("skip CPU Statevector dump at 14+ qubits")
            psi = Statevector.from_instruction(opt_circuit.decompose(reps=8))
            psi_data = np.asarray(psi.data)
        except Exception as exc:
            print(f"     psi dump skipped ({type(exc).__name__}: {exc})", flush=True)
            psi_data = np.zeros(n_amp, dtype=np.complex128)
        try:
            labels = h_tot.paulis.to_labels()
        except Exception:
            labels = [p.to_label() for p in h_tot.paulis]
        pauli_labels = np.array(labels, dtype="U")
        pauli_coeffs = np.asarray(h_tot.coeffs, dtype=np.complex128)
        arrays: dict[str, np.ndarray] = {
            "h1": np.asarray(h1, dtype=float),
            "h2": np.asarray(h2, dtype=float),
            "mo_coeff": np.asarray(C, dtype=float),
            "mo_energy": np.asarray(mo_e, dtype=float),
            "pauli_labels": pauli_labels,
            "pauli_coeffs": pauli_coeffs,
            "psi_real": np.real(psi_data).astype(float),
            "psi_imag": np.imag(psi_data).astype(float),
            "e_casci_roots": np.asarray(e_casci_roots, dtype=float),
        }
        states_json = []
        for st in state_pack:
            k = int(st["root"])
            arrays[f"ci_{k}"] = _ci_array(st["ci"])
            arrays[f"rdm1_cas_{k}"] = np.asarray(st["rdm1"], dtype=float)
            if st["rdm1_a"] is not None:
                arrays[f"rdm1_a_{k}"] = st["rdm1_a"]
            if st["rdm1_b"] is not None:
                arrays[f"rdm1_b_{k}"] = st["rdm1_b"]
            dip = np.asarray(st["dipole_au"], dtype=float)
            arrays[f"dipole_au_{k}"] = dip
            states_json.append(
                {
                    "root": k,
                    "label": (
                        "ground in this multiplicity and CAS"
                        if k == 0
                        else f"CASCI root {k} (same spin manifold)"
                    ),
                    "energy": float(st["energy"]),
                    "source": "casci",
                    "s2": float(st["s2"]),
                    "spin": float(st["spin"]),
                    "dipole_au": [float(x) for x in dip],
                    "dipole_debye": [float(x) * _AU_TO_DEBYE for x in dip],
                    "dipole_norm_debye": float(np.linalg.norm(dip) * _AU_TO_DEBYE),
                    "arrays": f"ci_{k}, rdm1_cas_{k}, dipole_au_{k}",
                }
            )
        if vqe_theta is not None:
            arrays["vqe_theta"] = vqe_theta
        if vqd_thetas is not None:
            for i, th in enumerate(vqd_thetas):
                arrays[f"vqd_theta_{i}"] = th
                psi_i = Statevector.from_instruction(
                    circuit.assign_parameters(th).decompose(reps=8)
                )
                arrays[f"vqd_psi_real_{i}"] = np.real(np.asarray(psi_i.data)).astype(float)
                arrays[f"vqd_psi_imag_{i}"] = np.imag(np.asarray(psi_i.data)).astype(float)
        if e_vqd is not None:
            arrays["e_vqd"] = np.asarray(e_vqd, dtype=float)
        payload = {
            "schema": _SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "code": "mol_simulate.simulate_molecule",
                "pyscf": _pkg_ver("pyscf"),
                "qiskit": _pkg_ver("qiskit"),
                "qiskit_nature": _pkg_ver("qiskit-nature"),
                "qiskit_aer": _pkg_ver("qiskit-aer"),
                "backend": None if session is None else session.backend_info,
            },
            "system": {
                "name": name,
                "charge": int(charge),
                "multiplicity": int(multiplicity),
                "n_electrons": int(mol.nelectron),
                "unit": unit,
                "geometry_tag": geometry_tag,
                "atoms": atoms_json,
            },
            "model": {
                "basis": basis,
                "scf": scf_name,
                "active": key_active,
                "n_spatial": int(n_spatial),
                "n_core": int(n_core),
                "n_cas": int(ncas),
                "n_alpha_cas": int(n_alpha_cas),
                "n_beta_cas": int(n_beta_cas),
                "n_roots": int(len(state_pack)),
                "mo_role": _mo_roles(n_spatial, n_core, ncas),
                "note": (
                    "CASCI roots are classical keys in this CAS. "
                    "VQE/VQD stamp the mapped CAS H. Full-space FCI is a "
                    "different Hamiltonian if active!=full."
                ),
            },
            "identity": {
                "name": name,
                "charge": int(charge),
                "multiplicity": int(multiplicity),
                "basis": basis,
                "unit": unit,
                "geometry_tag": geometry_tag,
                "atoms": atoms_json,
                "active": key_active,
                "n_cas": int(ncas),
                "n_core": int(n_core),
                "n_roots": int(len(state_pack)),
                "mapper": mapper.lower().strip(),
                "ansatz": ansatz_id,
                "quantum_roots": int(quantum_roots),
            },
            "energies": {
                "e_nn": e_nn,
                "e_hf": e_hf,
                "e_casci": e_casci,
                "e_casci_roots": e_casci_roots,
                "e_fci_full": e_fci_full,
                "e_circuit": float(e_ref),
                "e_vqe": e_vqe,
                "e_vqd": e_vqd,
                "e_shots": e_shots,
                "e_shots_vqd": e_shots_vqd,
                "unit": "hartree",
            },
            "states": states_json,
            "quantum": {
                "mapper": mapper.lower().strip(),
                "n_qubits": int(h_tot.num_qubits),
                "n_pauli": int(len(h_tot)),
                "ansatz": ansatz_id,
                "n_parameters": int(n_par),
                "optimizer": optimizer if n_par else None,
                "nfev": nfev if n_par else None,
                "quantum_roots": int(quantum_roots),
                "vqd_beta": float(vqd_beta) if quantum_roots > 1 else None,
                "e_vqe": e_vqe,
                "e_vqd": e_vqd,
                "e_shots": e_shots,
                "e_shots_vqd": e_shots_vqd,
                "shots": int(shots) if shots > 0 else 0,
                "checkpoint": e_vqd is None and e_vqe is not None and quantum_roots > 1,
                "use_gpu": bool(use_gpu),
                "device": None if session is None else session.device,
                "seed": seed,
                "runtime_s": {
                    "chemistry": chem_s,
                    "vqe": t_vqe_s,
                    "vqd": t_vqd_s,
                    "wall": time.perf_counter() - t_all,
                    **({} if session is None else dict(session.profile)),
                },
                "stages": stages,
                "optimizer_status": {
                    "vqe": vqe_status,
                    "vqd": vqd_status,
                },
                "arrays": "vqe_theta, psi_real, psi_imag, vqd_theta_*, vqd_psi_*",
            },
        }
        record_path = _write_record(
            project=str(project), payload=payload, arrays=arrays
        )

    identity = {
        "name": name,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "basis": basis,
        "unit": unit,
        "geometry_tag": geometry_tag,
        "atoms": atoms_json,
        "active": key_active,
        "n_cas": int(ncas),
        "n_core": int(n_core),
        "n_roots": int(len(state_pack)),
        "mapper": mapper.lower().strip(),
        "ansatz": ansatz_id,
        "quantum_roots": int(quantum_roots),
    }
    prev = _load_record(project, identity) if write_record else None
    prev_vqd = (prev or {}).get("energies", {}).get("e_vqd") or []
    if quantum_roots == 1:
        quantum_done = (
            prev is not None
            and prev.get("energies", {}).get("e_vqe") is not None
        )
    else:
        quantum_done = (
            isinstance(prev_vqd, list)
            and len(prev_vqd) >= quantum_roots
            and _record_has_vqd_thetas(project, prev)
        )

    if quantum_done:
        _step("skip         complete quantum record")
        energies = prev.get("energies") or {}
        e_vqe = energies.get("e_vqe")
        e_vqd = energies.get("e_vqd")
        e_shots = energies.get("e_shots")
        e_shots_vqd = energies.get("e_shots_vqd")
        nfev = _as_int((prev.get("quantum") or {}).get("nfev"))
        e_ref = energies.get("e_circuit")
        digest = _digest_identity(identity)
        hits = list(
            (_REPO / "data" / "projects" / str(project)).glob(f"*_{digest}.json")
        )
        record_path = hits[0] if hits else None
        print(f"     using {record_path.name if record_path else 'existing json'}")
    elif n_par == 0:
        opt_circuit = circuit
        e_ref = _statevector_energy(circuit, h_tot)
        _step("HF circuit")
        print(f"     E_circuit = {e_ref:+.8f} hartree")
        if (
            not extra_frozen_occ
            and multiplicity == 1
            and abs(e_ref - e_hf) > 1e-5
        ):
            raise RuntimeError("HF circuit disagrees with SCF.")
        if quantum_roots > 1:
            print("     VQD skipped: ansatz has no parameters (use ucc-* or hea).")
        emit_record()
    else:
        zeros = np.zeros(n_par)
        skip_cpu_sv = bool(use_gpu) or int(h_tot.num_qubits) >= 14
        if skip_cpu_sv:
            e_ref = float(e_hf)
            _step(f"ansatz       {ansatz_id}  {n_par} parameters")
            print(
                "     skip CPU Statevector theta=0  "
                "(E_HF; Aer GPU does VQE, not Terra decompose)",
                flush=True,
            )
        else:
            e_ref = _statevector_energy(circuit.assign_parameters(zeros), h_tot)
            _step(f"ansatz       {ansatz_id}  {n_par} parameters")
            print(f"     E(theta=0) = {e_ref:+.8f} hartree", flush=True)
        if ansatz_id.startswith("ucc") and not extra_frozen_occ:
            if abs(e_ref - e_hf) > 1e-4:
                print("     warning: theta=0 is not full-space E_HF")

        opt = _optimizer(
            optimizer, maxiter, n_par, ansatz_id, cap_evals=cap_evals
        )
        session = make_session(use_gpu, seed=seed)
        session.backend_info["n_qubits"] = int(h_tot.num_qubits)
        session.profile["other_s"] = session.profile.get("other_s", 0.0) + chem_s
        best = [e_ref]
        arr_name = (prev or {}).get("arrays")
        if (
            prev is not None
            and prev.get("energies", {}).get("e_vqe") is not None
            and arr_name
        ):
            npz_path = _REPO / "data" / "projects" / str(project) / str(arr_name)
            if npz_path.is_file():
                with np.load(npz_path) as packed:
                    if "vqe_theta" in packed.files:
                        vqe_theta = np.asarray(packed["vqe_theta"], dtype=float)
                if vqe_theta is not None:
                    e_vqe = float(prev["energies"]["e_vqe"])
                    opt_circuit = circuit.assign_parameters(vqe_theta)
                    nfev = _as_int((prev.get("quantum") or {}).get("nfev"))
                    prev_stages = (prev.get("quantum") or {}).get("stages") or []
                    if isinstance(prev_stages, list):
                        stages.extend(prev_stages)
                    _step("resume       ground VQE checkpoint")
                    print(
                        f"     E_VQE = {e_vqe:+.8f} hartree  (from disk)",
                        flush=True,
                    )

        if vqe_theta is None:
            def _cb(n_eval, _theta, energy, _meta) -> None:
                e = float(np.real(energy))
                improved = e < best[0] - 1e-6
                if improved:
                    best[0] = e
                if n_eval == 1 or n_eval % 20 == 0 or improved:
                    print(
                        f"     nfev={n_eval:4d}  E={e:+.8f}",
                        flush=True,
                    )

            t0 = time.perf_counter()
            result = VQE(
                session.estimator,
                circuit,
                opt,
                callback=_cb,
                initial_point=zeros,
            ).compute_minimum_eigenvalue(h_tot)
            t_vqe_s = time.perf_counter() - t0
            e_vqe = float(np.real(result.eigenvalue))
            nfev = _as_int(result.cost_function_evals)
            vqe_theta = np.asarray(result.optimal_point, dtype=float)
            opt_circuit = circuit.assign_parameters(result.optimal_point)
            vqe_status = _opt_status(result)
            vqe_status["runtime_s"] = t_vqe_s
            stages.append(
                {
                    "name": "ground",
                    "energy": e_vqe,
                    "nfev": nfev,
                    "runtime_s": t_vqe_s,
                    "optimizer": vqe_status,
                    "utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _step(
                f"VQE          {optimizer}  ({session.device}, exact E(theta))"
            )
            print(f"     E_VQE = {e_vqe:+.8f} hartree", flush=True)
            print(f"     evals = {nfev}", flush=True)
            print(f"     runtime = {t_vqe_s:.1f} s", flush=True)
            drop = e_ref - e_vqe
            above_cas = e_vqe - e_casci
            print(f"     vs start  {drop:+.8f}   (positive = below theta=0)")
            print(f"     vs CASCI  {above_cas:+.8f}   (zero = hit the CAS floor)")
            if drop < 1e-6:
                print(
                    "     VQE stayed at the start. Raise maxiter, use L_BFGS_B, "
                    "or cut max_spatial (fewer theta)."
                )
            elif above_cas > 5e-3:
                print(
                    "     Below HF, still above CASCI. More evals or a tighter "
                    "CAS (smaller max_spatial) usually helps."
                )
            if quantum_roots > 1:
                emit_record()
                sys.stdout.flush()
                _step("checkpoint   ground saved before VQD k>=2")

        if quantum_roots > 1 and vqe_theta is not None:
            try:
                fidelity = ComputeUncompute(session.sampler)
            except ValueError:
                from qiskit.primitives import Sampler as TerraSampler

                print(
                    "     VQD overlap fallback: Terra Sampler (BaseSampler)",
                    flush=True,
                )
                fidelity = ComputeUncompute(TerraSampler())
            betas = np.full(max(quantum_roots - 1, 1), float(vqd_beta))
            init_vqd = [np.asarray(vqe_theta, dtype=float)]
            init_vqd.extend(np.zeros(n_par) for _ in range(quantum_roots - 1))
            print(
                f"     VQD k={quantum_roots}  beta={float(vqd_beta):.4f}  "
                f"(same H, overlap penalty; start from VQE theta*)",
                flush=True,
            )
            try:
                try:
                    vqd = VQD(
                        session.estimator,
                        fidelity,
                        circuit,
                        opt,
                        k=quantum_roots,
                        betas=betas,
                        initial_point=init_vqd,
                        callback=_vqd_callback(),
                    )
                except (TypeError, ValueError):
                    betas = np.full(quantum_roots, float(vqd_beta))
                    vqd = VQD(
                        session.estimator,
                        fidelity,
                        circuit,
                        opt,
                        k=quantum_roots,
                        betas=betas,
                        initial_point=init_vqd,
                        callback=_vqd_callback(),
                    )
                t0 = time.perf_counter()
                result = vqd.compute_eigenvalues(h_tot)
                t_vqd_s = time.perf_counter() - t0
                e_vqd = [float(np.real(x)) for x in np.atleast_1d(result.eigenvalues)]
                e_vqe = e_vqd[0]
                nfev = _as_int(getattr(result, "cost_function_evals", None))
                vqd_thetas = _unpack_thetas(result, n_par, quantum_roots)
                if not vqd_thetas:
                    raise RuntimeError("VQD returned energies but no optimal_point(s).")
                vqe_theta = vqd_thetas[0]
                opt_circuit = circuit.assign_parameters(vqd_thetas[0])
                vqd_status = _opt_status(result)
                vqd_status["runtime_s"] = t_vqd_s
                for i, ev in enumerate(e_vqd):
                    stages.append(
                        {
                            "name": "ground" if i == 0 else f"excited_{i}",
                            "energy": ev,
                            "nfev": nfev,
                            "runtime_s": t_vqd_s if i > 0 else t_vqe_s,
                            "optimizer": vqd_status,
                            "utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                _step(f"VQD          {optimizer}  k={quantum_roots}")
                for i, ev in enumerate(e_vqd):
                    key = e_casci_roots[i] if i < len(e_casci_roots) else None
                    line = f"     E_VQD[{i}] = {ev:+.8f} hartree"
                    if key is not None:
                        tag = "FCI" if key_active == "full" else "CASCI"
                        line += f"   vs {tag} {ev - key:+.8f}"
                    print(line, flush=True)
                print(f"     evals = {nfev}", flush=True)
                print(f"     runtime = {t_vqd_s:.1f} s", flush=True)
                emit_record()
                sys.stdout.flush()
            except Exception as exc:
                print(
                    f"     VQD failed ({type(exc).__name__}): {exc}",
                    flush=True,
                )
                traceback.print_exc()
                if e_vqd is not None and opt_circuit is not None:
                    try:
                        emit_record()
                        print("     saved partial VQD energies", flush=True)
                    except Exception:
                        traceback.print_exc()
                _step("keep         ground checkpoint; grid can resume this point")

        if (
            shots > 0
            and opt_circuit is not None
            and (quantum_roots == 1 or e_vqd is not None)
        ):
            try:
                e_shots = session.estimate_shots(opt_circuit, h_tot, shots)
                _step(f"Aer shots    {shots}  (ground circuit)")
                print(f"     E_shots = {e_shots:+.8f} hartree", flush=True)
                if vqd_thetas is not None and len(vqd_thetas) > 1:
                    e_shots_vqd = [e_shots]
                    for i, th in enumerate(vqd_thetas[1:], start=1):
                        qc_i = circuit.assign_parameters(th)
                        ei = session.estimate_shots(qc_i, h_tot, shots)
                        e_shots_vqd.append(ei)
                        print(f"     E_shots[{i}] = {ei:+.8f} hartree", flush=True)
            except Exception as exc:
                print(
                    f"     Aer failed ({type(exc).__name__}): {exc}",
                    flush=True,
                )
                traceback.print_exc()

        emit_record()
        session.report()

    print("-----")
    print(f"     E_HF      {e_hf:+.8f}   full-space SCF")
    if e_vqe is not None:
        print(f"     E_VQE     {e_vqe:+.8f}   exact ground <psi|H_cas|psi>")
    if e_vqd is not None:
        print(f"     E_VQD     {[round(x, 8) for x in e_vqd]}")
    if e_shots is not None:
        print(f"     E_shots   {e_shots:+.8f}   Aer of the ground circuit")
    print(f"     E_CASCI   {e_casci:+.8f}   classical exact in the CAS (VQE target)")
    if e_fci_full is not None and key_active != "full":
        print(f"     E_FCI     {e_fci_full:+.8f}   classical exact, all MOs (not this H)")
    print("-----")

    return {
        "name": name,
        "e_nn": e_nn,
        "e_hf": e_hf,
        "e_casci": e_casci,
        "e_casci_roots": e_casci_roots,
        "e_fci_full": e_fci_full,
        "e_vqe": e_vqe,
        "e_vqd": e_vqd,
        "e_shots": e_shots,
        "n_spatial": n_spatial,
        "n_cas": ncas,
        "n_core": n_core,
        "n_qubits": int(h_tot.num_qubits),
        "n_alpha_cas": n_alpha_cas,
        "n_beta_cas": n_beta_cas,
        "ansatz": ansatz_id,
        "nfev": nfev,
        "n_roots": int(len(state_pack)),
        "quantum_roots": int(quantum_roots),
        "record_path": str(record_path) if record_path else None,
        "use_gpu": bool(use_gpu),
        "device": None if session is None else session.device,
        "stages": stages,
        "runtime_s": time.perf_counter() - t_all,
    }
