# Molecule simulator (`mol_simulate.py`)

**What this is:** how `simulate_molecule` turns one geometry into SCF, CASCI keys, a qubit Hamiltonian, VQE/VQD, and a JSON+npz record.
**What this is not:** the GPU host manual. That is [GPU_SIMULATION.md](GPU_SIMULATION.md).

Entry point: `simulate_molecule(...)` in `mol_simulate.py`. The NH3 grid is `main.py`; combining records is `scripts/combine_records.py`.

---

## How to read the symbols

| Write | Say | Means |
|---|---|---|
| `H_cas` | H-cas | Qubit Hamiltonian of the active space plus nuclear/frozen constant |
| `|psi(theta)>` | psi of theta | Ansatz state at parameter vector theta |
| `E(theta)` | E of theta | `<psi(theta)| H_cas |psi(theta)>` |
| `k` | k | VQD state index (1 = ground hunt, 2 = first excited hunt) |
| `beta` | beta | Overlap penalty weight in VQD |
| CAS(8e, 6o) | C-A-S eight electrons six orbitals | Active space used for NH3 STO-3G here |

English: VQE turns theta to lower E(theta). VQD does that again with a penalty if the new state overlaps a previous one.

---

## Inputs

All keyword arguments of `simulate_molecule` except `atoms` (positional).

| Argument | Typical NH3 (`main.py`) | Role |
|---|---|---|
| `atoms` | N at origin, three H, Angstrom | Geometry |
| `name` | `NH3` | Record stem |
| `charge` | 0 | Molecular charge |
| `multiplicity` | 1 or 3 | `2S+1`; chooses RHF vs ROHF |
| `basis` | `sto3g` | PySCF basis |
| `unit` | `Angstrom` | Coordinate unit |
| `mapper` | `bk` | `jw` or `bk` (Bravyi-Kitaev) |
| `ansatz` | `ucc-d` | `hf`, `ucc-s`, `ucc-d`, `ucc-sd`, `hea` |
| `optimizer` | `COBYLA` | also `SLSQP`, `L_BFGS_B` |
| `maxiter` | 40 | Floor; COBYLA maxfun is raised unless `cap_evals` |
| `shots` | 4096 | Aer shot stamp after optimize; `0` skips |
| `active` | `auto` | `auto` = freeze cores + cap; `full` = all MOs |
| `max_spatial` | 6 | Cap on active spatial orbitals (NH3 -> 12 qubits) |
| `n_roots` | 6 | How many CASCI roots to store as classical keys |
| `quantum_roots` | 1 or 2 | 1 = VQE only; 2+ = VQD k states |
| `vqd_beta` | 5.0 | Overlap penalty |
| `write_record` | True on the dataset | Write JSON+npz |
| `project` | `nh3-sto3g-cas` | Folder under `data/projects/` |
| `geometry_tag` | e.g. `pyramidal_R1.012` | Part of identity / resume key |
| `use_gpu` | `main.py` True | See gpu_simulator |
| `cap_evals` | False; True on bench | Do not inflate COBYLA maxfun |
| `seed` | 42 | Aer / GPU shot seed |

Windows without PySCF: the module re-execs under WSL micromamba env `flibe`. On RunPod, PySCF is in the venv; no WSL.

---

## High-level flow

```text
atoms, charge, multiplicity, basis
  -> PySCF Mole.build
  -> RHF (singlet) or ROHF (open shell)
  -> active='auto': freeze cores, CAS window, cap max_spatial
  -> CASCI n_roots  (energies, CI, 1-RDM, dipole, S^2)
  -> ElectronicEnergy integrals -> mapper -> SparsePauliOp H_cas
  -> ansatz (HF or UCC or HEA)
  -> if record complete (e_vqd + vqd_theta_*): skip quantum
  -> else VQE on H_cas  (checkpoint if quantum_roots>1)
  -> if quantum_roots>1: VQD k states, beta overlap penalty
  -> optional Aer shots
  -> JSON + npz  (always after a successful quantum block)
```

---

## Steps and design reasons

### 1. Molecule and SCF

PySCF builds the geometry string, nuclear energy `E_nn`, and a restricted (open-shell) Hartree-Fock energy `E_HF`. Reason: the qubit problem is a **CAS Hamiltonian**, not full FCI. SCF MOs define the frozen/active/virtual cut.

### 2. Active space (`active='auto'`)

Cores: 1s on atoms with Z >= 3 (N 1s frozen for NH3). Occupied valence plus lowest virtuals, truncated so spatial orbitals <= `max_spatial` (6 -> 12 spin-orbitals -> 12 qubits). If the cap still overflows, extra occupied MOs freeze too (`extra_frozen_occ`).

Reason: UCC-D parameter count grows fast. 12 qubits / ~76 doubles is already a long COBYLA hunt. Score VQE against **CASCI in this window**, not full-space FCI. Full FCI is printed when `n_spatial <= 10` as a different Hamiltonian.

`active='full'` keeps every MO (STO-3G NH3: 8 spatial -> 16 qubits). The dataset command is `python main.py all`, which stays on `active='auto'` (CAS, 12 qubits). A full-space UCC-SD grid was started and then stopped; it is not a launch mode.

PySCF FCI is run with extra roots, then filtered to the SCF multiplicity by S^2 (singlet keys for m=1, triplet keys for m=3). Reason: unfiltered FCI root 1 is often a different spin, so "match FCI[1]" would be the wrong target. `fix_spin_` is not used; it stalled Davidson on this 8-orbital NH3.

GPU makes 2^16 statevectors cheap. UCC-SD is still a truncated excitation operator; it is not a complete FCI expansion. Score `vs FCI` in the log is the honest delta. Do not treat a small GPU as a proof of exact FCI.

### 3. CASCI / FCI keys (`n_roots`)

Classical exact roots in the chosen space: energy, CI vector, 1-RDM (and spin RDMs if available), dipole in au and Debye, `S^2`. These are the chemistry keys in the record. VQE/VQD do not replace them.

### 4. Qubit Hamiltonian

```text
H_cas  =  mapped(fermionic CAS integrals)  +  E_const * I
```

`E_const` is the frozen-core plus nuclear piece from `get_h1eff`. Mapper default in the dataset is Bravyi-Kitaev (`bk`), not Jordan-Wigner. Identity includes the mapper, so JW and BK are different records.

### 5. Ansatz

- `hf`: Hartree-Fock bitstring circuit, zero parameters (no VQE).
- `ucc-*`: Qiskit Nature UCC on the CAS, HF initial state. `theta=0` should sit near `E_HF` if the CAS is not extra-frozen.
- `hea`: hardware-efficient `ry` + linear entanglement after HF.

COBYLA `rhobeg=0.05` for UCC so the first step does not jump to a high-energy region. Default SciPy `rhobeg=1` is a bad scale for amplitudes ~0.01-0.1.

### 6. Optimizer budget

If `n_par >= 8` and not `cap_evals`, COBYLA `maxfun = min(cap, max(maxiter, 8*n_par))` with `cap=8000` when `n_par >= 80` else `4000`. NH3 CAS UCC-D: 76 parameters -> 608 evals per hunt. Full-space UCC-SD has more angles, so the higher cap applies. `cap_evals=True` (bench) honors `maxiter` as a hard cap.

### 7. Resume

Identity hash (not GPU/CPU) keys the JSON filename. A record is **quantum-complete** when:

- `quantum_roots == 1` and `e_vqe` is set, or
- `quantum_roots > 1` and `e_vqd` has at least k numbers **and** the npz has `vqd_theta_*`.

Energies without thetas are treated as incomplete so VQD is not skipped after a crash in result packing.

If `e_vqe` and `vqe_theta` exist but VQD is missing, ground VQE is reused (`resume`).

### 8. VQE then VQD

```text
VQE:   min_theta  E(theta)  =  <psi(theta)| H_cas |psi(theta)>

VQD:   for k = 1, 2, ...
         min_theta  E(theta) + sum_{j<k}  beta_j * |<psi(theta)| psi_j>|^2
```

`use_gpu=True` takes estimator and sampler from `make_session`. Overlap is compute-uncompute via a Terra `BaseSampler` (`BackendSampler` on Aer GPU). After VQE, a checkpoint is written before VQD so a VQD crash does not lose the ground state.

### 9. Shots

If `shots > 0` and the quantum loop finished, Aer estimates `E` with that many shots (GPU or CPU Aer). This is a stamp, not the optimizer metric. Optimizer energies are exact (or Aer approximation=True statevector) expectation values.

### 10. Record write

Atomic JSON replace (`.json.tmp` then replace). npz is compressed. Identity does not include `use_gpu`, so a GPU pod can resume a CPU ground checkpoint.

---

## Outputs

Return dict (always): `e_nn`, `e_hf`, `e_casci`, `e_casci_roots`, `e_fci_full`, `e_vqe`, `e_vqd`, `e_shots`, sizes (`n_spatial`, `n_cas`, `n_qubits`, ...), `nfev`, `record_path`, `use_gpu`, `device`, `stages`, `runtime_s`.

When `write_record=True`, `data/projects/<project>/<Name>_m<M>_<12-hex>.json` plus `.npz`.

JSON (schema `ammonia-deep-dive.record.v1`):

- `provenance` — package versions, GPU backend dict
- `system` — name, charge, multiplicity, geometry_tag, atoms
- `model` — basis, SCF, CAS sizes, MO roles
- `identity` — resume key
- `energies` — hartree
- `states` — CASCI keys (dipole, S^2)
- `quantum` — qubits, Pauli count, nfev, checkpoint flag, runtimes, stages

npz arrays (names vary): `h1`, `h2`, `mo_coeff`, `mo_energy`, Pauli labels/coeffs, `psi_real/imag`, `ci_*`, `rdm1_*`, `dipole_au_*`, `vqe_theta`, `vqd_theta_*`, `vqd_psi_*`, `e_vqd`.

`python scripts/combine_records.py nh3-sto3g-cas` unions JSON into `corpus.json`.

---

## NH3 grid (`main.py`)

| Tag | Geometry |
|---|---|
| `pyramidal_R0.900` | pyramid, R(N-H)=0.90 A, H-N-H=106.7 deg |
| `pyramidal_R1.012` | equilibrium-style R |
| `pyramidal_R1.120` | stretched |
| `planar_R1.012` | planar, same R |

Each tag x multiplicity 1 and 3. Commands: `python main.py all` (dataset), `sweep` (CASCI+HF), `vqe`, `vqd`, `doctor`, `bench`.

---

## Design decisions (short)

| Choice | Reason |
|---|---|
| CAS not FCI as VQE target | Qubit H is the CAS; FCI is a different operator |
| Freeze N 1s | Standard minimal-basis core; keeps qubits finite |
| BK mapper on the dataset | Fixed identity; JW is a different experiment |
| UCC-D + COBYLA rhobeg 0.05 | Chemistry-motivated doubles; step size matches amplitudes |
| VQE then VQD k=2 | Ground checkpoint before the long excited hunt |
| beta=5 | Default overlap penalty in this dataset (not a fitted constant) |
| Complete = energies + thetas | Crash after VQD must not skip the next `all` |
| GPU only in primitives | Do not rewrite chemistry in CUDA-Q |
