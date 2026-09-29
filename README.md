# Ammonia STO-3G CAS energies with VQE and VQD

**Santhosh Reddy** · study · MIT · v0.2.0 (2026)
**Status:** Computational experiment. Not peer-reviewed. Not a laboratory spectrum.

Eight ammonia geometries and spin states, minimal basis, a 12-qubit active-space Hamiltonian. Hartree–Fock and CASCI are the classical references. VQE/VQD stamps a ground energy and one excited energy on that same Hamiltonian. The numbers below are the records in `data/projects/nh3-sto3g-cas/`, from a Qiskit Aer GPU statevector run. They are not coupled-cluster or experimental NH3 energies.

**Keywords:** ammonia, STO-3G, CASCI, Bravyi–Kitaev, UCC, VQE, VQD, Qiskit Aer.

## Question

At a few N–H distances, pyramidal and planar, singlet and triplet: how close does a doubles-only unitary coupled-cluster ansatz, optimized by VQE and then VQD, get to CASCI on the same active-space Hamiltonian?

## Result

Energies in hartree. `E_VQD[0]` is the ground stamp. `E_VQD[1]` is the excited stamp. CASCI root 1 is the classical excited root in the same active space, not a measured UV line.

| Geometry | Mult. | E_HF | E_CASCI | E_VQD[0] | E_VQD[1] | CASCI root 1 |
|---|---:|---:|---:|---:|---:|---:|
| pyramidal R=0.900 A | 1 | -55.392114 | -55.418370 | -55.418311 | -52.534017 | -54.780945 |
| pyramidal R=1.012 A | 1 | -55.454039 | -55.487828 | -55.487712 | -52.916299 | -54.995126 |
| pyramidal R=1.120 A | 1 | -55.434000 | -55.477095 | -55.476803 | -53.984661 | -55.096899 |
| planar R=1.012 A | 1 | -55.437526 | -55.469484 | -55.469291 | -54.133618 | -55.087234 |
| pyramidal R=0.900 A | 3 | -54.765221 | -54.784622 | -54.784456 | -53.716280 | -54.763474 |
| pyramidal R=1.012 A | 3 | -54.975317 | -55.000187 | -54.999863 | -53.951918 | -54.961989 |
| pyramidal R=1.120 A | 3 | -55.070612 | -55.102250 | -55.101628 | -54.766449 | -55.050574 |
| planar R=1.012 A | 3 | -55.070550 | -55.092977 | -55.092652 | -53.115863 | -54.940645 |

Ground VQD sits within about 0.001 hartree of CASCI on every point. The excited VQD stamp sits 0.28 to 2.25 hartree above CASCI root 1. Same Hamiltonian, same ansatz family. The excited hunt did not find that CASCI root.

Regenerate the table from the files:

```text
python scripts/combine_records.py nh3-sto3g-cas
python -m pytest tests -q
```

The pytest check does not rerun VQE. It checks that those eight files are present and that the ground/excited pattern above still holds.

## Method

```text
N at the origin, three H, Angstrom
  -> PySCF RHF (singlet) or ROHF (triplet), STO-3G
  -> freeze N 1s, keep 6 spatial orbitals (12 qubits)
  -> CASCI, six roots, as the classical keys
  -> Bravyi-Kitaev map
  -> UCC doubles, COBYLA, rhobeg 0.05, about 8 evaluations per angle
  -> VQE ground, then VQD k=2, overlap penalty beta=5
  -> optional Aer shot stamp, 4096 shots
```

Say out loud: the circuit prepares a trial state. The Hamiltonian is a sum of Pauli strings. VQD adds a penalty when the new state overlaps the previous one. It does not diagonalize the CAS.

Grid: pyramidal R(N–H) in {0.90, 1.012, 1.12} angstrom, plus planar R=1.012 angstrom. Multiplicity 1 and 3. Angle 106.7 degrees on the pyramids.

The published run used Qiskit 1.4.2, Qiskit Nature 0.7.2, qiskit-algorithms 0.3.1, and qiskit-aer-gpu 0.15.1 on an NVIDIA GPU. A current laptop install (`requirements.txt`) is Qiskit 2.5.2 and Nature 0.8.0. Same chemistry construction. Not the same wheel. `requirements.txt` is the laptop pin. The GPU pin is written in that file as comments and in [docs/GPU_SIMULATION.md](docs/GPU_SIMULATION.md).

## Reproducibility

Clone, then follow [GUIDE.md](GUIDE.md). Short version:

```text
python -m pip install -r requirements.txt
python -m pytest tests -q
python main.py sweep
```

`sweep` is Hartree–Fock and CASCI only. It is the check that the chemistry stack runs. `python main.py all` repeats the VQE/VQD grid and takes hours. The energies in the table are already in `data/projects/nh3-sto3g-cas/`.

GPU: set `AMMONIA_USE_GPU=1` only after `python gpu_simulator.py doctor` prints `GPU init ok`. See [docs/GPU_SIMULATION.md](docs/GPU_SIMULATION.md). No RunPod API key. No IBM job.

## Limitations

- STO-3G and CAS(8e, 6o) are a model. Not a complete-basis ammonia spectrum.
- One bond angle on the pyramids. Four geometries, two spins. Not a surface.
- Excited VQD is not CASCI root 1. Quoting `E_VQD[1]` as "the" excited state of ammonia would be wrong.
- Twelve-qubit statevectors are small. A GPU speeds the repeated Aer calls. It does not change the molecule. If the card looks idle, the time is usually Python building the circuit.
- VQD overlap on the GPU path uses Aer shots (1024), not an analytic inner product. Shot stamps at 4096 shots will not match the statevector energy digit for digit.
- A full-orbital UCC singles-and-doubles run (16 qubits, no CAS) was started and stopped. It is not in this repository.

## Layout

```text
README.md            this page
GUIDE.md             install and what to run first
main.py              geometry grid
mol_simulate.py      SCF, CAS, map, VQE/VQD, records
gpu_simulator.py     Aer CPU or GPU session
scripts/combine_records.py
data/projects/nh3-sto3g-cas/   eight JSON+npz records
docs/                method notes, GPU notes, intuition, lab book
tests/               checks the eight records
```

## Citation

Cite this repository for the pipeline and the table. Cite PySCF and Qiskit Nature for the libraries. Do not cite the table as experimental ammonia thermochemistry.

```text
@software{reddy2026ammonia,
  author  = {Santhosh Reddy},
  title   = {Ammonia STO-3G CAS energies with VQE and VQD},
  year    = {2026},
  license = {MIT},
  note    = {Study. Eight geometries. Ground VQD near CASCI. Excited VQD above CASCI root 1.}
}
```
