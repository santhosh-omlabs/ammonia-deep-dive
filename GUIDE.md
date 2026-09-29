# Guide

Install, check the published records, optionally rerun chemistry. The energies on the README are already in `data/projects/nh3-sto3g-cas/`. You do not have to repeat the VQE grid to read the result.

Do not send jobs to a paid quantum processor. This repo never calls the RunPod HTTP API.

## 0. What you need

Python 3.10 or newer.

```text
python -m pip install -r requirements.txt
```

That file pins the laptop stack (Qiskit 2.5.2, Qiskit Nature 0.8.0, PySCF). `pytest` is in the same file.

On native Windows, `pip install pyscf` often fails. Two ways through:

- Linux, macOS, or WSL where `import pyscf` works. Run the commands below in that environment.
- This lab PC: WSL micromamba env `flibe` already has PySCF. `mol_simulate.py` re-runs itself there if Windows Python cannot import PySCF. The env name `flibe` is only an interpreter. This repo does not import another project.

If you are not on that PC, ignore `flibe` and install PySCF yourself (conda-forge is the reliable route).

## 1. Check the published table

From this folder:

```text
python -m pytest tests -q
```

Expected: two passed tests. They read the eight JSON records. No GPU, no VQE, no PySCF.

Then, if you want one combined file:

```text
python scripts/combine_records.py nh3-sto3g-cas
```

That writes `data/corpus/` (gitignored). The per-geometry JSON files are the source.

## 2. Chemistry only (minutes)

```text
python main.py sweep
```

Four geometries, multiplicity 1 and 3. Hartree–Fock and CASCI. No VQE. Expect energies near the `E_HF` and `E_CASCI` columns on the README. Last digits can move with a PySCF version change.

`python main.py doctor` prints Aer devices. On a laptop it should list CPU. `GPU init ok` appears only when the GPU wheel is installed.

## 3. One VQE, or the whole grid

These are long. A singlet point is 12 qubits, about 76 angles, about 608 COBYLA evaluations, then a second hunt for the excited state. On the GPU run, one point was on the order of minutes. On a laptop CPU statevector it can be much longer. The README table does not depend on you finishing this.

CPU (default):

```text
python main.py vqe
```

That is one pyramidal singlet, ground state only, `R = 1.012` angstrom.

The full grid:

```text
python main.py all
```

Eight points, ground and one excited state each. Records land in `data/projects/nh3-sto3g-cas/`. A finished point (both VQD energies and the parameter arrays) is skipped on a later run. A ground-only checkpoint is resumed.

## 4. GPU

Only if `python gpu_simulator.py doctor` prints `GPU init ok`.

The GPU wheel on PyPI (`qiskit-aer-gpu` 0.15.1) does not install next to Qiskit 2.5.2. Use the pins in the comments of `requirements.txt` and the steps in [docs/GPU_SIMULATION.md](docs/GPU_SIMULATION.md). Then:

```text
export AMMONIA_USE_GPU=1
python main.py all
```

Windows PowerShell: `$env:AMMONIA_USE_GPU = "1"`.

`bash run.sh` on a Linux GPU machine sets that variable, runs doctor, a short bench, then `all`.

Without `AMMONIA_USE_GPU`, `all` / `vqe` / `vqd` use the CPU statevector. Sweep never uses the GPU.

## 5. What not to run

- Do not densify the 12-qubit Hamiltonian with `to_matrix()`.
- Do not point this at IBM hardware. Nothing here requests a QPU.
- Do not treat `E_VQD[1]` as CASCI root 1. The test fails if that gap closes by accident in a way that contradicts the published records, and the README states the gap on purpose.
- The folder `data/projects/nh3-sto3g-fci/` is not part of this study. It is gitignored if it exists locally.
