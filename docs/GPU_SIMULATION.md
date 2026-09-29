# GPU simulation (Qiskit Aer + RunPod)

This file is the operating note for NVIDIA GPU runs. Chemistry stays in `mol_simulate.py`. GPU only replaces the statevector primitive.

This repo does **not** call the RunPod HTTP API and does **not** need a RunPod password or API secret. Access is SSH plus files on the pod disk.

---

## How the pieces connect

```text
main.py  (AMMONIA_USE_GPU=1)
  -> mol_simulate.simulate_molecule(..., use_gpu=True)
       -> gpu_simulator.make_session(use_gpu=True)
            -> diagnose()  (versions, Aer devices, GPU probe)
            -> pin MIG UUID if nvidia-smi -L shows MIG-...
            -> AerSimulator(method="statevector", device="GPU")
            -> Estimator for VQE/VQD energy  (GPU, exact/approximation)
            -> BaseSampler for VQD overlap   (BackendSampler on that Aer GPU)
            -> Aer Estimator for optional shot stamp
       -> VQE then VQD on the same mapped Hamiltonian
       -> JSON + npz under data/projects/<project>/
```

`gpu_simulator.py` never builds NH3 geometries or CAS. `mol_simulate.py` never talks to RunPod. The only flag is `use_gpu`.

On the GPU wheel used in the 2026 NH3 run (`qiskit-aer-gpu` 0.15.1 + Qiskit 1.4.2):

```text
qiskit-algorithms 0.3  wants primitive V1  (run(circuits, observables, params))
qiskit-aer.primitives.Sampler  is NOT Terra BaseSampler
  -> VQD overlap uses qiskit.primitives.BackendSampler(Aer GPU backend)
```

Laptop path (`use_gpu=False`, Qiskit 2.x) still uses Terra `StatevectorEstimator` / `StatevectorSampler`.

---

## One-time setup

Do these once per machine or once per new pod image. Do not mix them into every later run.

### A. RunPod account and SSH key (once on your PC)

1. Create a RunPod account (the email on the account is the login; it is not the SSH username).
2. On the PC that will `scp`/`ssh`, keep an Ed25519 key. Public key is the `.pub` line only. Never paste the private key into RunPod or chat.
3. RunPod console: **Settings -> SSH Public Keys**. Paste the public key. Save.
4. Some images ignore account keys until you also put the same line in the pod's `~/.ssh/authorized_keys` (see first connect below).

SSH username on these pods is almost always `root`. Linux uid is 0. There is no extra "root user id".

### B. Choose a GPU pod template (once per experiment type)

Use a **Linux x86_64** NVIDIA GPU pod. This project needs:

- CUDA driver visible to `nvidia-smi`
- Python 3.10+ (3.12 worked)
- Ability to create a venv and `pip install`
- **SSH port exposed** (TCP). Jupyter 8888 is optional.

What worked for the NH3 dataset: official/community **GPU Jupyter** style image, CUDA toolkit on the host, SSH + Jupyter. Example hardware stamp: NVIDIA RTX PRO 6000, driver 580, CUDA 13 reported by `nvidia-smi`, MIG slice enabled.

`qiskit-aer-gpu` on PyPI (0.15.1) is a **CUDA 12** wheel. A CUDA 13 *driver* is usually fine; do not require a CUDA 13 Aer wheel that does not exist. CUDA 11 hosts: `pip install qiskit-aer-gpu-cu11` instead.

Pick a template with SSH. Do not add Kubernetes, Terraform, or serverless.

### C. Python environment on the pod (once per pod disk)

From the repo root on the pod (`/workspace/ammonia-deep-dive` is a good path):

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install numpy==1.26.4 scipy==1.13.1 pyscf
python -m pip install qiskit==1.4.2 qiskit-nature==0.7.2 qiskit-algorithms==0.3.1
python -m pip install qiskit-aer-gpu==0.15.1
python gpu_simulator.py doctor
```

`doctor` must show Aer devices including `GPU` and `GPU init ok`. cuStateVec `True` is good; if Aer ignores it, energy still runs on GPU.

Why these pins (do not "upgrade everything"):

| Constraint | Why |
|---|---|
| `qiskit-aer-gpu` 0.15.1 is the newest GPU wheel on PyPI | 0.17 CPU Aer exists; GPU wheel stopped at 0.15.1 |
| GPU wheel needs Qiskit 1.x (`convert_to_target`) | Qiskit 2.5.2 + that wheel ImportError |
| Nature 0.8 wants `qiskit-algorithms>=0.4` and Qiskit 2 | Cannot sit on the GPU wheel |
| Nature 0.7.2 uses `np.in1d` | NumPy 2.5 removed it; pin NumPy 1.26 |

Laptop records used Qiskit 2.5.2 / Nature 0.8.0. GPU records used 1.4.2 / 0.7.2. Same NH3 loop and CAS Hamiltonian construction; not the same wheel.

Debian/Ubuntu images: `pip install` into **venv**, not system Python (`externally-managed-environment`).

### D. First SSH if account keys did not inject

Web Terminal on the pod (no SSH yet):

```text
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA... comment' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Paste **one public-key line**. Ctrl+V often fails in RunPod's terminal; use Ctrl+Shift+V or right-click paste.

---

## Recurring workflow (a later GPU run)

This is the loop after the venv exists on a **persistent volume**.

1. Start the GPU pod in the RunPod UI. Wait until logs say the pod is ready.
2. Copy **SSH host and port** from Connect (example: `81.27.69.178` port `30157`). Internal `:22` is not the public port.
3. From the PC:

```text
ssh -p <port> -i ~/.ssh/id_ed25519 root@<host>
```

4. On the pod:

```text
cd /workspace/ammonia-deep-dive
source .venv/bin/activate
python gpu_simulator.py doctor
python main.py all
```

On the pod, `export AMMONIA_USE_GPU=1` before `all`, `vqe`, `vqd`, or `bench`. `run.sh` and `start_all.sh` set that. Without the variable, those commands use the CPU statevector. Sweep (CASCI only) never needs the GPU. The dataset command is `python main.py all` (CAS, 12 qubits, UCC doubles).

5. When the grid finishes, copy `data/projects/nh3-sto3g-cas/` to the PC, then **stop the pod** (billing continues while it runs).

6. Optional helpers from this repo:

```text
python gpu_simulator.py ssh-cmd
bash start_all.sh          # nohup python -u main.py all > all.log
python main.py bench       # short CPU then GPU VQE, not the full COBYLA budget
```

---

## Reconnect: stopped pod, new port, new pod

RunPod gives a **new SSH port** (sometimes new IP) every start. The username stays `root`.

### Same disk, pod stopped then started

Volume still has `.venv` and `data/projects/`. You only need the new host:port.

```text
ssh -p <NEW_PORT> -i ~/.ssh/id_ed25519 root@<host>
cd /workspace/ammonia-deep-dive
source .venv/bin/activate
python gpu_simulator.py doctor
python main.py all
```

`main.py all` resumes: complete VQD records (energies **and** `vqd_theta_*` in the npz) are skipped; ground-only checkpoints resume VQE theta and continue VQD.

If SSH is `Permission denied (publickey)`, Web Terminal and append the public key to `authorized_keys` again (some images reset `/root` but keep `/workspace`).

### New GPU pod (new disk)

Treat as one-time C plus copy the repo:

```text
scp -P <port> -i ~/.ssh/id_ed25519 -r . root@<host>:/workspace/ammonia-deep-dive
```

Avoid Windows paths with spaces as the scp destination; copy to `/workspace/...` on the pod. Then recreate the venv (section C). Copy old `data/projects/` onto the new disk if you want resume.

### New SSH key

Generate a key, add the **public** line to RunPod settings **and** `authorized_keys`, connect with `-i` pointing at that private key. Restart the pod after changing account keys if injection still fails.

### MIG GPUs

If `nvidia-smi -L` shows a `MIG-...` UUID, `gpu_simulator._select_mig()` sets `CUDA_VISIBLE_DEVICES` when it is empty. If Aer has no GPU device, export that UUID yourself and rerun `doctor`.

---

## Mistakes to avoid (this project actually hit these)

1. **Assume GPU Aer is active** because `nvidia-smi` works. Always `python gpu_simulator.py doctor`. Empty Aer devices = wrong wheel or Qiskit/Aer mix.
2. **Install `qiskit-aer` 0.17 and `qiskit-aer-gpu` 0.15 together.** Uninstall CPU Aer; one GPU build only.
3. **Qiskit 2.5 + aer-gpu 0.15.** Use the 1.4.2 stack on the pod.
4. **NumPy 2.x with Nature 0.7.2.** `np.in1d` is gone; Bravyi-Kitaev mapping crashes.
5. **Pass Aer `Sampler` into VQD `ComputeUncompute` on algorithms 0.3.** It is not `BaseSampler`. Use `BackendSampler` (this is what `gpu_simulator` does).
6. **`skip_transpilation=True` on UCC.** Aer then sees `EvolvedOps` and dies. Leave transpilation on.
7. **`nfev = int(array or 0)` after VQD.** VQD returns an array of eval counts; that `or` raises. Unpack with `_as_int`. A point is complete only if `e_vqd` **and** `vqd_theta_*` exist, or the next `all` will skip a hunt that never saved thetas.
8. **12 qubits does not saturate a datacenter GPU.** Statevector is 2^12 amplitudes. GPU still helped the VQE/VQD loop (~2 min vs long CPU hunts); PySCF/CASCI stays on CPU. Low GPU util is expected.
9. **`pkill -f python` or `pkill -f main.py` from SSH.** The pattern can kill the SSH session itself. Kill the numeric PID of `python -u main.py all`.
10. **Public SSH port vs container 22.** Use the Connect panel host:port.
11. **Do not send RunPod passwords or API keys** for this workflow.
12. **Leave the pod running after the job.** Stop it once data is copied.

---

## What GPU accelerates vs what it does not

```text
PySCF SCF / CASCI / integrals     CPU, seconds
Fermion-to-qubit map              CPU, seconds
UCC circuit + <psi|H|psi> loop    Aer GPU (this is the long part)
VQD overlap circuits              BackendSampler on Aer GPU
4096-shot stamp                   Aer GPU Estimator
JSON/npz write                    CPU
```

NH3 VQD k=2, 12 qubits, UCC-D ~76 parameters, COBYLA ~608 evals per hunt: full 8-point grid on this GPU was about **one hour of VQD** after ground VQE was already checkpointed, versus **many hours** on a laptop CPU (sleep/shutdown risk). Ground VQD[0] matched CASCI to ~1e-4 hartree. VQD[1] stayed well above the CASCI excited root (ansatz + overlap penalty, not a GPU bug).
