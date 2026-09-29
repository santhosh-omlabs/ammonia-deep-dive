"""Aer statevector CPU/GPU primitives, plus RunPod SSH helpers (no API keys).

Use this from mol_simulate.simulate_molecule(use_gpu=...). On a RunPod NVIDIA
pod, use_gpu=True builds:

    AerSimulator(method="statevector", device="GPU")

and wraps it as EstimatorV2 / SamplerV2 so VQE/VQD stay the same algorithms.

This module never talks to the RunPod HTTP API. Optional env (SSH only):

    RUNPOD_SSH_HOST   pod IP or ssh.runpod.io
    RUNPOD_SSH_PORT   example 22123
    RUNPOD_SSH_USER   default root

  python gpu_simulator.py doctor
  python gpu_simulator.py ssh-cmd
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_DIAGNOSED = False


def _pkg_ver(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _aer_pkg_ver() -> str | None:
    return _pkg_ver("qiskit-aer-gpu") or _pkg_ver("qiskit-aer")


def _select_mig() -> str | None:
    """If the pod uses NVIDIA MIG and CUDA_VISIBLE_DEVICES is empty, pin it."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return os.environ["CUDA_VISIBLE_DEVICES"]
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "-L"], stderr=subprocess.STDOUT, timeout=8, text=True
        )
    except Exception:
        return None
    for line in out.splitlines():
        if "MIG-" in line and "UUID:" in line:
            uuid = line.split("UUID:")[-1].strip().rstrip(")")
            if uuid.startswith("MIG-"):
                os.environ["CUDA_VISIBLE_DEVICES"] = uuid
                print(f"[ok] CUDA_VISIBLE_DEVICES={uuid}")
                return uuid
    return None


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            timeout=8,
            text=True,
        )
        return out.strip() or None
    except Exception as exc:
        return f"(nvidia-smi failed: {type(exc).__name__}: {exc})"


def available_aer_devices() -> list[str]:
    try:
        from qiskit_aer import AerSimulator

        raw = AerSimulator().available_devices()
        return [str(x) for x in raw]
    except Exception:
        return []


def diagnose(*, probe_gpu: bool = True) -> dict[str, Any]:
    """Print runtime facts. Does not assume GPU acceleration is active."""
    _select_mig()
    devices = available_aer_devices()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "qiskit": _pkg_ver("qiskit"),
        "qiskit_aer": _aer_pkg_ver(),
        "qiskit_nature": _pkg_ver("qiskit-nature"),
        "qiskit_algorithms": _pkg_ver("qiskit-algorithms"),
        "pyscf": _pkg_ver("pyscf"),
        "aer_devices": devices,
        "nvidia_smi": _nvidia_smi(),
        "gpu_init_ok": False,
        "custatevec": None,
        "selected_device": None,
        "error": None,
    }
    print("----- Aer / GPU doctor -----")
    print(f"     python          {info['python']}")
    print(f"     qiskit          {info['qiskit']}")
    print(f"     qiskit-aer      {info['qiskit_aer']}")
    print(f"     qiskit-nature   {info['qiskit_nature']}")
    print(f"     qiskit-algs     {info['qiskit_algorithms']}")
    print(f"     pyscf           {info['pyscf']}")
    print(f"     Aer devices     {devices}")
    print(f"     NVIDIA GPU      {info['nvidia_smi'] or '(none detected)'}")
    if probe_gpu and any(d.upper() == "GPU" for d in devices):
        try:
            backend, cu = _aer_backend("GPU", seed=1)
            info["gpu_init_ok"] = True
            info["custatevec"] = cu
            info["selected_device"] = str(backend.options.device)
            print(f"     GPU init        ok  device={info['selected_device']}")
            print(f"     cuStateVec      {cu}")
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            print(f"     GPU init        FAILED  {info['error']}")
    elif probe_gpu:
        print("     GPU init        skipped (Aer has no GPU device)")
    print("-----")
    return info


def _probe(sim) -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.save_statevector()
    result = sim.run(qc, shots=1).result()
    if not result.success:
        raise RuntimeError(f"Aer GPU probe failed: {result.status}")
    _ = result.get_statevector()


def _aer_backend(device: str, *, seed: int | None, try_custatevec: bool = True):
    from qiskit_aer import AerSimulator

    device = device.upper()
    kw: dict[str, Any] = {
        "method": "statevector",
        "device": device,
        "precision": "double",
    }
    if seed is not None:
        kw["seed_simulator"] = int(seed)
    cu_used = False
    if device == "GPU" and try_custatevec:
        kw["cuStateVec_enable"] = True
        try:
            sim = AerSimulator(**kw)
            _probe(sim)
            return sim, True
        except Exception:
            kw["cuStateVec_enable"] = False
            cu_used = False
    sim = AerSimulator(**kw)
    if device == "GPU":
        _probe(sim)
    return sim, cu_used


class _TimedJob:
    def __init__(self, job, bucket: dict[str, float], key: str):
        self._job = job
        self._bucket = bucket
        self._key = key
        self._t0 = time.perf_counter()

    def result(self):
        try:
            return self._job.result()
        finally:
            self._bucket[self._key] = self._bucket.get(self._key, 0.0) + (
                time.perf_counter() - self._t0
            )
            nkey = self._key + "_calls"
            self._bucket[nkey] = self._bucket.get(nkey, 0.0) + 1.0

    def __getattr__(self, name: str):
        return getattr(self._job, name)


def _time_primitive(inner, bucket: dict[str, float], key: str):
    """Keep the real primitive type (VQE checks BaseEstimatorV2)."""
    orig = inner.run

    def run(*args, **kwargs):
        return _TimedJob(orig(*args, **kwargs), bucket, key)

    inner.run = run  # type: ignore[method-assign]
    return inner


@dataclass
class SimSession:
    use_gpu: bool
    device: str
    estimator: Any
    sampler: Any
    shot_estimator_factory: Any
    backend_info: dict[str, Any]
    profile: dict[str, float] = field(default_factory=dict)

    def estimate_shots(self, circuit, hamiltonian, shots: int) -> float:
        import numpy as np

        qc = circuit.decompose(reps=8)
        t0 = time.perf_counter()
        est = self.shot_estimator_factory()
        job = est.run([qc], [hamiltonian], shots=int(shots))
        value = float(np.real(job.result().values[0]))
        self.profile["shots_s"] = self.profile.get("shots_s", 0.0) + (
            time.perf_counter() - t0
        )
        self.profile["shots_calls"] = self.profile.get("shots_calls", 0.0) + 1.0
        return value

    def report(self) -> None:
        p = self.profile
        est_s = p.get("estimator_s", 0.0)
        sam_s = p.get("sampler_s", 0.0)
        shots_s = p.get("shots_s", 0.0)
        other = p.get("other_s", 0.0)
        total = est_s + sam_s + shots_s + other
        print("----- runtime split -----")
        print(f"     device           {self.device}  (use_gpu={self.use_gpu})")
        print(f"     estimator        {est_s:.2f} s  calls={int(p.get('estimator_calls', 0))}")
        print(f"     overlap sampler  {sam_s:.2f} s  calls={int(p.get('sampler_calls', 0))}")
        print(f"     Aer shots stamp  {shots_s:.2f} s  calls={int(p.get('shots_calls', 0))}")
        print(f"     chemistry/other  {other:.2f} s")
        if total > 0:
            print(
                f"     share            est {100 * est_s / total:.1f}%  "
                f"overlap {100 * sam_s / total:.1f}%  "
                f"shots {100 * shots_s / total:.1f}%  "
                f"other {100 * other / total:.1f}%"
            )
        nq = self.backend_info.get("n_qubits")
        if nq is not None and int(nq) <= 16 and self.use_gpu:
            print(
                "     note: at ~12 qubits the statevector is tiny; GPU may "
                "not beat CPU if Python/UCC construction dominates."
            )
        print("-----")


def _v1_overlap_sampler(backend=None):
    """ComputeUncompute on qiskit-algorithms 0.3 needs Terra BaseSampler.

    Aer Sampler is a different class and raises ValueError. Prefer
    BackendSampler on the Aer GPU backend; fall back to Terra Sampler.
    """
    from qiskit.primitives import Sampler as TerraSampler

    try:
        from qiskit.primitives import BaseSampler
    except ImportError:
        BaseSampler = object  # type: ignore[misc, assignment]
    if backend is not None:
        try:
            from qiskit.primitives import BackendSampler

            samp = BackendSampler(backend, skip_transpilation=False)
            if isinstance(samp, BaseSampler):
                print("[ok] VQD overlap  BackendSampler (Aer GPU)")
                return samp
        except Exception as exc:
            print(f"     BackendSampler unused ({type(exc).__name__}: {exc})")
    print("[ok] VQD overlap  Terra Sampler (CPU, BaseSampler)")
    return TerraSampler()


def _algorithms_want_v2() -> bool:
    """qiskit-algorithms 0.3 talks V1 primitives; 0.4+ talks V2."""
    try:
        import inspect

        from qiskit_algorithms import VQE

        ann = inspect.signature(VQE.__init__).parameters["estimator"].annotation
        return "V2" in str(ann)
    except Exception:
        ver = _pkg_ver("qiskit-algorithms") or "0"
        return ver.split(".")[:2] >= ["0", "4"]


def make_session(use_gpu: bool, *, seed: int | None = 42) -> SimSession:
    """CPU or GPU primitives for VQE/VQD. Picks V1 vs V2 from qiskit-algorithms."""
    global _DIAGNOSED
    if use_gpu:
        _select_mig()
    if not _DIAGNOSED:
        diagnose(probe_gpu=bool(use_gpu))
        _DIAGNOSED = True

    profile: dict[str, float] = {}
    info: dict[str, Any] = {
        "use_gpu": bool(use_gpu),
        "seed": seed,
        "qiskit": _pkg_ver("qiskit"),
        "qiskit_aer": _aer_pkg_ver(),
    }

    want_v2 = _algorithms_want_v2()
    backend_opts = {
        "method": "statevector",
        "device": "GPU" if use_gpu else "CPU",
        "precision": "double",
    }
    if seed is not None:
        backend_opts["seed_simulator"] = int(seed)

    if not use_gpu:
        if want_v2:
            from qiskit.primitives import StatevectorEstimator, StatevectorSampler

            est = StatevectorEstimator()
            samp = StatevectorSampler()
            prim = "StatevectorEstimator/StatevectorSampler (V2)"
        else:
            from qiskit.primitives import Estimator, Sampler

            est = Estimator()
            samp = Sampler()
            prim = "Estimator/Sampler (V1, exact)"
        from qiskit_aer.primitives import Estimator as AerEstimatorV1

        def _shots():
            return AerEstimatorV1(
                backend_options={"method": "statevector", "device": "CPU"}
            )

        info.update({"device": "CPU", "primitive": prim, "custatevec": False})
        print(f"[ok] backend      CPU  {prim}")
        return SimSession(
            use_gpu=False,
            device="CPU",
            estimator=est,
            sampler=samp,
            shot_estimator_factory=_shots,
            backend_info=info,
            profile=profile,
        )

    devices = available_aer_devices()
    if not any(d.upper() == "GPU" for d in devices):
        raise RuntimeError(
            "use_gpu=True but Aer has no GPU device. "
            "On RunPod: pip install qiskit-aer-gpu (needs Qiskit 1.x; "
            "see requirements.txt). Then python gpu_simulator.py doctor."
        )

    from qiskit_aer.primitives import Estimator as AerEstimatorV1

    backend, cu = _aer_backend("GPU", seed=seed)
    device = str(getattr(backend.options, "device", "GPU"))
    backend_opts["device"] = "GPU"
    backend_opts["cuStateVec_enable"] = bool(cu)

    if want_v2:
        from qiskit_aer.primitives import EstimatorV2, SamplerV2

        est = EstimatorV2.from_backend(
            backend,
            options={
                "default_precision": 0.0,
                "backend_options": dict(backend_opts),
            },
        )
        samp = SamplerV2.from_backend(
            backend,
            options={
                "backend_options": dict(backend_opts),
                "run_options": {"shots": 1024},
            },
        )
        prim = "Aer EstimatorV2 + SamplerV2"
    else:
        est = AerEstimatorV1(
            backend_options=dict(backend_opts),
            approximation=True,
            skip_transpilation=False,
        )
        samp = _v1_overlap_sampler(backend)
        prim = "Aer Estimator V1 GPU + BaseSampler overlap"

    def _shots():
        return AerEstimatorV1(backend_options=dict(backend_opts))

    info.update(
        {
            "device": device,
            "primitive": prim,
            "custatevec": bool(cu),
            "aer_devices": devices,
            "nvidia_smi": _nvidia_smi(),
        }
    )
    print(
        f"[ok] backend      GPU  AerSimulator(method=statevector)  "
        f"cuStateVec={cu}  {prim}"
    )
    return SimSession(
        use_gpu=True,
        device=device,
        estimator=est,
        sampler=samp,
        shot_estimator_factory=_shots,
        backend_info=info,
        profile=profile,
    )


def runpod_ssh_parts() -> dict[str, str]:
    return {
        "host": os.environ.get("RUNPOD_SSH_HOST", "").strip(),
        "port": os.environ.get("RUNPOD_SSH_PORT", "").strip() or "22",
        "user": os.environ.get("RUNPOD_SSH_USER", "").strip() or "root",
    }


def print_runpod_needed() -> None:
    print("----- what I need from you (RunPod) -----")
    print("     Do not send a password or API secret.")
    print("     After you start a GPU pod, paste:")
    print("       1. SSH host  (IP or ssh.runpod.io)")
    print("       2. SSH port  (often not 22)")
    print("       3. SSH user  (usually root)")
    print("     Optional: the Jupyter URL if you prefer a notebook instead of SSH.")
    print("     Optional: mounted volume path if the pod has one (e.g. /workspace).")
    print("     Then on this machine (example):")
    parts = runpod_ssh_parts()
    if parts["host"]:
        print("     env already set:")
        print(f"       RUNPOD_SSH_HOST={parts['host']}")
        print(f"       RUNPOD_SSH_PORT={parts['port']}")
        print(f"       RUNPOD_SSH_USER={parts['user']}")
        print_ssh_commands()
    else:
        print("     setx RUNPOD_SSH_HOST  <host>")
        print("     setx RUNPOD_SSH_PORT  <port>")
        print("     python gpu_simulator.py ssh-cmd")
    print("-----")


def print_ssh_commands() -> None:
    p = runpod_ssh_parts()
    if not p["host"]:
        print("     RUNPOD_SSH_HOST is empty. Start a pod, then set the env vars.")
        return
    remote = f"{p['user']}@{p['host']}"
    port = p["port"]
    print(f"     ssh -p {port} {remote}")
    print(
        f"     scp -P {port} -r . {remote}:/workspace/ammonia-deep-dive"
        "   (run from this repo on a machine that has scp)"
    )
    print(
        f"     ssh -p {port} {remote} "
        "'cd /workspace/ammonia-deep-dive && python gpu_simulator.py doctor'"
    )


def _main(argv: list[str]) -> int:
    cmd = (argv[1] if len(argv) > 1 else "doctor").lower()
    if cmd in ("doctor", "check", "diag"):
        diagnose(probe_gpu=True)
        return 0
    if cmd in ("ssh-cmd", "runpod", "needed"):
        print_runpod_needed()
        return 0
    print("usage: python gpu_simulator.py [doctor|ssh-cmd]")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
