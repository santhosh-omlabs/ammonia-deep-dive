# Research note

Lab book. The table on the README is this run. I have not rerun the eight VQD points since copying the records off the GPU machine.

## What I wanted

One ammonia grid I could compare to a classical active-space answer: a few bond lengths, pyramid and plane, singlet and triplet. STO-3G, nitrogen 1s frozen, six spatial orbitals, Bravyi–Kitaev, UCC doubles, VQE then VQD with two states. CASCI in that space is the reference. Full FCI of all orbitals is a different Hamiltonian, so I did not score VQE against it.

## What finished

NVIDIA GPU, Qiskit Aer statevector, `qiskit-aer-gpu` 0.15.1 with Qiskit 1.4.2, Nature 0.7.2, algorithms 0.3.1. Eight records in `data/projects/nh3-sto3g-cas/`.

Ground VQD landed on CASCI (within about a millihartree, usually closer). The excited VQD number did not. On the pyramidal R=0.900 angstrom singlet, CASCI root 1 is -54.780945 hartree and `E_VQD[1]` is -52.534017 hartree. The other seven points are the same kind of miss, smallest gap about 0.28 hartree (pyramidal R=1.120 angstrom, triplet). I do not read that as a GPU bug. The cost being minimized is energy plus an overlap penalty, on a doubles-only ansatz. The log line is a real number on `H_cas`. It is not the CASCI excited root.

Chemistry (SCF, CASCI) was seconds. The wall time was COBYLA calling a 12-qubit statevector a few hundred times per hunt. The GPU made that loop short enough to finish the grid. Utilization on the card stayed low because 2^12 amplitudes is a small kernel next to the Python that builds the circuit.

## What I stopped

I tried the same grid with no active-space cut: all eight spatial orbitals, 16 qubits, UCC singles and doubles, scored against spin-filtered FCI. The first geometry's ground state came within about 0.01 hartree of FCI. The excited hunt sat about 1.3 hartree high and each evaluation was mostly CPU circuit work, not GPU linear algebra. I stopped the machine. Those files are not in this repo. The study is the CAS grid.

## Environment note

Laptop checks use the Qiskit 2.5 pins in `requirements.txt`. The numbers in the JSON files were produced with the 1.4 GPU pins. I would not mix the two installs in one virtualenv.
