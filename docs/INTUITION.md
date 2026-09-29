# Intuition — ammonia electronic structure (VQE/VQD)

The numbers and the limits are on the README. This page is only the picture.

---

## Ammonia is a pyramid that can flatten

NH3 is a nitrogen with three hydrogens. The usual shape is a **pyramid**. If you push the hydrogens into one plane, you get a **planar** geometry. This repo asks: at a few bond lengths and at those two shapes, what are the electronic energies (ground and a first excited stamp) in a small model (STO-3G, active space).

The simulator never "sees" a smell. It sees coordinates in Angstrom and a charge/spin.

---

## Spin: singlet vs triplet

**Multiplicity 1 (singlet):** paired electrons, the usual closed-shell picture (RHF).
**Multiplicity 3 (triplet):** two unpaired electrons of the same spin (ROHF).

They are different Hamiltonians. Do not mix their energies in one sentence without saying which spin.

---

## CAS is a small stage, not the whole theater

Full quantum chemistry on all orbitals is expensive. This code **freezes** the nitrogen 1s core, keeps a handful of valence orbitals (six spatial -> twelve qubits), and solves that **active space** classically (CASCI) as the truth for VQE.

Analogy: you rehearse a scene with six actors, not the whole city. The scene can be exact **for those actors**. It is not the movie of the city. Full FCI on all MOs is a different play.

---

## VQE is turning knobs on a circuit

The ansatz is a circuit with knobs `theta` (UCC doubles here). Hartree-Fock is `theta = 0`. VQE asks a classical optimizer (COBYLA) to turn knobs to lower

```text
E(theta)  =  < psi(theta) | H_cas | psi(theta) >
```

English: average energy of the prepared state. If VQE hits CASCI, the knobs found the CAS ground state. If it stays at Hartree-Fock, the hunt was too short or the first step was too large (that is why `rhobeg` is 0.05, not 1).

---

## VQD is "do it again, but not the same state"

For an excited stamp, VQD minimizes energy **plus** a penalty if the new state overlaps a previous one (weight `beta`).

```text
cost_k(theta)  =  E(theta)  +  beta * |< psi(theta) | psi_previous >|^2
```

English: stay low energy, stay unlike the ground circuit. In this dataset the ground VQD energy sat on CASCI; the first excited stamp often stayed **well above** the CASCI excited root. That means the penalty/ansatz hunt did not find that CASCI state. It is still a defined number on the same `H_cas`, not a failed GPU.

---

## GPU is a faster calculator, not a different molecule

The chemistry (PySCF, CAS, mapping) is classical CPU. The GPU (Aer statevector on RunPod) repeats "apply circuit, measure energy" thousands of times. Twelve qubits is a **tiny** statevector. The GPU still shortened the long COBYLA loops versus a laptop that sleeps after hours. It does not make STO-3G into a lab spectrum.

---

## Where the analogies break

- A pyramid of balls is not the Born-Oppenheimer potential surface.
- "Knobs" are not experimental laser settings; they are circuit parameters.
- "Excited" here is VQD on the mapped CAS H, not a measured UV peak.
