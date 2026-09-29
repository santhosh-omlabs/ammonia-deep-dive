# Docs

Read in this order.

1. [../README.md](../README.md) — question, eight-point table, limits.
2. [../GUIDE.md](../GUIDE.md) — install, pytest, sweep, when a GPU is worth turning on.
3. [MOL_SIMULATE.md](MOL_SIMULATE.md) — what `simulate_molecule` does, argument by argument.
4. [INTUITION.md](INTUITION.md) — pyramid, active space, why the excited number sits high.
5. [RESEARCH_NOTE.md](RESEARCH_NOTE.md) — the GPU run, and the full-space attempt that was stopped.
6. [GPU_SIMULATION.md](GPU_SIMULATION.md) — only if you are putting this on an NVIDIA machine.

`data/projects/nh3-sto3g-cas/` holds the eight JSON+npz records. `python scripts/combine_records.py nh3-sto3g-cas` stacks the JSON.
