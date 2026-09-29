# Data

`projects/nh3-sto3g-cas/` is the study. Eight JSON files and matching `.npz` files. JSON holds energies, identity, and run metadata. The `.npz` holds parameter vectors and arrays needed to resume or rebuild the state. Both are committed so a clone can check the README table without rerunning VQE.

`python scripts/combine_records.py nh3-sto3g-cas` writes a corpus under `data/corpus/`. That output is gitignored.

`projects/nh3-sto3g-fci/` is not part of the study. If a local copy exists, it stays gitignored.
