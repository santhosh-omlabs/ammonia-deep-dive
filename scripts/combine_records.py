"""Union JSON records from one or more data/projects/<name>/ folders.

Run from the repo root:

  python scripts/combine_records.py nh3-sto3g-cas
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_SCHEMA = "ammonia-deep-dive.record.v1"
_CORPUS_SCHEMA = "ammonia-deep-dive.corpus.v1"


def combine_records(projects: list[str], *, outfile: Path | None = None) -> Path:
    if not projects:
        raise ValueError("pass at least one project name.")
    records: list[dict[str, Any]] = []
    for name in projects:
        folder = _REPO / "data" / "projects" / name
        if not folder.is_dir():
            raise FileNotFoundError(f"missing project folder: {folder}")
        for path in sorted(folder.glob("*.json")):
            if path.name == "corpus.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != _SCHEMA:
                print(f"skip {path.name}: schema {data.get('schema')!r}")
                continue
            data["_file"] = str(path.relative_to(_REPO)).replace("\\", "/")
            records.append(data)
    if not records:
        raise RuntimeError("no v1 records found in those project folders.")
    body = {
        "schema": _CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "projects": list(projects),
        "n_records": len(records),
        "records": records,
    }
    if outfile is None:
        if len(projects) == 1:
            outfile = _REPO / "data" / "projects" / projects[0] / "corpus.json"
        else:
            joined = "_".join(projects)
            outdir = _REPO / "data" / "corpus"
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{joined}.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {outfile.relative_to(_REPO)}  ({len(records)} records)")
    return outfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine record JSON files from named project folders."
    )
    parser.add_argument(
        "projects",
        nargs="+",
        help="project folder names under data/projects/",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    combine_records(args.projects, outfile=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
