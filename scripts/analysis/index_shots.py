#!/usr/bin/env python3
"""
Build or inspect the SQLite index over the raw HDF5 shots.

The index stores every entry of a shot's ``globals`` group as a column, so
shots can be selected by experimental parameter without opening hundreds of
files. It is a cache: delete data/db/shots.db and rebuild at any time.

    python scripts/analysis/index_shots.py --build coarsening   # index a dataset
    python scripts/analysis/index_shots.py --list               # show recent shots
    python scripts/analysis/index_shots.py --columns            # show the schema

This replaces the former src/quantum_sim/analysis.py, which was a script
living inside the installable package and opened a database connection as an
import side effect.
"""

import argparse

from quantum_sim.config import PATHS, dataset, dataset_names
from quantum_sim.io import Database


def build(db, dataset_name: str) -> None:
    """Index every shot of a dataset, extending the schema to fit its globals."""
    ds = dataset(dataset_name)
    paths = ds.shot_paths("*.h5")
    if not paths:
        raise SystemExit(f"No shots found for dataset {dataset_name!r} in {ds.path}")

    print(f"indexing {len(paths)} shots from {ds.path}")
    fields = db.generate_column_structure(paths[0])
    db.commit_column_structure(fields)
    db.create_new_shot(paths)
    print(f"done: {db.Shot.select().count()} records in {db.path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", metavar="DATASET", choices=dataset_names(),
                        help="index this dataset from config/experiments.yaml")
    parser.add_argument("--list", action="store_true", help="print the most recent shots")
    parser.add_argument("--columns", action="store_true", help="print the table schema")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if not any((args.build, args.list, args.columns)):
        parser.error("choose at least one of --build, --list, --columns")

    print(f"database: {PATHS.database}")
    with Database() as db:
        if args.build:
            build(db, args.build)

        if args.columns:
            print("\ncolumns:")
            for name in sorted(db.Shot._meta.fields):
                print(f"  {name}")

        if args.list:
            print(f"\nmost recent {args.limit} shots:")
            query = db.Shot.select().order_by(db.Shot.shot_time.desc()).limit(args.limit)
            for shot in query:
                print(f"  {shot.shot_name}  num={shot.shot_num}  t={shot.shot_time}")


if __name__ == "__main__":
    main()
