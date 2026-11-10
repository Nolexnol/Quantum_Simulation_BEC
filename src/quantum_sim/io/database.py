"""SQLite index over the raw HDF5 shot files."""

import functools
import json
import os
from datetime import datetime

import h5py
import numpy as np
from peewee import (
    BooleanField,
    CharField,
    DoubleField,
    IntegerField,
    Model,
    SqliteDatabase,
    TimestampField,
    make_snake_case,
)
from playhouse.migrate import SqliteMigrator, migrate
from playhouse.reflection import generate_models
from playhouse.sqlite_ext import JSONField
from tqdm import tqdm

from ..config import PATHS


def default_database_path():
    """Location of the shot index, from :mod:`quantum_sim.config`.

    Was hard-coded to an absolute path under a single user's home directory,
    which made the module unusable on any other machine.
    """
    PATHS.db.mkdir(parents=True, exist_ok=True)
    return str(PATHS.database)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

np_json_dumps = functools.partial(json.dumps, cls=NumpyArrayEncoder)


def walk_dir(_dir):
    files = []

    for root, _dirs, _files in os.walk(_dir, topdown=False):
        for name in _files:
            if name.endswith(".h5"):
                files.append(os.path.join(root, name))

    return files


class Database:
    """SQLite index over the raw HDF5 shots.

    The index stores every entry of a shot's ``globals`` group as a column, so
    shots can be selected by experimental parameter without opening 750 files.
    Pass ``path`` to work with an index other than the project default.
    """

    def __init__(self, path=None, verbose=True):
        self.Shot = None
        self.cn = self.column_name = {}
        self.path = str(path) if path else default_database_path()
        self.db = SqliteDatabase(self.path)
        self.db.connect()

        self.get_shot_model()

        if self.Shot and verbose:
            print(
                f"Database loaded from {self.path}. "
                f"Number of records: {self.Shot.select().count()}."
            )

    def close(self):
        """Close the connection. Supports use as a context manager."""
        if not self.db.is_closed():
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def make_table(self):
        class _BaseModel(Model):
            class Meta:
                database = self.db

        class Shot(_BaseModel):
            shot_num = IntegerField()
            shot_time = TimestampField(index=True)
            script = CharField()
            shot_path = CharField()
            shot_name = CharField(unique=True, index=True)

        Shot.drop_table(safe=True)
        self.db.create_tables([Shot])

        return Shot

    def generate_column_structure(self, shot_file):
        f = h5py.File(shot_file, 'r')

        self.cn = {}
        fields = {}

        for g_ in f['globals'].attrs.keys():
            item = np.array(f['globals'].attrs[g_])

            g = make_snake_case(g_)

            self.cn[g_] = g

            if not item.shape:
                if item.dtype.name == 'int32':
                    fields[g] = IntegerField(null=True)
                elif item.dtype.name == 'bool':
                    fields[g] = BooleanField(null=True)
                elif item.dtype.name == 'float64':
                    fields[g] = DoubleField(null=True)
                elif item.dtype.name.startswith('str'):
                    fields[g] = CharField(null=True)
            elif item.shape or item.dtype.name.startswith('object'):
                fields[g] = JSONField(json_dumps=np_json_dumps, null=True)

        return fields

    def commit_column_structure(self, fields):
        migrator = SqliteMigrator(self.db)

        migrate(
            *[migrator.add_column('shot', k, f) for k, f in fields.items() if k not in self.Shot._meta.fields ],
        )

        models = generate_models(self.db)
        self.Shot = models['shot']

    def get_shot_model(self):
        models = generate_models(self.db)
        if 'shot' not in models:
            print(
                "Shot model doesn't exist. Proceed to create a skeleton. \n"
                "You need to run `generate_column_structure` and `commit_column_structure` \n"
                "to recreate the table structure."
            )
            self.Shot = self.make_table()
        else:
            self.Shot = models['shot']

    def _column_name(self, col):
        if col in self.cn:
            return self.cn[col]
        else:
            self.cn[col] = make_snake_case(col)
            return self.cn[col]

    def create_new_shot(self, shot_file_paths, on_conflict='ignore'):
        if isinstance(shot_file_paths, (str, os.PathLike)):
            # `type(x) == str` missed pathlib.Path, so a single Path was
            # iterated character by character instead of indexed as one shot.
            shot_file_paths = [shot_file_paths]
        shot_file_paths = [os.fspath(p) for p in shot_file_paths]

        for fn in tqdm(shot_file_paths):
            shot_name = os.path.basename(fn)
            if self.Shot.select().where(self.Shot.shot_name == shot_name):
                continue

            try:
                f = h5py.File(fn, "r")
            except OSError:
                print(f"Warning: Unable to open {fn}, is it corrupted?")
                continue

            if "data" not in f:
                continue

            shot_time = datetime.strptime(f.attrs['run time'], "%Y%m%dT%H%M%S").timestamp()

            g = {}

            for k in f["globals"].attrs.keys():
                if self._column_name(k) in self.Shot._meta.fields.keys():
                    val = np.asarray(f['globals'].attrs[k])
                    if not val.shape and not val:
                        # null entry
                        continue
                    g[self._column_name(k)] = f['globals'].attrs[k]

            q = self.Shot.insert(
                shot_num=f.attrs['sequence_index'],
                shot_time=shot_time,
                script=f.attrs['script_basename'],
                shot_path=fn,
                shot_name=shot_name,
                **g
            ).on_conflict(on_conflict)

            q.execute()
