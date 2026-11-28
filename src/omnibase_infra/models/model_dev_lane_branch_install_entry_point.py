# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One entry point a dev-lane branch install writes into an installed dist-info (OMN-15920).

The file install copies source only, so a node a branch adds is on disk but absent
from the installed distribution's ``entry_points.txt``. The runtime finds nodes
through those entry points (``read_package_inventory`` iterates the ``onex.nodes``
group), so the install declares the entry point beside the files and the installer
writes it into the container's dist-info, never a hand edit.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, ConfigDict, field_validator

_DIST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_GROUP = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}$")
_VALUE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(:[A-Za-z_][A-Za-z0-9_.]*)?$")

PATH_PREFIX = "entry_points:"

# Run in the container with the runtime's own interpreter: dist, group, name, value
# are argv, never interpolated into the program text.
UPSERT_PROGRAM = """
import importlib.metadata as md, os, shutil, sys, tempfile
dist, group, name, value = sys.argv[1:5]
d = md.distribution(dist)
rel = next((f for f in (d.files or []) if f.name == "entry_points.txt"
            and f.parent.name.endswith(".dist-info")), None)
if rel is None:
    sys.exit("no entry_points.txt in the dist-info of " + dist)
path = str(d.locate_file(rel))
lines = open(path, encoding="utf-8").read().splitlines()
new = name + " = " + value
header = "[" + group + "]"
start = next((i for i, l in enumerate(lines) if l.strip() == header), None)
if start is None:
    if lines and lines[-1].strip():
        lines.append("")
    lines += [header, new]
else:
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("[")), len(lines))
    hit = next((i for i in range(start + 1, end)
                if lines[i].split("=", 1)[0].strip() == name), None)
    if hit is not None:
        lines[hit] = new
    else:
        at = end
        while at > start + 1 and not lines[at - 1].strip():
            at -= 1
        lines.insert(at, new)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as h:
    h.write("\\n".join(lines) + "\\n")
shutil.copymode(path, tmp)
os.replace(tmp, path)
"""

READ_PROGRAM = """
import importlib.metadata as md, sys
dist, group, name = sys.argv[1:4]
for ep in md.distribution(dist).entry_points:
    if ep.group == group and ep.name == name:
        print(ep.name + " = " + ep.value)
        break
"""


class ModelDevLaneBranchInstallEntryPoint(BaseModel):
    """``name = value`` in ``[group]`` of the installed ``dist`` entry_points.txt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dist: str
    group: str
    name: str
    value: str

    @field_validator("dist")
    @classmethod
    def _dist_ok(cls, value: str) -> str:
        if not _DIST.match(value):
            raise ValueError(f"distribution name {value!r} is not a plain name")
        return value

    @field_validator("group")
    @classmethod
    def _group_ok(cls, value: str) -> str:
        if not _GROUP.match(value):
            raise ValueError(f"entry point group {value!r} is not a plain name")
        return value

    @field_validator("name")
    @classmethod
    def _name_ok(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError(f"entry point name {value!r} is not a plain name")
        return value

    @field_validator("value")
    @classmethod
    def _value_ok(cls, value: str) -> str:
        if not _VALUE.match(value):
            raise ValueError(
                f"entry point value {value!r} must be module.path[:attribute]"
            )
        return value

    @property
    def line(self) -> str:
        """The entry as entry_points.txt writes it."""
        return f"{self.name} = {self.value}"

    @property
    def sha256(self) -> str:
        """Hash of the line: the expected side of the readback row."""
        return hashlib.sha256(self.line.encode("utf-8")).hexdigest()

    @property
    def path(self) -> str:
        """The readback row's path cell for this entry point."""
        return f"{PATH_PREFIX}{self.dist}:{self.group}:{self.name}"

    def upsert_argv(self, container: str) -> list[str]:
        return [
            "docker",
            "exec",
            "-u",
            "0",
            container,
            "/app/.venv/bin/python",
            "-c",
            UPSERT_PROGRAM,
            self.dist,
            self.group,
            self.name,
            self.value,
        ]

    def read_argv(self, container: str) -> list[str]:
        return [
            "docker",
            "exec",
            container,
            "/app/.venv/bin/python",
            "-c",
            READ_PROGRAM,
            self.dist,
            self.group,
            self.name,
        ]
