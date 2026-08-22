"""Adapter version registry: vNNNN dirs with meta.json + a `current` pointer.

Layout under one directory:
    v0001/, v0002/, ...   adapter files copied from the candidate + meta.json
    current               symlink to the latest version (fallback: current.json)
``v0000`` is the implicit base model: path None, no directory on disk.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Union

from .types import AdapterVersion, UpdateCandidate

_VERSION_RE = re.compile(r"^v(\d{4,})$")


class Registry:
    def __init__(self, dir: str):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)

    def base(self) -> AdapterVersion:
        """The implicit base version (no adapter weights)."""
        return AdapterVersion("v0000", None, None)

    def publish(
        self,
        candidate: Union[UpdateCandidate, str],
        parent: str,
        provenance: list[str],
    ) -> AdapterVersion:
        """Copy a candidate dir into the next vNNNN and point `current` at it.

        `candidate` is an UpdateCandidate or a path to an adapter dir; its
        contents are copied to ``{dir}/vNNNN`` and a ``meta.json``
        ({name, parent, provenance, created_from}) is written there.
        """
        src = candidate.adapter_path if isinstance(candidate, UpdateCandidate) else candidate
        nums = [int(_VERSION_RE.match(n).group(1)) for n in self._names()]
        name = f"v{max(nums, default=0) + 1:04d}"
        dst = os.path.join(self.dir, name)
        if src is not None and os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(dst, exist_ok=True)
        meta = {
            "name": name,
            "parent": parent,
            "provenance": list(provenance),
            "created_from": src,
        }
        with open(os.path.join(dst, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._set_current(name)
        return AdapterVersion(name=name, path=dst, parent=parent, provenance=list(provenance))

    def current(self) -> AdapterVersion:
        """The version `current` points at (base if nothing was published)."""
        return self.get(self._current_name())

    def get(self, name: str) -> AdapterVersion:
        if name == "v0000":
            return self.base()
        path = os.path.join(self.dir, name)
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        return AdapterVersion(
            name=name,
            path=path,
            parent=meta.get("parent"),
            provenance=list(meta.get("provenance", [])),
        )

    def history(self) -> list[AdapterVersion]:
        """All versions, base first, in publish order."""
        return [self.base()] + [self.get(n) for n in self._names()]

    def _names(self) -> list[str]:
        return sorted(
            n
            for n in os.listdir(self.dir)
            if _VERSION_RE.match(n) and os.path.isdir(os.path.join(self.dir, n))
        )

    def _set_current(self, name: str) -> None:
        link = os.path.join(self.dir, "current")
        pointer = os.path.join(self.dir, "current.json")
        try:
            if os.path.islink(link):
                os.remove(link)
            os.symlink(name, link)
            if os.path.exists(pointer):
                os.remove(pointer)
        except OSError:
            with open(pointer, "w") as f:
                json.dump({"name": name}, f)

    def _current_name(self) -> str:
        link = os.path.join(self.dir, "current")
        if os.path.islink(link):
            return os.path.basename(os.readlink(link))
        pointer = os.path.join(self.dir, "current.json")
        if os.path.exists(pointer):
            with open(pointer) as f:
                return json.load(f)["name"]
        return "v0000"
