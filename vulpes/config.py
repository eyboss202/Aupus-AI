"""Strict YAML <-> dataclass config handling shared by every part of Vulpes.

Why strict: a typo in a YAML key (``n_layer`` instead of ``n_layers``) would
otherwise be silently ignored and a run would quietly use the default value.
For long training runs that is an expensive mistake, so unknown keys are an
error, and every value is type-checked against the dataclass annotation.

Why save the resolved config: every run writes the *final* config (after
defaults and command-line overrides are applied) next to its checkpoints, so
any checkpoint can be reproduced or loaded without guessing what produced it.
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class _Loader(yaml.SafeLoader):
    """SafeLoader that also recognises YAML 1.2 floats such as ``3e-4``.

    PyYAML implements YAML 1.1, where a float needs a dot (``3.0e-4``) and
    ``3e-4`` is loaded as the *string* "3e-4". Learning rates are almost always
    written that way, so we add the YAML 1.2 float pattern.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str | typing.IO[str]) -> Any:
    return yaml.load(text, Loader=_Loader)  # SafeLoader subclass: never builds arbitrary objects


class ConfigError(ValueError):
    """Raised when a config file does not match its dataclass schema."""


def _check_type(name: str, value: Any, expected: Any) -> Any:
    """Validate (and lightly coerce) one value against a type annotation.

    Only the types configs actually use are supported: bool, int, float, str,
    ``X | None``, ``list[X]`` and nested dataclasses. ``int`` is accepted where
    ``float`` is expected (YAML writes ``1`` not ``1.0``), but ``bool`` is never
    accepted as a number even though ``bool`` subclasses ``int`` in Python.
    """
    origin = typing.get_origin(expected)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(expected)
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in (a for a in args if a is not type(None)):
            try:
                return _check_type(name, value, arg)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError("; ".join(errors))
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{name}: expected a list, got {type(value).__name__}")
        (item_type,) = typing.get_args(expected)
        return [_check_type(f"{name}[{i}]", v, item_type) for i, v in enumerate(value)]
    if dataclasses.is_dataclass(expected):
        if not isinstance(value, dict):
            raise ConfigError(f"{name}: expected a mapping, got {type(value).__name__}")
        return from_dict(expected, value, _prefix=f"{name}.")
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{name}: expected float, got {value!r}")
        return float(value)
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name}: expected int, got {value!r}")
        return value
    if expected in (bool, str):
        if not isinstance(value, expected):
            raise ConfigError(f"{name}: expected {expected.__name__}, got {value!r}")
        return value
    raise ConfigError(f"{name}: unsupported annotation {expected!r}")


def from_dict(cls: type[T], data: dict[str, Any], _prefix: str = "") -> T:
    """Build dataclass ``cls`` from a plain dict, rejecting unknown/missing keys."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key(s) for {cls.__name__}: {', '.join(_prefix + k for k in unknown)}")
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _check_type(_prefix + name, data[name], hints[name])
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"missing required key: {_prefix + name}")
    return cls(**kwargs)


def to_dict(obj: Any) -> dict[str, Any]:
    """Dataclass -> plain dict (recursively), suitable for YAML/JSON."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"{obj!r} is not a dataclass instance")
    return dataclasses.asdict(obj)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping at the top level."""
    with open(path, encoding="utf-8") as f:
        data = _yaml_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_config(cls: type[T], path: str | Path, overrides: dict[str, Any] | None = None) -> T:
    """Load a YAML file into dataclass ``cls``, applying optional top-level overrides."""
    data = load_yaml(path)
    if overrides:
        data = {**data, **overrides}
    try:
        return from_dict(cls, data)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def save_config(obj: Any, path: str | Path) -> None:
    """Write a dataclass config as YAML (key order preserved, human-readable)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(obj), f, sort_keys=False)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key=value`` command-line override; the value is parsed as YAML.

    Using YAML for the value means ``lr=3e-4`` becomes a float, ``compile=false``
    a bool and ``name=foo`` a string, without a custom mini-language.
    """
    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise ConfigError(f"override must look like key=value, got {text!r}")
    return key.strip(), _yaml_load(raw)
