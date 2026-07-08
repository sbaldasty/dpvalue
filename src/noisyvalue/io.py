"""File I/O for NoisyValue containers."""

import json
import numpy as np
import pandas as pd
import sympy as sp

from .consolidate import consolidate
from .core import NoisyFloat, NoisyInt, NoisyBool
from .graph import DerivedNode, LatentNode, NoiseNode
from .pandas_ext import NoisyBoolArray, NoisyFloatArray, NoisyIntArray

_VERSION = 3

_NOISY_COLUMN_CLASSES = {
    "noisyfloat": NoisyFloatArray,
    "noisyint": NoisyIntArray,
    "noisybool": NoisyBoolArray,
}

_SP_NAMESPACE = vars(sp)


def deserialize(d, built, accept):
    cls = _KIND_BY_TAG[d["kind"]]
    if not issubclass(cls, accept):
        raise TypeError(f"Expected {accept.__name__}, got {cls.__name__}")
    return cls.from_dict(d, built)


def _node_name(node):
    """Return a serialization name for a node that is stable within a save() call."""
    if isinstance(node, DerivedNode):
        return f"derived_{id(node)}"
    return str(node.expr)


# ── node serialization ──────────────────────────────────────────────────────────

def _noise_node_params_to_dict(node):
    t = type(node).type_name
    if NoiseNode.registry.get(t) is not type(node):
        raise TypeError(f"Unknown NoiseNode type: {type(node)}")
    return {"type": t, "params": [sp.srepr(p) for p in node.params]}


def _node_to_dict(node):
    if isinstance(node, LatentNode):
        return {"kind": "LatentUnit"}
    if isinstance(node, NoiseNode):
        return {
            "kind": "NoiseUnit",
            "source": _noise_node_params_to_dict(node),
            "deps": [_node_name(dep) for dep in node.deps],
        }
    if isinstance(node, DerivedNode):
        return {
            "kind": "DerivedUnit",
            "definition": sp.srepr(node.expr),
            "constraints": [sp.srepr(c) for c in node.constraints],
            "deps": [_node_name(dep) for dep in node.deps],
        }
    raise TypeError(f"Unknown Node type: {type(node)}")


class Unit:
    @classmethod
    def matches(cls, obj):
        return isinstance(obj, cls.matches_type)

    @classmethod
    def to_dict(cls, obj):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d, built):
        raise NotImplementedError


class Container(Unit):
    @classmethod
    def children(cls, obj):
        """NoisyValue leaves contained in obj, in the order rebuild()/to_dict() expect."""
        raise NotImplementedError

    @classmethod
    def rebuild(cls, obj, it):
        """An equivalent container with each leaf NoisyValue replaced by next(it)."""
        raise NotImplementedError


class ColumnContainer(Container):
    pass


class PlainColumnContainer(ColumnContainer):
    @classmethod
    def matches(cls, obj):
        return isinstance(obj, pd.Series) and type(obj.array) not in _NOISY_COLUMN_CLASSES.values()

    @classmethod
    def children(cls, obj):
        return []

    @classmethod
    def rebuild(cls, obj, it):
        return obj

    @classmethod
    def to_dict(cls, obj):
        return {"kind": cls.__name__, "dtype": str(obj.dtype), "values": obj.tolist()}

    @classmethod
    def from_dict(cls, d, built):
        return pd.Series(d["values"], dtype=d["dtype"])


class NoisyColumnContainer(ColumnContainer):
    array_cls = None

    @classmethod
    def matches(cls, obj):
        return isinstance(obj, pd.Series) and type(obj.array) is cls.array_cls

    @classmethod
    def children(cls, obj):
        arr = obj.array
        return [arr[i] for i in range(len(arr)) if not arr._mask[i]]

    @classmethod
    def rebuild(cls, obj, it):
        arr = obj.array
        values = [pd.NA if arr._mask[i] else next(it) for i in range(len(arr))]
        return pd.Series(cls.array_cls._from_sequence(values), index=obj.index)

    @classmethod
    def to_dict(cls, obj):
        arr = obj.array
        elements = [
            None if arr._mask[i]
            else {"obs": arr[i]._obs, "root": _node_name(arr[i]._root)}
            for i in range(len(arr))
        ]
        return {"kind": cls.__name__, "elements": elements}

    @classmethod
    def from_dict(cls, d, built):
        scalar_cls = cls.array_cls._scalar_cls
        values = [
            pd.NA if e is None else scalar_cls(e["obs"], built[e["root"]])
            for e in d["elements"]
        ]
        return pd.Series(cls.array_cls._from_sequence(values))


class FloatColumnContainer(NoisyColumnContainer):
    array_cls = NoisyFloatArray


class IntColumnContainer(NoisyColumnContainer):
    array_cls = NoisyIntArray


class BoolColumnContainer(NoisyColumnContainer):
    array_cls = NoisyBoolArray


class ValueContainer(Container):
    @classmethod
    def children(cls, obj):
        return [obj]

    @classmethod
    def rebuild(cls, obj, it):
        return next(it)

    @classmethod
    def to_dict(cls, obj):
        return {
            "kind": cls.__name__,
            "obs": obj._obs,
            "root": _node_name(obj._root),
        }

    @classmethod
    def from_dict(cls, d, built):
        return cls.matches_type(d["obs"], built[d["root"]])


class FloatContainer(ValueContainer):
    matches_type = NoisyFloat


class IntContainer(ValueContainer):
    matches_type = NoisyInt


class BoolContainer(ValueContainer):
    matches_type = NoisyBool


class ArrayContainer(Container):
    @classmethod
    def matches(cls, obj):
        return isinstance(obj, np.ndarray)

    @classmethod
    def children(cls, obj):
        return list(obj.flat)

    @classmethod
    def rebuild(cls, obj, it):
        arr = np.empty(obj.shape, dtype=object)
        for i in range(arr.size):
            arr.flat[i] = next(it)
        return arr

    @classmethod
    def to_dict(cls, obj):
        return {
            "kind": cls.__name__,
            "shape": list(obj.shape),
            "elements": [
                {"kind": _kind_for(v).__name__, "obs": v._obs, "root": _node_name(v._root)}
                for v in obj.flat
            ],
        }

    @classmethod
    def from_dict(cls, d, built):
        arr = np.empty(tuple(d["shape"]), dtype=object)
        for i, edict in enumerate(d["elements"]):
            arr.flat[i] = deserialize(edict, built, accept=ValueContainer)
        return arr


class TableContainer(Container):
    @classmethod
    def matches(cls, obj):
        return isinstance(obj, pd.DataFrame)

    @classmethod
    def children(cls, obj):
        flat = []
        for col in obj.columns:
            flat.extend(_kind_for(obj[col]).children(obj[col]))
        return flat

    @classmethod
    def rebuild(cls, obj, it):
        columns = {
            col: _kind_for(obj[col]).rebuild(obj[col], it) for col in obj.columns
        }
        return pd.DataFrame(columns, index=obj.index)

    @classmethod
    def to_dict(cls, obj):
        return {
            "kind": cls.__name__,
            "columns": {
                col: _kind_for(obj[col]).to_dict(obj[col]) for col in obj.columns
            },
        }

    @classmethod
    def from_dict(cls, d, built):
        columns = {
            name: deserialize(col, built, accept=ColumnContainer)
            for name, col in d["columns"].items()
        }
        return pd.DataFrame(columns)


class TupleContainer(Container):
    """Shared logic for top-level tuple containers.

    Items must be a NoisyValue, ndarray, or DataFrame — a tuple is only
    ever one level deep, matching save()'s documented contract — so nested
    lists/tuples are rejected rather than accepted silently.
    """

    @classmethod
    def matches(cls, obj):
        return isinstance(obj, tuple)

    @classmethod
    def _item_kind(cls, item):
        item_kind = _kind_for(item)
        if item_kind not in (FloatContainer, IntContainer, BoolContainer, ArrayContainer, TableContainer):
            raise TypeError(
                f"List/tuple items must be NoisyValue, ndarray, or DataFrame, got {type(item)}"
            )
        return item_kind

    @classmethod
    def children(cls, obj):
        flat = []
        for item in obj:
            flat.extend(cls._item_kind(item).children(item))
        return flat

    @classmethod
    def rebuild(cls, obj, it):
        return tuple(cls._item_kind(item).rebuild(item, it) for item in obj)

    @classmethod
    def to_dict(cls, obj):
        return {"kind": cls.__name__, "items": [cls._item_kind(item).to_dict(item) for item in obj]}

    @classmethod
    def from_dict(cls, d, built):
        return tuple(
            _KIND_BY_TAG[item["kind"]].from_dict(item, built) for item in d["items"]
        )


_KINDS = [FloatContainer, IntContainer, BoolContainer, ArrayContainer, TableContainer, TupleContainer,
          FloatColumnContainer, IntColumnContainer, BoolColumnContainer, PlainColumnContainer]
_KIND_BY_TAG = {k.__name__: k for k in _KINDS}


def _kind_for(obj):
    for kind in _KINDS:
        if kind.matches(obj):
            return kind
    return None


def _collect_nodes(kind, container):
    nodes = {}
    for v in kind.children(container):
        for node in v._root.closure():
            name = _node_name(node)
            if name not in nodes:
                nodes[name] = node
    return nodes


def save(path, container):
    """Save a NoisyValue, ndarray of NoisyValues, or list/tuple of either to a JSON file."""
    kind = _kind_for(container)
    if kind is None:
        raise TypeError(f"Unsupported container type: {type(container)}")
    flat = kind.children(container)
    consolidated = consolidate(*flat)
    container = kind.rebuild(container, iter(consolidated))
    nodes = _collect_nodes(kind, container)
    doc = {
        "version": _VERSION,
        "nodes": {name: _node_to_dict(node) for name, node in nodes.items()},
        "container": kind.to_dict(container),
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


# ── node deserialization ────────────────────────────────────────────────────────

def _topo_sort(nodes_dict):
    visited = set()
    order = []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        for dep in nodes_dict[name].get("deps", []):
            visit(dep)
        order.append(name)

    for name in nodes_dict:
        visit(name)
    return order


def _parse_expr(s, name_map):
    expr = eval(s, _SP_NAMESPACE)  # noqa: S307 — we wrote the file
    for old, new_sym in name_map.items():
        expr = expr.subs(sp.Symbol(old), new_sym)
    return expr


def _load_noise_node(source_dict, name_map, deps=()):
    t = source_dict["type"]
    cls = NoiseNode.registry.get(t)
    if cls is None:
        raise ValueError(f"Unknown source type: {t!r}")
    params = [_parse_expr(p, name_map) for p in source_dict["params"]]
    return cls(params, deps)


def _load_nodes(nodes_dict):
    order = _topo_sort(nodes_dict)
    name_map = {}  # old symbol name str -> new Symbol
    built = {}     # old symbol name str -> Node

    for old_name in order:
        nd = nodes_dict[old_name]
        kind = nd["kind"]
        deps = [built[dep_name] for dep_name in nd.get("deps", [])]

        def remap(s, _map=name_map):
            return _parse_expr(s, _map)

        if kind == "LatentUnit":
            node = LatentNode()
            name_map[old_name] = node.expr
        elif kind == "NoiseUnit":
            node = _load_noise_node(nd["source"], name_map, deps=deps)
            name_map[old_name] = node.expr
        elif kind == "DerivedUnit":
            node = DerivedNode(
                remap(nd["definition"]),
                constraints=[remap(c) for c in nd["constraints"]],
                deps=deps,
            )
        else:
            raise ValueError(f"Unknown node kind: {kind!r}")

        built[old_name] = node

    return built


def load(path):
    """Load a container saved by save()."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("version") != _VERSION:
        raise ValueError(f"Unsupported file version: {doc.get('version')!r}")
    built = _load_nodes(doc["nodes"])
    cdict = doc["container"]
    kind = _KIND_BY_TAG.get(cdict.get("kind"))
    if kind is None:
        raise ValueError(f"Unknown container kind: {cdict.get('kind')!r}")
    return kind.from_dict(cdict, built)
