"""File I/O for NoisyValue containers."""

import json
import numpy as np
import pandas as pd
import sympy as sp

from . import util
from .consolidate import consolidate
from .core import NoisyFloat, NoisyInt, NoisyBool
from .graph import BinomialNode, DerivedNode, DiscreteGaussianNode, LatentNode, NormalNode
from .pandas_ext import NoisyBoolArray, NoisyFloatArray, NoisyIntArray

_VERSION = 4

_NOISY_COLUMN_CLASSES = {
    "noisyfloat": NoisyFloatArray,
    "noisyint": NoisyIntArray,
    "noisybool": NoisyBoolArray,
}

_SP_NAMESPACE = vars(sp)


def container_from_json(accept, d, built):
    cls = kind_for_tag(d["kind"])
    util.require_subclass(accept, cls)
    return cls.from_dict(d, built)

def node_from_json(d, deps, remap):
    cls = kind_for_tag(d["kind"])
    util.require_subclass(NodeUnit, cls)
    return cls.from_dict(d, deps, remap)

def node_to_dict(node):
    cls = _kind_for(node)
    util.require_subclass(NodeUnit, cls)
    return cls.to_dict(node)

def _node_name(node):
    """Return a serialization name for a node that is stable within a save() call."""
    if isinstance(node, DerivedNode):
        return f"derived_{id(node)}"
    return str(node.expr)


class Unit:
    _kinds = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in cls.__bases__:
            Unit._kinds.pop(base.__name__, None)
        Unit._kinds[cls.__name__] = cls

    @classmethod
    def matches(cls, obj):
        return isinstance(obj, cls.matches_type)

    @classmethod
    def to_dict(cls, obj):
        raise NotImplementedError


def kind_for_tag(tag):
    return Unit._kinds[tag]


class NodeUnit(Unit):
    @classmethod
    def to_dict(cls, node):
        return {"kind": cls.__name__, "deps": [_node_name(dep) for dep in node.deps]}

    @classmethod
    def from_dict(cls, d, deps, remap):
        raise NotImplementedError


class LatentUnit(NodeUnit):
    matches_type = LatentNode

    @classmethod
    def from_dict(cls, d, deps, remap):
        return LatentNode()


class DerivedUnit(NodeUnit):
    matches_type = DerivedNode

    @classmethod
    def to_dict(cls, node):
        d = super().to_dict(node)
        d["definition"] = sp.srepr(node.expr)
        d["constraints"] = [sp.srepr(c) for c in node.constraints]
        return d

    @classmethod
    def from_dict(cls, d, deps, remap):
        return DerivedNode(
            remap(d["definition"]),
            constraints=[remap(c) for c in d["constraints"]],
            deps=deps)


class NoiseUnit(NodeUnit):
    @classmethod
    def to_dict(cls, node):
        d = super().to_dict(node)
        d["params"] = [sp.srepr(p) for p in node.params]
        return d

    @classmethod
    def from_dict(cls, d, deps, remap):
        params = [remap(p) for p in d["params"]]
        return cls.matches_type(params, deps)


class NormalUnit(NoiseUnit):
    matches_type = NormalNode


class BinomialUnit(NoiseUnit):
    matches_type = BinomialNode


class DiscreteGaussianUnit(NoiseUnit):
    matches_type = DiscreteGaussianNode


class Container(Unit):
    @classmethod
    def children(cls, obj):
        """NoisyValue leaves contained in obj, in the order rebuild()/to_dict() expect."""
        raise NotImplementedError

    @classmethod
    def rebuild(cls, obj, it):
        """An equivalent container with each leaf NoisyValue replaced by next(it)."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d, built):
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


class TopLevelUnit(Container):
    pass


class ValueContainer(TopLevelUnit):
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


class ArrayContainer(TopLevelUnit):
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
            arr.flat[i] = container_from_json(ValueContainer, edict, built)
        return arr


class TableContainer(TopLevelUnit):
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
            name: container_from_json(ColumnContainer, col, built)
            for name, col in d["columns"].items()
        }
        return pd.DataFrame(columns)


class TupleContainer(TopLevelUnit):
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
            container_from_json(TopLevelUnit, item, built) for item in d["items"]
        )


def _kind_for(obj):
    for kind in Unit._kinds.values():
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
        "nodes": {name: node_to_dict(node) for name, node in nodes.items()},
        "container": kind.to_dict(container),
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


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


def _load_nodes(nodes_dict):
    order = _topo_sort(nodes_dict)
    name_map = {}  # old symbol name str -> new Symbol
    built = {}     # old symbol name str -> Node

    for old_name in order:
        nd = nodes_dict[old_name]
        deps = [built[dep_name] for dep_name in nd.get("deps", [])]

        def remap(s, _map=name_map):
            return _parse_expr(s, _map)

        node = node_from_json(nd, deps, remap)
        name_map[old_name] = node.expr
        built[old_name] = node

    return built


def load(path):
    """Load a container saved by save()."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("version") != _VERSION:
        raise ValueError(f"Unsupported file version: {doc.get('version')!r}")
    built = _load_nodes(doc["nodes"])
    return container_from_json(TopLevelUnit, doc["container"], built)
