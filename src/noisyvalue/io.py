"""File I/O for NoisyValue containers."""

import json
import numpy as np
import pandas as pd
import sympy as sp

from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import BooleanAtom

from . import util
from .analysis import NoisyContingencyTable
from .consolidate import consolidate
from .core import NoisyFloat, NoisyInt, NoisyBool
from .graph import BinomialNode, DerivedNode, DiscreteGaussianNode, LatentNode, NormalNode
from .pandas_ext import NoisyBoolArray, NoisyFloatArray, NoisyIntArray

VERSION = 6

# The closed set of compound expression types this library ever builds (see
# core.py's bin_op/unary_op/guarded). Deserialization only ever instantiates
# classes from this table -- never eval()s attacker-controlled text -- so a
# maliciously crafted file cannot reach arbitrary Python execution.
_EXPR_CLASSES = {
    cls.__name__: cls
    for cls in (
        sp.Add, sp.Mul, sp.Pow,
        sp.exp, sp.log, sp.floor, sp.Abs,
        sp.Piecewise, ExprCondPair,
        sp.And, sp.Or, sp.Not,
        sp.Eq, sp.Ne,
        sp.core.relational.StrictLessThan, sp.core.relational.LessThan,
        sp.core.relational.StrictGreaterThan, sp.core.relational.GreaterThan,
    )
}


def expr_to_dict(expr):
    """Serialize a SymPy expression to a JSON-safe structural tree."""
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Symbol):
        return {"tag": "Symbol", "name": expr.name}
    if isinstance(expr, sp.core.numbers.NaN):
        return {"tag": "NaN"}
    if isinstance(expr, BooleanAtom):
        return {"tag": "Boolean", "value": bool(expr)}
    if isinstance(expr, sp.Float):
        return {"tag": "Float", "value": repr(float(expr)), "precision": expr._prec}
    if isinstance(expr, sp.Rational):
        return {"tag": "Rational", "p": int(expr.p), "q": int(expr.q)}
    cls_name = type(expr).__name__
    if cls_name not in _EXPR_CLASSES:
        raise TypeError(f"Cannot serialize expression of type {cls_name!r}: {expr!r}")
    return {"tag": cls_name, "args": [expr_to_dict(a) for a in expr.args]}


def expr_from_dict(d, name_map):
    """Reconstruct a SymPy expression from expr_to_dict() output.

    `name_map` maps old symbol name -> already-rebuilt expression, so shared
    symbols resolve to the same (possibly freshly renamed) node.
    """
    tag = d["tag"]
    if tag == "Symbol":
        if d["name"] not in name_map:
            raise ValueError(f"Unresolved symbol in serialized expression: {d['name']!r}")
        return name_map[d["name"]]
    if tag == "NaN":
        return sp.nan
    if tag == "Boolean":
        return sp.true if d["value"] else sp.false
    if tag == "Float":
        return sp.Float(d["value"], precision=d["precision"])
    if tag == "Rational":
        return sp.Rational(d["p"], d["q"])
    if tag not in _EXPR_CLASSES:
        raise ValueError(f"Unknown or disallowed expression type in file: {tag!r}")
    cls = _EXPR_CLASSES[tag]
    args = [expr_from_dict(a, name_map) for a in d["args"]]
    return cls(*args)


class Serializer:
    subclasses = {}
    tag = None

    @classmethod
    def for_tag(cls, tag):
        result = cls.subclasses[tag]
        util.require_subclass(cls, result)
        return result

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "tag" in cls.__dict__:
            Serializer.subclasses[cls.tag] = cls

    @classmethod
    def matches(cls, obj):
        return isinstance(obj, cls.matches_type)

    @classmethod
    def to_dict(cls, obj):
        raise NotImplementedError


class NodeSerializer(Serializer):
    @classmethod
    def to_dict(cls, node):
        return {"kind": cls.tag, "deps": [_node_name(dep) for dep in node.deps]}

    @classmethod
    def from_dict(cls, d, deps, name_map):
        raise NotImplementedError


class LatentNodeSerializer(NodeSerializer):
    tag = "latent"
    matches_type = LatentNode

    @classmethod
    def from_dict(cls, d, deps, name_map):
        return LatentNode()


class DerivedNodeSerializer(NodeSerializer):
    tag = "derived"
    matches_type = DerivedNode

    @classmethod
    def to_dict(cls, node):
        d = super().to_dict(node)
        d["definition"] = expr_to_dict(node.expr)
        d["constraints"] = [expr_to_dict(c) for c in node.constraints]
        return d

    @classmethod
    def from_dict(cls, d, deps, name_map):
        return DerivedNode(
            expr_from_dict(d["definition"], name_map),
            constraints=[expr_from_dict(c, name_map) for c in d["constraints"]],
            deps=deps)


class NoiseNodeSerializer(NodeSerializer):
    @classmethod
    def to_dict(cls, node):
        d = super().to_dict(node)
        d["params"] = [expr_to_dict(p) for p in node.params]
        return d

    @classmethod
    def from_dict(cls, d, deps, name_map):
        params = [expr_from_dict(p, name_map) for p in d["params"]]
        return cls.matches_type(params, deps)


class NormalNodeSerializer(NoiseNodeSerializer):
    tag = "normal"
    matches_type = NormalNode


class BinomialNodeSerializer(NoiseNodeSerializer):
    tag = "binomial"
    matches_type = BinomialNode


class DiscreteGaussianNodeSerializer(NoiseNodeSerializer):
    tag = "discrete_gaussian"
    matches_type = DiscreteGaussianNode


class ContainerSerializer(Serializer):
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


class SeriesSerializer(ContainerSerializer):
    array_classes = []


class PlainSeriesSerializer(SeriesSerializer):
    tag = "plain_series"

    @classmethod
    def matches(cls, obj):
        return isinstance(obj, pd.Series) and type(obj.array) not in SeriesSerializer.array_classes

    @classmethod
    def children(cls, obj):
        return []

    @classmethod
    def rebuild(cls, obj, it):
        return obj

    @classmethod
    def to_dict(cls, obj):
        return {"kind": cls.tag, "dtype": str(obj.dtype), "values": obj.tolist()}

    @classmethod
    def from_dict(cls, d, built):
        return pd.Series(d["values"], dtype=d["dtype"])


class NoisySeriesSerializer(SeriesSerializer):
    array_cls = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.array_classes.append(cls.array_cls)

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
        return {"kind": cls.tag, "elements": elements}

    @classmethod
    def from_dict(cls, d, built):
        scalar_cls = cls.array_cls._scalar_cls
        values = [
            pd.NA if e is None else scalar_cls(e["obs"], built[e["root"]])
            for e in d["elements"]
        ]
        return pd.Series(cls.array_cls._from_sequence(values))


class FloatSeriesSerializer(NoisySeriesSerializer):
    array_cls = NoisyFloatArray
    tag = "float_series"


class IntSeriesSerializer(NoisySeriesSerializer):
    array_cls = NoisyIntArray
    tag = "int_series"


class BoolSeriesSerializer(NoisySeriesSerializer):
    array_cls = NoisyBoolArray
    tag = "bool_series"


class NoisyValueSerializer(ContainerSerializer):
    @classmethod
    def children(cls, obj):
        return [obj]

    @classmethod
    def rebuild(cls, obj, it):
        return next(it)

    @classmethod
    def to_dict(cls, obj):
        return {
            "kind": cls.tag,
            "obs": obj._obs,
            "root": _node_name(obj._root),
        }

    @classmethod
    def from_dict(cls, d, built):
        return cls.matches_type(d["obs"], built[d["root"]])


class NoisyFloatSerializer(NoisyValueSerializer):
    tag = "noisy_float"
    matches_type = NoisyFloat


class NoisyIntSerializer(NoisyValueSerializer):
    tag = "noisy_int"
    matches_type = NoisyInt


class NoisyBoolSerializer(NoisyValueSerializer):
    tag = "noisy_bool"
    matches_type = NoisyBool


class ArraySerializer(ContainerSerializer):
    tag = "array"
    matches_type = np.ndarray

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
            "kind": cls.tag,
            "shape": list(obj.shape),
            "elements": [serializer_for_obj(v).to_dict(v) for v in obj.flat],
        }

    @classmethod
    def from_dict(cls, d, built):
        arr = np.empty(tuple(d["shape"]), dtype=object)
        for i, edict in enumerate(d["elements"]):
            arr.flat[i] = container_from_json(NoisyValueSerializer, edict, built)
        return arr


class DataFrameSerializer(ContainerSerializer):
    tag = "dataframe"
    matches_type = pd.DataFrame

    @classmethod
    def children(cls, obj):
        flat = []
        for col in obj.columns:
            flat.extend(serializer_for_obj(obj[col]).children(obj[col]))
        return flat

    @classmethod
    def rebuild(cls, obj, it):
        columns = {
            col: serializer_for_obj(obj[col]).rebuild(obj[col], it) for col in obj.columns
        }
        return pd.DataFrame(columns, index=obj.index)

    @classmethod
    def to_dict(cls, obj):
        return {
            "kind": cls.tag,
            "columns": {
                col: serializer_for_obj(obj[col]).to_dict(obj[col]) for col in obj.columns
            },
        }

    @classmethod
    def from_dict(cls, d, built):
        columns = {
            name: container_from_json(SeriesSerializer, col, built)
            for name, col in d["columns"].items()
        }
        return pd.DataFrame(columns)


class ContingencyTableSerializer(ContainerSerializer):
    tag = "contingency_table"
    matches_type = NoisyContingencyTable

    @classmethod
    def children(cls, obj):
        return ArraySerializer.children(obj.tbl)

    @classmethod
    def rebuild(cls, obj, it):
        return NoisyContingencyTable(ArraySerializer.rebuild(obj.tbl, it))

    @classmethod
    def to_dict(cls, obj):
        return {"kind": cls.tag, "array": ArraySerializer.to_dict(obj.tbl)}

    @classmethod
    def from_dict(cls, d, built):
        return NoisyContingencyTable(ArraySerializer.from_dict(d["array"], built))


class TupleSerializer(ContainerSerializer):
    tag = "tuple"
    matches_type = tuple

    @classmethod
    def _item_serializer(cls, item):
        item_serializer = serializer_for_obj(item)
        util.require_subclass(ContainerSerializer, item_serializer)
        return item_serializer

    @classmethod
    def children(cls, obj):
        flat = []
        for item in obj:
            flat.extend(cls._item_serializer(item).children(item))
        return flat

    @classmethod
    def rebuild(cls, obj, it):
        return tuple(cls._item_serializer(item).rebuild(item, it) for item in obj)

    @classmethod
    def to_dict(cls, obj):
        return {"kind": cls.tag, "items": [cls._item_serializer(item).to_dict(item) for item in obj]}

    @classmethod
    def from_dict(cls, d, built):
        return tuple(
            container_from_json(ContainerSerializer, item, built) for item in d["items"]
        )


def container_from_json(accept, d, built):
    cls = Serializer.for_tag(d["kind"])
    return cls.from_dict(d, built)

def node_from_json(d, deps, name_map):
    cls = Serializer.for_tag(d["kind"])
    return cls.from_dict(d, deps, name_map)

def node_to_dict(node):
    cls = serializer_for_obj(node)
    util.require_subclass(NodeSerializer, cls)
    return cls.to_dict(node)

def _node_name(node):
    """Return a serialization name for a node that is stable within a save() call."""
    if isinstance(node, DerivedNode):
        return f"derived_{id(node)}"
    return str(node.expr)

def serializer_for_obj(obj):
    for serializer in Serializer.subclasses.values():
        if serializer.matches(obj):
            return serializer
    return None


def save(path, container):
    """Save a NoisyValue, ndarray of NoisyValues, or list/tuple of either to a JSON file."""
    serializer = serializer_for_obj(container)
    if serializer is None:
        raise TypeError(f"Unsupported container type: {type(container)}")
    flat = serializer.children(container)
    consolidated = consolidate(*flat)
    container = serializer.rebuild(container, iter(consolidated))

    nodes = {}
    for v in serializer.children(container):
        for node in v._root.closure():
            name = _node_name(node)
            if name not in nodes:
                nodes[name] = node

    doc = {
        "version": VERSION,
        "nodes": {name: node_to_dict(node) for name, node in nodes.items()},
        "container": serializer.to_dict(container),
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


def load(path):
    """Load a container saved by save()."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("version") != VERSION:
        raise ValueError(f"Unsupported file version: {doc.get('version')!r}")

    nodes_dict = doc["nodes"]
    order = _topo_sort(nodes_dict)
    name_map = {}  # old symbol name str -> new Symbol
    built = {}  # old symbol name str -> Node

    for old_name in order:
        nd = nodes_dict[old_name]
        deps = [built[dep_name] for dep_name in nd.get("deps", [])]

        node = node_from_json(nd, deps, name_map)
        name_map[old_name] = node.expr
        built[old_name] = node

    return container_from_json(ContainerSerializer, doc["container"], built)
