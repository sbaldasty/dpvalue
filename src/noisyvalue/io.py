"""File I/O for NoisyValue containers."""

import json
import numpy as np
import pandas as pd
import sympy as sp

from sympy import Abs, Add, And, Eq, Float, Mul, Ne, Not, Or, Pow, Piecewise, Rational, Symbol, exp, log, floor
from sympy.core.numbers import Infinity, NaN, NegativeInfinity
from sympy.core.relational import GreaterThan, LessThan, StrictGreaterThan, StrictLessThan
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import BooleanAtom

from . import util
from .analysis import NoisyContingencyTable
from .core import NoisyFloat, NoisyInt, NoisyBool, consolidate
from .graph import BinomialNode, DerivedNode, DiscreteGaussianNode, DiscreteLaplaceNode, LatentNode, NormalNode
from .graph import GaussianCensoredNode, LaplaceCensoredNode
from .pandas_ext import NoisyBoolArray, NoisyFloatArray, NoisyIntArray


VERSION = 6


class Serializer:
    subclasses = {}
    tag = None

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


# ----------------------------------------------------------------------------
# SymPy serializers
# ----------------------------------------------------------------------------


class SympySerializer(Serializer):
    @classmethod
    def from_dict(cls, d, name_map):
        raise NotImplementedError


class SympySymbolSerializer(SympySerializer):
    tag = "Symbol"
    matches_type = Symbol

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag, "name": obj.name}

    @classmethod
    def from_dict(cls, d, name_map):
        if d["name"] not in name_map:
            raise ValueError(f"Unresolved symbol in serialized expression: {d['name']!r}")
        return name_map[d["name"]]


class SympyNaNSerializer(SympySerializer):
    tag = "NaN"
    matches_type = NaN

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag}

    @classmethod
    def from_dict(cls, d, name_map):
        return sp.nan


class SympyInfinitySerializer(SympySerializer):
    tag = "Infinity"
    matches_type = Infinity

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag}

    @classmethod
    def from_dict(cls, d, name_map):
        return sp.oo


class SympyNegativeInfinitySerializer(SympySerializer):
    tag = "NegativeInfinity"
    matches_type = NegativeInfinity

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag}

    @classmethod
    def from_dict(cls, d, name_map):
        return -sp.oo


class SympyBooleanSerializer(SympySerializer):
    tag = "Boolean"
    matches_type = BooleanAtom

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag, "value": bool(obj)}

    @classmethod
    def from_dict(cls, d, name_map):
        return sp.true if d["value"] else sp.false


class SympyFloatSerializer(SympySerializer):
    tag = "Float"
    matches_type = Float

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag, "value": repr(float(obj)), "precision": obj._prec}

    @classmethod
    def from_dict(cls, d, name_map):
        return Float(d["value"], precision=d["precision"])


class SympyRationalSerializer(SympySerializer):
    tag = "Rational"
    matches_type = Rational

    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag, "p": int(obj.p), "q": int(obj.q)}

    @classmethod
    def from_dict(cls, d, name_map):
        return Rational(d["p"], d["q"])


class SympyExprSerializer(SympySerializer):
    @classmethod
    def to_dict(cls, obj):
        return {"tag": cls.tag, "args": [expr_to_dict(a) for a in obj.args]}

    @classmethod
    def from_dict(cls, d, name_map):
        args = [expr_from_dict(a, name_map) for a in d["args"]]
        return cls.matches_type(*args)


class SympyAbsSerializer(SympyExprSerializer):
    matches_type = Abs
    tag = "Abs"


class SympyAddSerializer(SympyExprSerializer):
    matches_type = Add
    tag = "Add"


class SympyMulSerializer(SympyExprSerializer):
    matches_type = Mul
    tag = "Mul"


class SympyPowSerializer(SympyExprSerializer):
    matches_type = Pow
    tag = "Pow"


class SympyExpSerializer(SympyExprSerializer):
    matches_type = exp
    tag = "exp"


class SympyLogSerializer(SympyExprSerializer):
    matches_type = log
    tag = "log"


class SympyFloorSerializer(SympyExprSerializer):
    matches_type = floor
    tag = "floor"


class SympyPiecewiseSerializer(SympyExprSerializer):
    matches_type = Piecewise
    tag = "Piecewise"


class SympyExprCondPairSerializer(SympyExprSerializer):
    matches_type = ExprCondPair
    tag = "ExprCondPair"


class SympyAndSerializer(SympyExprSerializer):
    matches_type = And
    tag = "And"


class SympyOrSerializer(SympyExprSerializer):
    matches_type = Or
    tag = "Or"


class SympyNotSerializer(SympyExprSerializer):
    matches_type = Not
    tag = "Not"


class SympyEqSerializer(SympyExprSerializer):
    matches_type = Eq
    tag = "Eq"


class SympyNeSerializer(SympyExprSerializer):
    matches_type = Ne
    tag = "Ne"


class SympyStrictLessThanSerializer(SympyExprSerializer):
    matches_type = StrictLessThan
    tag = "StrictLessThan"


class SympyLessThanSerializer(SympyExprSerializer):
    matches_type = LessThan
    tag = "LessThan"


class SympyStrictGreaterThanSerializer(SympyExprSerializer):
    matches_type = StrictGreaterThan
    tag = "StrictGreaterThan"


class SympyGreaterThanSerializer(SympyExprSerializer):
    matches_type = GreaterThan
    tag = "GreaterThan"


# ----------------------------------------------------------------------------
# Node serializers
# ----------------------------------------------------------------------------


class NodeSerializer(Serializer):
    @classmethod
    def to_dict(cls, node):
        return {"tag": cls.tag, "deps": [_node_name(dep) for dep in node.deps]}

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


class DiscreteLaplaceNodeSerializer(NoiseNodeSerializer):
    tag = "discrete_laplace"
    matches_type = DiscreteLaplaceNode


class GaussianCensoredNodeSerializer(NoiseNodeSerializer):
    tag = "gaussian_censored"
    matches_type = GaussianCensoredNode


class LaplaceCensoredNodeSerializer(NoiseNodeSerializer):
    tag = "laplace_censored"
    matches_type = LaplaceCensoredNode


# ----------------------------------------------------------------------------
# Container serializers
# ----------------------------------------------------------------------------


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
        return {"tag": cls.tag, "dtype": str(obj.dtype), "values": obj.tolist()}

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
        return {"tag": cls.tag, "elements": elements}

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
            "tag": cls.tag,
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
            "tag": cls.tag,
            "shape": list(obj.shape),
            "elements": [serializer_for_obj(v, accept=NoisyValueSerializer).to_dict(v) for v in obj.flat]
        }

    @classmethod
    def from_dict(cls, d, built):
        arr = np.empty(tuple(d["shape"]), dtype=object)
        for i, edict in enumerate(d["elements"]):
            arr.flat[i] = container_from_dict(NoisyValueSerializer, edict, built)
        return arr


class DataFrameSerializer(ContainerSerializer):
    tag = "dataframe"
    matches_type = pd.DataFrame

    @classmethod
    def children(cls, obj):
        flat = []
        for col in obj.columns:
            flat.extend(serializer_for_obj(obj[col], accept=SeriesSerializer).children(obj[col]))
        return flat

    @classmethod
    def rebuild(cls, obj, it):
        columns = {
            col: serializer_for_obj(obj[col], accept=SeriesSerializer).rebuild(obj[col], it) for col in obj.columns
        }
        return pd.DataFrame(columns, index=obj.index)

    @classmethod
    def to_dict(cls, obj):
        return {
            "tag": cls.tag,
            "columns": {
                col: serializer_for_obj(obj[col], accept=SeriesSerializer).to_dict(obj[col]) for col in obj.columns
            },
        }

    @classmethod
    def from_dict(cls, d, built):
        columns = {
            name: container_from_dict(SeriesSerializer, col, built)
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
        return {"tag": cls.tag, "array": ArraySerializer.to_dict(obj.tbl)}

    @classmethod
    def from_dict(cls, d, built):
        return NoisyContingencyTable(ArraySerializer.from_dict(d["array"], built))


class TupleSerializer(ContainerSerializer):
    tag = "tuple"
    matches_type = tuple

    @classmethod
    def _item_serializer(cls, item):
        item_serializer = serializer_for_obj(item, accept=ContainerSerializer)
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
        return {"tag": cls.tag, "items": [cls._item_serializer(item).to_dict(item) for item in obj]}

    @classmethod
    def from_dict(cls, d, built):
        return tuple(
            container_from_dict(ContainerSerializer, item, built) for item in d["items"]
        )


def container_from_dict(accept, d, built):
    cls = serializer_for_tag(accept, d["tag"])
    return cls.from_dict(d, built)

def node_from_dict(d, deps, name_map):
    cls = serializer_for_tag(NodeSerializer, d["tag"])
    return cls.from_dict(d, deps, name_map)

def node_to_dict(node):
    cls = serializer_for_obj(node, NodeSerializer)
    return cls.to_dict(node)

def _node_name(node):
    """Return a serialization name for a node that is stable within a save() call."""
    if isinstance(node, DerivedNode):
        return f"derived_{id(node)}"
    return str(node.expr)

def serializer_for_obj(obj, accept=Serializer):
    for serializer in Serializer.subclasses.values():
        if serializer.matches(obj):
            return util.require_subclass(accept, serializer)
    raise TypeError(f"Cannot find serializer for type {type(obj).__name__!r}")

def serializer_for_tag(accept, tag):
    result = Serializer.subclasses[tag]
    return util.require_subclass(accept, result)

def expr_to_dict(expr):
    """Serialize a SymPy expression to a JSON-safe structural tree."""
    expr = sp.sympify(expr)
    cls = serializer_for_obj(expr, accept=SympySerializer)
    return cls.to_dict(expr)

def expr_from_dict(d, name_map):
    tag = d["tag"]
    if tag not in Serializer.subclasses:
        raise ValueError(f"Unknown or disallowed expression type in file: {tag!r}")
    cls = serializer_for_tag(SympySerializer, tag)
    return cls.from_dict(d, name_map)

def save(path, container):
    """Save a NoisyValue, ndarray of NoisyValues, or list/tuple of either to a JSON file."""
    serializer = serializer_for_obj(container, accept=ContainerSerializer)
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


def load(path):
    """Load a container saved by save()."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("version") != VERSION:
        raise ValueError(f"Unsupported file version: {doc.get('version')!r}")

    nodes_dict = doc["nodes"]
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

    name_map = {}  # old symbol name str -> new Symbol
    built = {}  # old symbol name str -> Node

    for old_name in order:
        nd = nodes_dict[old_name]
        deps = [built[dep_name] for dep_name in nd.get("deps", [])]

        node = node_from_dict(nd, deps, name_map)
        name_map[old_name] = node.expr
        built[old_name] = node

    return container_from_dict(ContainerSerializer, doc["container"], built)
