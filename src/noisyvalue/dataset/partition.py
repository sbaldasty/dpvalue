"""Partitions of a finite atom set, and the lattice operations over them.

Every structural statement a dataset makes -- an axis binning, a geographic
level, the coarsening a constraint is stated over, the blocks a solver may
treat independently -- is a family of disjoint groups of atoms.  Modelling
them as one type means the reconciliation between any two of them is one
operation instead of a bespoke helper per pair.

The two operations that matter:

- **join** (`Partition.join`) is the finest common coarsening.  It answers
  "a query is binned one way and the evidence another; what is the coarsest
  thing they agree about?"  Each join block reports whether the two sides
  cover *the same* atoms in it, which is what decides whether the evidence
  pins that block's total or merely bounds it.
- **meet** (`Partition.meet`) is the coarsest common refinement, needed when
  two partitions are incomparable and the only way to reconcile them is to
  descend to atoms and re-aggregate.

A partition here is not required to cover the atom set.  Real ones do not:
a constraint stated over "nursing homes only" selects three of eight
categories, and a sub-selection pins nothing about the categories it omits.
`is_cover` is load-bearing, not decoration.
"""


class Partition:
    """Disjoint groups of atoms drawn from `range(size)`.

    Group order is significant and preserved: it is the position order of the
    axis (or level) the partition describes, parallel to that axis's labels.
    Atom order inside a group is not, and is canonicalized.
    """

    def __init__(self, groups, size, name=None):
        self.size = int(size)
        self.name = name

        canonical = []
        seen = {}
        for position, group in enumerate(groups):
            atoms = tuple(sorted(set(int(a) for a in group)))
            for atom in atoms:
                if not 0 <= atom < self.size:
                    raise ValueError(
                        f"partition {name!r}: atom {atom} outside range({self.size})")
                if atom in seen:
                    raise ValueError(
                        f"partition {name!r}: atom {atom} appears in groups "
                        f"{seen[atom]} and {position}")
                seen[atom] = position
            canonical.append(atoms)

        self.groups = tuple(canonical)
        self._block_of = seen

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def trivial(cls, size, name=None):
        """One group holding every atom: the fully marginalized partition."""
        return cls((tuple(range(size)),), size, name)

    @classmethod
    def discrete(cls, size, name=None):
        """One group per atom: the finest partition."""
        return cls(tuple((i,) for i in range(size)), size, name)

    @classmethod
    def from_labels(cls, labels, size, name=None):
        """Group atoms by a label per atom, in first-appearance order."""
        order, groups = [], {}
        for atom, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
                order.append(label)
            groups[label].append(atom)
        return cls(tuple(tuple(groups[label]) for label in order), size, name)

    @classmethod
    def from_ranges(cls, ranges, size, name=None):
        """One group per inclusive `(lo, hi)` pair."""
        return cls(tuple(tuple(range(lo, hi + 1)) for lo, hi in ranges), size, name)

    # ── basic queries ───────────────────────────────────────────────────────

    def __len__(self):
        return len(self.groups)

    def __repr__(self):
        label = "" if self.name is None else f" {self.name!r}"
        kind = "partition" if self.is_cover else "partial cover"
        return f"<{kind}{label}: {len(self.groups)} groups of {self.size} atoms>"

    @property
    def covered(self):
        return frozenset(self._block_of)

    @property
    def is_cover(self):
        """True when every atom lands in some group."""
        return len(self._block_of) == self.size

    def block_of(self, atom):
        """Position of the group holding `atom`, or None if uncovered."""
        return self._block_of.get(int(atom))

    def refines(self, other):
        """True when every group of self sits inside one group of `other`."""
        if self.size != other.size:
            raise ValueError("partitions over different atom sets")
        for group in self.groups:
            blocks = {other.block_of(a) for a in group}
            if len(blocks) != 1 or None in blocks:
                return False
        return True

    # ── lattice ─────────────────────────────────────────────────────────────

    def join(self, other):
        """Finest common coarsening, as a tuple of `JoinBlock`s.

        Positions of the two partitions are merged whenever they share an
        atom, transitively.  A block records the positions it merged on each
        side, so callers can translate between the two indexings, and whether
        the two sides cover identical atoms within it -- the test that decides
        if an exact total on one side transfers to the other.

        Atoms neither side covers are absent.  Atoms only one side covers
        produce blocks that are not `exact`.
        """
        if self.size != other.size:
            raise ValueError("partitions over different atom sets")

        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for side, part in (("a", self), ("b", other)):
            for position in range(len(part.groups)):
                parent[(side, position)] = (side, position)
        for atom in self.covered & other.covered:
            union(("a", self.block_of(atom)), ("b", other.block_of(atom)))

        members = {}
        for key in parent:
            members.setdefault(find(key), []).append(key)

        blocks = []
        for group in members.values():
            a_positions = tuple(sorted(p for side, p in group if side == "a"))
            b_positions = tuple(sorted(p for side, p in group if side == "b"))
            a_atoms = frozenset(a for p in a_positions for a in self.groups[p])
            b_atoms = frozenset(a for p in b_positions for a in other.groups[p])
            blocks.append(JoinBlock(a_positions, b_positions, a_atoms, b_atoms))
        # order by where the block starts on the left, so output is stable
        blocks.sort(key=lambda b: (b.a_positions or (len(self.groups),), b.b_positions))
        return tuple(blocks)

    def meet(self, other):
        """Coarsest common refinement: the nonempty pairwise intersections."""
        if self.size != other.size:
            raise ValueError("partitions over different atom sets")
        groups = []
        for mine in self.groups:
            for theirs in other.groups:
                overlap = tuple(a for a in mine if a in set(theirs))
                if overlap:
                    groups.append(overlap)
        groups.sort(key=lambda g: g[0])
        return Partition(tuple(groups), self.size)

    # ── translation helpers ─────────────────────────────────────────────────

    def touching(self, atoms):
        """Positions of groups that intersect `atoms`."""
        atoms = frozenset(atoms)
        return tuple(p for p, g in enumerate(self.groups) if atoms & frozenset(g))

    def contained_in(self, atoms):
        """Positions of groups that sit entirely inside `atoms`."""
        atoms = frozenset(atoms)
        return tuple(p for p, g in enumerate(self.groups) if frozenset(g) <= atoms)


class JoinBlock:
    """One block of a join: the positions each side contributes, and its atoms.

    `exact` says the two sides cover the same atoms here, so a total known on
    one side is a total known on the other.  `contained` is the weaker fact
    that the left side sits inside the right, which still yields an upper
    bound when the quantities are nonnegative.
    """

    def __init__(self, a_positions, b_positions, a_atoms, b_atoms):
        self.a_positions = a_positions
        self.b_positions = b_positions
        self.a_atoms = a_atoms
        self.b_atoms = b_atoms

    def __repr__(self):
        return (f"<JoinBlock a={self.a_positions} b={self.b_positions} "
                f"{'exact' if self.exact else 'inexact'}>")

    @property
    def exact(self):
        return self.a_atoms == self.b_atoms

    @property
    def contained(self):
        return self.a_atoms <= self.b_atoms
