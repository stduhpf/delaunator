from math import sqrt
import numpy as np

Sphere = tuple[np.ndarray, float]


class Point:
    _instances: list["Point"] = []

    @classmethod
    def make(cls, x, y, z, name=None) -> "Point":
        new = cls(x, y, z, name, cc=hash("Called correctly"))
        try:
            old = [x for x in cls._instances if x.val == new.val][0]
            del new
            return old
        except (ValueError, IndexError):
            cls._instances.append(new)
            return new

    def __init__(self, x, y, z, name: str = None, cc: int = 0) -> None:
        assert cc == hash("Called correctly")
        self.val = np.array([x, y, z])
        self.name: str = name
        self.in_segments: list["Segment"] = []
        self.in_triangles: list["Triangle"] = []
        self.in_tetra: list["Tetra"] = []
        self.checked = False

    def __repr__(self) -> str:
        if self.name is None:
            return object.__repr__(self)
        return self.name

    def __cmp__(self, other: "Point"):
        assert self.val.shape == (3,) and other.val.shape == (3,)
        if self.val[0] == other.val[0]:
            if self.val[0] == other.val[0]:
                return self.val[2] - other.val[2]
            return self.val[1] - other.val[1]
        return self.val[0] - other.val[0]


class Segment:
    _instances: list[tuple[int, "Segment"]] = []

    @classmethod
    def make(cls, start, end) -> "Segment":
        try:
            return [x[1] for x in cls._instances if x[0] == hash(start) ^ hash(end)][0]
        except (ValueError, IndexError):
            new = cls(start, end, hash("Called correctly"))
            cls._instances.append((hash(new), new))
            return new

    def __init__(self, start: Point, end: Point, cc: int = 0) -> None:
        assert cc == hash("Called correctly")
        self.start = start
        self.end = end
        start.in_segments.append(self)
        end.in_segments.append(self)

        self.name = f"[{start}{end}]"

        self.in_triangles: list["Triangle"] = []
        self.in_tetra: list["Tetra"] = []

        self.checked = False

    def __repr__(self) -> str:
        if self.name is None:
            return object.__repr__(self)
        return self.name

    def __hash__(self) -> int:
        # print("hash called")
        return hash(self.start) ^ hash(self.end)

    def __eq__(self, other: "Segment") -> bool:
        return (self.start == other.start and self.end == other.end) or (
            self.start == other.end and self.end == other.start
        )

    def contains(self, p: Point) -> bool:
        return self.start == p or self.end == p

    def disjoint(self, other: "Segment") -> bool:
        return not self.contains(other.start) and not self.contains(other.end)

    def complement(self, p: Point) -> Point:
        assert self.contains(p), f"Point {p} cannot be found in segment {self}"
        return self.end if p == self.start else self.start


class Triangle:
    _instances: list[tuple[int, "Triangle"]] = []

    @classmethod
    def make(cls, A: Point, B: Point, C: Point) -> "Triangle":
        try:
            return [
                x[1] for x in cls._instances if x[0] == hash(A) ^ hash(B) ^ hash(C)
            ][0]
        except (ValueError, IndexError):
            new = cls(A, B, C, hash("Called correctly"))
            cls._instances.append((hash(new), new))
            return new

    def __init__(self, A: Point, B: Point, C: Point, cc: int = None) -> None:
        assert cc == hash("Called correctly")
        self.segments: list[Segment] = [
            Segment.make(A, B),
            Segment.make(B, C),
            Segment.make(C, A),
        ]
        for s in self.segments:
            s.in_triangles.append(self)

        self.vertices: list[Point] = [A, B, C]
        for vertex in self.vertices:
            vertex.in_triangles.append(self)

        self.name: str = "{" + f"{A}{B}{C}" + "}"
        self.in_tetra: list["Tetra"] = []

        self.checked = False

    def __hash__(self) -> int:
        # print("hash called")
        return hash(self.vertices[0]) ^ hash(self.vertices[1]) ^ hash(self.vertices[2])

    def __repr__(self) -> str:
        if self.name is None:
            return object.__repr__(self)
        return self.name

    def complementS(self, s: Segment) -> Point:
        assert s in self.segments
        return [p for p in self.vertices if not s.contains(p)][0]

    def complementP(self, p: Point) -> Segment:
        assert p in self.vertices
        return [s for s in self.segments if not s.contains(p)][0]


class Tetra:
    def setup(self):
        self.vert = np.vstack([vertex.val for vertex in self.vertices])
        self.o = self.vert[3]
        self.M = self.vert[:3] - self.o
        self.iM = np.linalg.inv(self.M)
        self.boundingSphere = getBoundingSphere(self)

    def __init__(
        self, A: Point, B: Point, C: Point, D: Point, name: str = None
    ) -> None:
        self.triangles: list[Triangle] = [
            Triangle.make(A, B, C),
            Triangle.make(A, B, D),
            Triangle.make(B, C, D),
            Triangle.make(A, C, D),
        ]
        for t in self.triangles:
            t.in_tetra.append(self)

        self.segments: list[Segment] = [
            Segment.make(A, B),
            Segment.make(A, C),
            Segment.make(A, D),
            Segment.make(B, C),
            Segment.make(B, D),
            Segment.make(C, D),
        ]
        for s in self.segments:
            s.in_tetra.append(self)

        self.vertices: list[Point] = [A, B, C, D]
        for vertex in self.vertices:
            vertex.in_tetra.append(self)

        self.name = f"({A}{B}{C}{D})"
        self.setup()

    def __repr__(self) -> str:
        if self.name is None:
            return object.__repr__(self)
        return self.name

    def complementT(self, t: Triangle) -> Point:
        assert t in self.triangles
        return [p for p in self.vertices if not p in t.vertices][0]

    def complementS(self, seg: Segment) -> Segment:
        assert seg in self.segments
        ret = [s for s in self.segments if seg.disjoint(s)]
        return ret[0]

    def complementP(self, p: Point) -> Triangle:
        assert p in self.vertices
        return [t for t in self.triangles if not p in t.vertices][0]


def getBoundingSphere(tetrahedron: Tetra) -> Sphere:
    o = tetrahedron.o
    M = tetrahedron.M
    iM = tetrahedron.iM
    k = np.diag(np.matmul(M, np.transpose(M))) * 0.5
    g = np.matmul(iM, k)
    return (g + o, np.linalg.norm(g))


def pointInTetra(point: Point, tetrahedron: Tetra) -> bool:
    o = tetrahedron.o
    iM = tetrahedron.iM
    c = np.matmul(point.val - o, iM)
    return (c >= 0).all() and np.dot(c, [1, 1, 1]) <= 1


def pointInSphere(point: Point, sphere: Sphere) -> bool:
    (center, radius) = sphere
    assert point.val.shape == (3,) and center.shape == (3,)
    return np.linalg.norm(point.val - center) <= radius


def pointInBoundSphere(point: Point, tetrahedron: Tetra) -> bool:
    return pointInSphere(point, tetrahedron.boundingSphere)


def addPoint(point: Point, tetrahedrization: list[Tetra]):
    containers: list[Tetra] = []
    for tet in tetrahedrization:
        if pointInBoundSphere(point, tet):
            containers.append(tet)
    containedFaces: list[Triangle] = []
    remainingFaces: list[Triangle] = []
    containedEdges: list[Segment] = []
    remainingEdges: list[Segment] = []
    for tet in containers:
        for f in tet.triangles:
            if not f.checked:
                f.checked = True
                contained = False
                for t in f.in_tetra:
                    if t != tet and t in containers:
                        containedFaces.append(f)
                        contained = True
                        break
                if not contained:
                    remainingFaces.append(f)
        for seg in tet.segments:
            if not seg.checked:
                seg.checked = True
                tets = [t for t in containers if t in seg.in_tetra]
                if len(tets) > 0:
                    # print(f"\t{seg}:")
                    s = tet.complementS(seg)
                    dest = s.start
                    vertex = s.end
                    t = tet
                    while True:
                        # print(f"dest : {dest}\n vertex: {vertex}\n seg: {s}\n t: {t}\n")
                        rem = [_t for _t in tets if _t != t and vertex in _t.vertices]
                        if len(rem) < 1:
                            # print("No loop\n")
                            remainingEdges.append(seg)
                            break
                        t = rem[0]
                        s = t.complementS(seg)

                        # print(f"{vertex}=>{s.complement(vertex)}\n")
                        vertex = s.complement(vertex)
                        if vertex == dest:
                            containedEdges.append(seg)
                            # print("Yes loop\n")
                            break

    newTet: list[Tetra] = []
    for face in remainingFaces:
        t = Tetra(p, *face.vertices)
        print(t)

    print(containedFaces)
    print(remainingFaces)
    print("\n")
    print(containedEdges)
    print(remainingEdges)
    for tet in containers:
        for f in tet.triangles:
            f.checked = False
        for seg in tet.segments:
            seg.checked = False


# A = Point.make(0.0, 0.0, 10.0, "_A_")
# B = Point.make(-20.0 * sqrt(2) / 3.0, 0.0, -10.0 / 3.0, "_B_")
# C = Point.make(20.0 * sqrt(2) / 6.0, 10.0 * sqrt(2.0 / 3.0), -10.0 / 3.0, "_C_")
# D = Point.make(20.0 * sqrt(2) / 6.0, -10.0 * sqrt(2.0 / 3.0), -10.0 / 3.0, "_D_")
# bigTetrahedron = Tetra(A, B, C, D)
# E = Point.make(0.0, 5.0, 10.0, "E")
# t2 = Tetra(A, B, C, E)


p = Point.make(0.02, 0.01, 0.01, "P")

A = Point.make(0.0, 0.0, 1.0, "_A_")
B = Point.make(0.0, 0.0, -1.0, "_B_")
C = Point.make(1.0, 0.0, 0.0, "_C_")
D = Point.make(0.0, 1.0, 0.0, "_D_")
E = Point.make(-1.0, 0.0, 0.0, "_E_")
F = Point.make(0.0, -1.0, 0.0, "_F_")

t1 = Tetra(A, B, C, D)
t2 = Tetra(A, B, D, E)
t3 = Tetra(A, B, E, F)
t4 = Tetra(A, B, F, C)


addPoint(p, [t1, t2, t3, t4])
