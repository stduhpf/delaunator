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
            old.name += "/" + new.name
            print(f"Warning, re-creating existing point {old}")
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
    _instances: list["Segment"] = []

    @classmethod
    def make(cls, start, end) -> "Segment":
        try:
            return [x for x in cls._instances if x.chkPoints(start, end)][0]
        except (ValueError, IndexError):
            new = cls(start, end, hash("Called correctly"))
            cls._instances.append(new)
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

    def __eq__(self, other: "Segment") -> bool:
        return (self.start == other.start and self.end == other.end) or (
            self.start == other.end and self.end == other.start
        )

    def chkPoints(self, start: Point, end: Point) -> bool:
        return all(vertex in [self.start, self.end] for vertex in [start, end])

    def contains(self, p: Point) -> bool:
        return self.start == p or self.end == p

    def disjoint(self, other: "Segment") -> bool:
        return not self.contains(other.start) and not self.contains(other.end)

    def complement(self, p: Point) -> Point:
        assert self.contains(p), f"Point {p} cannot be found in segment {self}"
        return self.end if p == self.start else self.start

    def remove_tetra(self, t: "Tetra") -> None:
        try:
            self.in_tetra.remove(t)
        except ValueError:
            print(f"somehow removing {t} from {self} failed : {self.in_tetra}")
        if len(self.in_tetra) + len(self.in_triangles) == 0:
            self.discard()

    def remove_triangle(self, t: "Triangle") -> None:
        try:
            self.in_triangles.remove(t)
        except ValueError:
            print(f"somehow removing {t} from {self} failed : {self.in_triangles}")
        if len(self.in_tetra) + len(self.in_triangles) == 0:
            self.discard()

    def discard(self):
        # print(f"discarding {self}")
        try:
            self.start.in_segments.remove(self)
        except ValueError:
            print(
                f"somehow removing {self} from {self.start} failed : {self.start.in_segments}"
            )
        try:
            self.end.in_segments.remove(self)
        except ValueError:
            print(
                f"somehow removing {self} from {self.end} failed : {self.end.in_segments}"
            )


class Triangle:
    _instances: list["Triangle"] = []

    @classmethod
    def make(cls, A: Point, B: Point, C: Point) -> "Triangle":
        try:
            return [x for x in cls._instances if x.chkPoints(A, B, C)][0]
        except (ValueError, IndexError):
            new = cls(A, B, C, hash("Called correctly"))
            cls._instances.append(new)
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

    def __eq__(self, other: "Triangle") -> bool:
        return all(vertex in self.vertices for vertex in other.vertices)

    def chkPoints(self, A: Point, B: Point, C: Point) -> bool:
        return all(vertex in self.vertices for vertex in [A, B, C])

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

    def remove_tetra(self, t: "Tetra") -> None:
        try:
            self.in_tetra.remove(t)
        except ValueError:
            print(f"somehow removing {t} from {self} failed : {self.in_tetra}")
        if len(self.in_tetra) == 0:
            self.discard()

    def discard(self):
        # print(f"discarding {self}")
        for vertex in self.vertices:
            try:
                vertex.in_triangles.remove(self)
            except ValueError:
                print(
                    f"somehow removing {self} from {vertex} failed : {vertex.in_triangles}"
                )
        for segment in self.segments:
            segment.remove_triangle(self)


class Tetra:
    _instances: list["Tetra"] = []

    @classmethod
    def make(cls, A: Point, B: Point, C: Point, D: Point) -> "Tetra":
        try:
            return [x for x in cls._instances if x.chkPoints(A, B, C, D)][0]
        except (ValueError, IndexError):
            new = cls(A, B, C, D)
            cls._instances.append(new)
            return new

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

    def __eq__(self, other: "Tetra") -> bool:
        return all(vertex in self.vertices for vertex in other.vertices)

    def chkPoints(self, A: Point, B: Point, C: Point, D: Point) -> bool:
        return all(vertex in self.vertices for vertex in [A, B, C, D])

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

    def discard(self):
        # print(f"discarding {self}")
        for vertex in self.vertices:
            try:
                vertex.in_tetra.remove(self)
            except ValueError:
                print(
                    f"somehow removing {self} from {vertex} failed : {vertex.in_tetra}"
                )
        for segment in self.segments:
            segment.remove_tetra(self)
        for triangle in self.triangles:
            triangle.remove_tetra(self)


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


def addPoint(tetrahedrization: list[Tetra], point: Point) -> list[Tetra]:
    containers: list[Tetra] = []
    for tet in tetrahedrization:
        if pointInBoundSphere(point, tet):
            containers.append(tet)
    containedFaces: list[Triangle] = []
    remainingFaces: list[Triangle] = []
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

    for tet in containers:
        for f in tet.triangles:
            f.checked = False

    newTet: list[Tetra] = []
    for face in remainingFaces:
        t = Tetra.make(point, *face.vertices)
        newTet.append(t)
    tetrahedrization = [
        tet for tet in tetrahedrization if not tet in containers
    ] + newTet

    for tet in containers:
        tet.discard()

    return tetrahedrization


def point_name(i: int):
    if i < 26:
        return chr(65 + i)
    else:
        s = ""
        while i:
            s = chr((97 if i >= 26 else 64) + i % 26) + s
            i = int(i / 26)
        return s


def coordsInTetra(point: Point, tetra: Tetra):
    loc = point.val - tetra.o
    return np.matmul(loc, tetra.iM)


def findLocalTetra(point: Point, tetrahedrization: list[Tetra]) -> Tetra:
    for t in tetrahedrization:
        if pointInTetra(point, t):
            return t
    return None


def run(points: list[tuple[str, list[float]]]) -> list[Tetra]:
    bigT: Tetra = Tetra(
        Point.make(0.0, 0.0, 100000000.0, "0"),
        Point.make(-200000000.0 * sqrt(2) / 3.0, 00000000.0, -100000000.0 / 3.0, "1"),
        Point.make(
            200000000.0 * sqrt(2) / 6.0,
            100000000.0 * sqrt(2.0 / 3.0),
            -100000000.0 / 3.0,
            "2",
        ),
        Point.make(
            200000000.0 * sqrt(2) / 6.0,
            -100000000.0 * sqrt(2.0 / 3.0),
            -100000000.0 / 3.0,
            "3",
        ),
    )

    tetrahedrization = [bigT]

    for (i, p) in enumerate(points):
        tetrahedrization = addPoint(
            tetrahedrization,
            Point.make(*(p[1]), p[0]),
        )

    tetrahedrization = [
        t
        for t in tetrahedrization
        if not any(p for p in t.vertices if p in bigT.vertices)
    ]

    return tetrahedrization
