from ast import arg
from datetime import datetime
from math import floor, sqrt
import numpy as np

Sphere = tuple[np.ndarray, float]

bigTVert = []


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

        self.vertices: list[Point] = [A, B, C]
        for vertex in self.vertices:
            vertex.in_triangles.append(self)

        self.name: str = "{" + f"{A} {B} {C}" + "}"
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
        self.o: np.ndarray = self.vert[3]
        self.M: np.matrix = self.vert[:3] - self.o
        try:
            self.iM: np.matrix = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            print(f"coplanar: {self}")
            self.iM = None
        self.boundingSphere: Sphere = getBoundingSphere(self)

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

        self.vertices: list[Point] = [A, B, C, D]
        for vertex in self.vertices:
            vertex.in_tetra.append(self)

        self.name = f"({A} {B} {C} {D})"
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
        for triangle in self.triangles:
            triangle.remove_tetra(self)

    def removeVertexFromCoords(self, coords: np.ndarray, ignore: list[Point]):
        r = [v for v in self.vertices if v in ignore]
        normalize = False
        for k in r:
            i = self.vertices.index(k)
            if i != 3:
                coords[i] = 0
            else:
                normalize = True
        if normalize:
            coords = coords / coords.sum()
        return coords


def getBoundingSphere(tetrahedron: Tetra) -> Sphere:
    o = tetrahedron.o
    M = tetrahedron.M
    iM = tetrahedron.iM
    if iM is None:
        return None
    k = np.diag(np.matmul(M, np.transpose(M))) * 0.5
    g = np.matmul(iM, k)
    return (g + o, np.linalg.norm(g))


def pointInTetra(point: Point, tetrahedron: Tetra) -> bool:
    o = tetrahedron.o
    iM = tetrahedron.iM
    if iM is None:
        return False
    c = np.matmul(point.val - o, iM)
    return (c >= 0).all() and np.dot(c, [1, 1, 1]) <= 1


def pointInSphere(point: Point, sphere: Sphere) -> bool:
    (center, radius) = sphere
    assert point.val.shape == (3,) and center.shape == (3,)
    return np.linalg.norm(point.val - center) <= radius


def pointInBoundSphere(point: Point, tetrahedron: Tetra) -> bool:
    if tetrahedron.iM is None:
        return True
    return pointInSphere(point, tetrahedron.boundingSphere)


def addPoint(tetrahedrization: list[Tetra], point: Point) -> list[Tetra]:
    containers: list[Tetra] = []
    for tet in tetrahedrization:
        if pointInBoundSphere(point, tet):
            containers.append(tet)
    containedFaces: list[Triangle] = []
    remainingFaces: list[Triangle] = []
    for tet in containers:
        for face in tet.triangles:
            if not face.checked:
                face.checked = True
                contained = False
                for t in face.in_tetra:
                    if t != tet and t in containers:
                        containedFaces.append(face)
                        contained = True
                        break
                if not contained:
                    remainingFaces.append(face)

    for tet in containers:
        for face in tet.triangles:
            face.checked = False

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


def coordsInTetra(point: Point, tetra: Tetra) -> tuple[np.ndarray, bool]:
    loc = point.val - tetra.o
    c = np.matmul(loc, tetra.iM)
    return (c, (c > -5.0e-08).all() and np.dot(c, [1, 1, 1]) < 1 + 5e-08)


tree: list[list[list[Tetra]]] = [
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
    [
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], []],
    ],
]
lastUsed: Tetra = None


def fract(x):
    return x - floor(x)


def vtoi(*args):
    if len(args) == 0:
        return 0
    return int(fract(args[0]) > 0.5) + 2 * vtoi(*args[1:])


def findLocalTetra(
    point: Point, tetrahedrization: list[Tetra]
) -> tuple[Tetra, np.ndarray]:
    global lastUsed
    global bigTVert
    global tree
    if not (lastUsed is None):
        c = coordsInTetra(point, lastUsed)
        if c[1]:
            return (lastUsed, lastUsed.removeVertexFromCoords(c[0], bigTVert))
    tc = [
        vtoi(*(point.val / 65025.0).tolist()),
        vtoi(*(point.val * 0.5 / 65025.0).tolist()),
        vtoi(*(point.val * 0.25 / 65025.0).tolist()),
    ]
    valid = all(t in range(8) for t in tc)
    if valid:
        for t in tree[tc[0]][tc[1]][tc[2]]:
            c = coordsInTetra(point, t)
            if c[1]:
                lastUsed = t
                return (t, t.removeVertexFromCoords(c[0], bigTVert))
    for t in tetrahedrization:
        c = coordsInTetra(point, t)
        if c[1]:
            if valid:
                tree[tc[0]][tc[1]][tc[2]].append(t)
            lastUsed = t
            return (t, t.removeVertexFromCoords(c[0], bigTVert))
    return None


def run(points: list[tuple[str, list[float]]]) -> list:
    t0 = datetime.now()
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

    global bigTVert

    bigTVert = bigT.vertices

    tetrahedrization = [bigT]
    for p in points:
        tetrahedrization = addPoint(
            tetrahedrization,
            Point.make(*(p[1]), p[0]),
        )

    print(f"Execution time: {datetime.now()- t0}")

    return tetrahedrization
