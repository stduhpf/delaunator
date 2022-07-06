from cmath import sqrt
from functools import cmp_to_key
from typing import List, Tuple
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

Sphere = Tuple[np.matrix, float]

nameCounter = 0
pointNameCounter = 0


class Face_o:
    vert: np.matrix
    adj: Tuple


class Point_o:
    val: np.matrix
    name: str

    def __init__(self, val: np.matrix, name=None) -> None:
        global pointNameCounter
        assert val.shape == (1, 3)
        self.val = val
        if name is None:
            pointNameCounter += 1
            self.name = chr(pointNameCounter + 64)
        else:
            self.name = name
        super().__init__()

    def __repr__(self) -> str:
        if self.name is None:
            return object.__repr__(self.val)
        return self.name


class Tetrahedron_o:
    vertex_list: list[Point_o]
    vert: np.matrix
    o: np.matrix = None
    M: np.matrix = None
    iM: np.matrix = None
    boundingSphere: Sphere = (None, None)
    neighbors: List["Tetrahedron_o"] = []
    name: str

    def __init__(self, vertices: list[Point_o]):
        global nameCounter
        assert len(vertices) == 4
        self.vert = np.vstack([vertex.val.reshape(-1) for vertex in vertices])
        self.vertex_list = vertices
        self.o = vertices[3].val
        self.M = self.vert[:3] - self.o
        self.iM = np.linalg.inv(self.M)
        self.boundingSphere = getBoundingSphere(self)
        self.neighbors = []
        nameCounter += 1
        self.name = chr(nameCounter + 64)
        if nameCounter > 91:
            self.name = None

    def __repr__(self) -> str:
        if self.name is None:
            return object.__repr__(self)
        return self.name


class Tetrahedrization(List[Tetrahedron_o]):
    def replace(self, __old: Tetrahedron_o, __new: List[Tetrahedron_o]):
        for t in self:
            if __old in t.neighbors:
                t.neighbors.remove(__old)
                for n in __new:
                    if haveCommonFace(n, t):
                        t.neighbors.append(n)
                        n.neighbors.append(t)
        self.remove(__old)
        self += __new


def compareFaces(x, y) -> float:
    if x[0] == y[0]:
        if x[1] == y[1]:
            return x[2] - y[2]
        else:
            return x[1] - y[1]
    else:
        return x[0] - y[0]


def haveCommonFace(A: Tetrahedron_o, B: Tetrahedron_o) -> bool:
    Al = sorted(A.vert.tolist(), key=cmp_to_key(compareFaces))
    Bl = sorted(B.vert.tolist(), key=cmp_to_key(compareFaces))
    for i in range(4):
        a = Al.copy()
        del a[i]
        for j in range(4):
            b = Bl.copy()
            del b[j]
            if a == b:
                return True
    return False


def getBoundingSphere(tetrahedron: Tetrahedron_o) -> Sphere:
    o = tetrahedron.o
    # o = np.matmul(np.matrix([[1, 0, 0, 0]]), tetrahedron)
    M = tetrahedron.M
    iM = tetrahedron.iM
    k = np.diag(np.matmul(M, np.transpose(M))) * 0.5
    g = np.matmul(iM, k)
    return (g + o, np.linalg.norm(g))


def splitT(tetrahedron: Tetrahedron_o, point: Point_o) -> List[Tetrahedron_o]:
    assert point.val.shape == (1, 3)
    # assert isInsideT(
    #     point, tetrahedron
    # ), f"Point {point} is outside tetrahedron {tetrahedron}"
    global nameCounter

    t: List[Tetrahedron_o] = []
    for i in range(4):
        v = tetrahedron.vertex_list.copy()
        v[i] = point
        t.append(Tetrahedron_o(v))
    for nt in t:
        nt.neighbors = t.copy()
        nt.neighbors.remove(nt)
    return t


def add_point(tetrahedrization: Tetrahedrization, point: Point_o):
    for t in tetrahedrization:
        if isInsideBS(point, t):
            news = splitT(t, point)
            print(f"\nreplacing {t}...")
            tetrahedrization.replace(t, news)
            break


# "COLLISION" CHECKS:


def isInsideT(point: Point_o, tetrahedron: Tetrahedron_o) -> bool:
    assert point.val.shape == (1, 3)
    o = tetrahedron.o
    iM = tetrahedron.iM
    c = np.matmul(point.val - o, iM)
    return (c >= 0).all() and np.dot(c, [1, 1, 1]) <= 1


def isInsideS(point: Point_o, sphere: Sphere) -> bool:
    assert point.val.shape == (1, 3)
    (center, radius) = sphere
    return np.linalg.norm(point.val - center) <= radius


def isInsideBS(point: Point_o, tetrahedron: Tetrahedron_o) -> bool:
    return isInsideS(point, tetrahedron.boundingSphere)


# PLOTTING:


def scatterPointsDemo(t: Tetrahedron_o, ax: Axes, N=1000) -> None:
    rPoints = np.array([[0, 0, 0]])
    bPoints = np.array([[0, 0, 0]])
    gPoints = np.array([[0, 0, 0]])
    for _ in range(N):
        point = np.random.rand(1, 3) * 1.5 - 0.25
        in_bs = isInsideBS(point, t)
        in_t = isInsideT(point, t)

        if not in_bs and not in_t:
            rPoints = np.r_[rPoints, point]
        elif not in_t:
            bPoints = np.r_[bPoints, point]
        else:
            gPoints = np.r_[gPoints, point]
    pt = np.transpose(rPoints)
    ax.scatter(pt[0], pt[1], pt[2], c="r", marker=".")
    pt = np.transpose(gPoints)
    ax.scatter(pt[0], pt[1], pt[2], c="g", marker=".")
    pt = np.transpose(bPoints)
    ax.scatter(pt[0], pt[1], pt[2], c="b", marker=".")


def plotSphere(sphere: Sphere, ax: Axes) -> None:
    (c, r) = sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v) * r + c[0, 0]
    y = np.sin(u) * np.sin(v) * r + c[0, 1]
    z = np.cos(v) * r + c[0, 2]
    ax.plot_wireframe(x, y, z, color="r")


def plotTetrahedron(t: Tetrahedron_o, ax: Axes) -> None:
    print(f"\n{t}:\n vertices = {t.vertex_list},\n neighbors = {t.neighbors}")
    for p in t.vertex_list:
        if p.name == "rem":
            print(f"point t remove detected:{p} in tetrahedron : {t}")
            return
    tx = [
        t.vert[0, 0],
        t.vert[1, 0],
        t.vert[2, 0],
        t.vert[3, 0],
        t.vert[0, 0],
        t.vert[2, 0],
        t.vert[1, 0],
        t.vert[3, 0],
    ]
    ty = [
        t.vert[0, 1],
        t.vert[1, 1],
        t.vert[2, 1],
        t.vert[3, 1],
        t.vert[0, 1],
        t.vert[2, 1],
        t.vert[1, 1],
        t.vert[3, 1],
    ]
    tz = [
        t.vert[0, 2],
        t.vert[1, 2],
        t.vert[2, 2],
        t.vert[3, 2],
        t.vert[0, 2],
        t.vert[2, 2],
        t.vert[1, 2],
        t.vert[3, 2],
    ]
    ax.plot(tx, ty, tz)
    # ax.scatter(
    #     t.boundingSphere[0][0, 0],
    #     t.boundingSphere[0][0, 1],
    #     t.boundingSphere[0][0, 2],
    #     c="black",
    #     marker="x",
    # )


def plotTetrahedrization(tz: List[Tetrahedron_o], ax: Axes):
    for t in tz:
        plotTetrahedron(t, ax)


if __name__ == "__main__":

    bigT: Tetrahedron_o = Tetrahedron_o(
        [
            Point_o(np.real(np.matrix([[0.0, 0.0, 1.0]]) * 10), name="rem"),
            Point_o(
                np.real(np.matrix([[-2.0 * sqrt(2) / 3.0, 0.0, -1.0 / 3.0]]) * 10),
                name="rem",
            ),
            Point_o(
                np.real(
                    np.matrix([[2.0 * sqrt(2) / 6.0, sqrt(2.0 / 3.0), -1.0 / 3.0]]) * 10
                ),
                name="rem",
            ),
            Point_o(
                np.real(
                    np.matrix([[2.0 * sqrt(2) / 6.0, -sqrt(2.0 / 3.0), -1.0 / 3.0]])
                    * 10
                ),
                name="rem",
            ),
        ]
    )
    # print(bigT.vert)
    # (c, r) = bigT.boundingSphere
    # print(f"center: {c}, radius: {r}")
    tetrahedrization = Tetrahedrization([bigT])

    # splitT(bigT, np.matrix([[0.0, 1.0, 0.0]]))

    ax = plt.axes(projection="3d")
    # ax.set_aspect("equal")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    xs = []
    ys = []
    zs = []
    for _ in range(100):
        point = np.random.rand(1, 3) * 2.0 - 1.0
        xs.append(point[0, 0])
        ys.append(point[0, 1])
        zs.append(point[0, 2])
        add_point(tetrahedrization, Point_o(point))
    print(xs)
    ax.scatter(xs, ys, zs, marker="x")

    # plotTetrahedron(bigT, ax)
    # plotSphere(t.boundingSphere, ax)

    # scatterPointsDemo(t, ax)

    plotTetrahedrization(tetrahedrization, ax)
    plt.show()
