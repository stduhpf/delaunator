from asyncio.windows_events import NULL
from dataclasses import dataclass
from tkinter import CENTER
from tkinter.messagebox import NO
from typing import Tuple
from xmlrpc.client import Boolean
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import numpy as np

Sphere = Tuple[np.matrix, float]


class Tetrahedron:
    vert: np.matrix
    o: np.matrix = NULL
    M: np.matrix = NULL
    iM: np.matrix = NULL
    boundingSphere: Sphere = (NULL, NULL)

    def __init__(self, vertices: np.matrix):
        assert vertices.shape == (4, 3)
        self.vert = vertices
        self.o = vertices[3]
        self.M = vertices[:3] - self.o
        self.iM = np.linalg.inv(self.M)
        self.boundingSphere = getBoundingSphere(self)


def getBoundingSphere(tetrahedron: Tetrahedron) -> Sphere:
    o = tetrahedron.o
    # o = np.matmul(np.matrix([[1, 0, 0, 0]]), tetrahedron)
    M = tetrahedron.M
    iM = tetrahedron.iM
    k = np.diag(np.matmul(M, np.transpose(M))) * 0.5
    g = np.matmul(iM, k)
    return (g + o, np.linalg.norm(g))


# "COLLISION" CHECKS:


def isInsideT(point: np.matrix, tetrahedron: Tetrahedron) -> Boolean:
    assert point.shape == (1, 3)
    o = tetrahedron.o
    iM = tetrahedron.iM
    c = np.matmul(point - o, iM)
    return (c >= 0).all() and np.dot(c, [1, 1, 1]) <= 1


def isInsideS(point: np.matrix, sphere: Sphere) -> Boolean:
    (center, radius) = sphere
    assert point.shape == (1, 3)
    return np.linalg.norm(point - center) <= radius


def isInsideBS(point: np.matrix, tetrahedron: Tetrahedron) -> Boolean:
    return isInsideS(point, tetrahedron.boundingSphere)


# PLOTTING:


def scatterPointsDemo(t: Tetrahedron, ax: Axes, N=1000) -> None:
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
    # pt = np.transpose(bPoints)
    # ax.scatter(pt[0], pt[1], pt[2], c="b", marker=".")


def plotTetrahedron(t: Tetrahedron, ax: Axes) -> None:
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
    ax.scatter(
        t.boundingSphere[0][0, 0],
        t.boundingSphere[0][0, 1],
        t.boundingSphere[0][0, 2],
        c="black",
        marker="x",
    )


if __name__ == "__main__":
    t: Tetrahedron = Tetrahedron(
        np.matrix([[0, 0, 0], [1, 0.5, 0], [0, 1, 1], [1, 0, 1]])
    )
    print(t.vert)
    (c, r) = getBoundingSphere(t)
    print(f"center: {c}, radius: {r}")
    ax = plt.axes(projection="3d")

    scatterPointsDemo(t, ax)

    plt.show()
    pass
