from time import sleep
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from unay import *


# PLOTTING:


def plotSphere(sphere: Sphere, ax: Axes) -> None:
    (c, r) = sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v) * r + c[0, 0]
    y = np.sin(u) * np.sin(v) * r + c[0, 1]
    z = np.cos(v) * r + c[0, 2]
    ax.plot_wireframe(x, y, z, color="r")


def plotSphere(sphere: Sphere, ax: Axes) -> None:
    (c, r) = sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v) * r + c[0]
    y = np.sin(u) * np.sin(v) * r + c[1]
    z = np.cos(v) * r + c[2]
    ax.plot_wireframe(x, y, z, color="r")


def mix(a: float, b: float) -> float:
    k = 0.25
    return a * (1 - k) + b * k


def plotTetrahedron(t: Tetra, ax: Axes, ignore: list[Point] = []) -> None:
    # print(f"\n{t}:\n vertices = {t.vertices}")
    for p in t.vertices:
        if p in ignore:
            # print(f"point to remove detected:{p} in tetrahedron : {t}")
            return
    c = sum(vertex.val for vertex in t.vertices) * 0.25
    tx = [
        mix(t.vert[0, 0], c[0]),
        mix(t.vert[1, 0], c[0]),
        mix(t.vert[2, 0], c[0]),
        mix(t.vert[3, 0], c[0]),
        mix(t.vert[0, 0], c[0]),
        mix(t.vert[2, 0], c[0]),
        mix(t.vert[1, 0], c[0]),
        mix(t.vert[3, 0], c[0]),
    ]
    ty = [
        mix(t.vert[0, 1], c[1]),
        mix(t.vert[1, 1], c[1]),
        mix(t.vert[2, 1], c[1]),
        mix(t.vert[3, 1], c[1]),
        mix(t.vert[0, 1], c[1]),
        mix(t.vert[2, 1], c[1]),
        mix(t.vert[1, 1], c[1]),
        mix(t.vert[3, 1], c[1]),
    ]
    tz = [
        mix(t.vert[0, 2], c[2]),
        mix(t.vert[1, 2], c[2]),
        mix(t.vert[2, 2], c[2]),
        mix(t.vert[3, 2], c[2]),
        mix(t.vert[0, 2], c[2]),
        mix(t.vert[2, 2], c[2]),
        mix(t.vert[1, 2], c[2]),
        mix(t.vert[3, 2], c[2]),
    ]
    ax.plot(tx, ty, tz)
    # plotSphere(t.boundingSphere, ax)


def plotTetrahedrization(tz: list[Tetra], ax: Axes, ignore: list[Point] = []):
    for t in tz:
        plotTetrahedron(t, ax, ignore)


if __name__ == "__main__":
    bigT: Tetra = Tetra(
        Point.make(0.0, 0.0, 10.0, "0"),
        Point.make(-20.0 * sqrt(2) / 3.0, 0.0, -10.0 / 3.0, "1"),
        Point.make(20.0 * sqrt(2) / 6.0, 10.0 * sqrt(2.0 / 3.0), -10.0 / 3.0, "2"),
        Point.make(20.0 * sqrt(2) / 6.0, -10.0 * sqrt(2.0 / 3.0), -10.0 / 3.0, "3"),
    )

    tetrahedrization = [bigT]

    ax = plt.axes(projection="3d")
    # ax.set_aspect("equal")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # xs = []
    # ys = []
    # zs = []
    for i in range(128):
        c = np.random.rand(1, 3) * 2.0 - 1.0
        tetrahedrization = addPoint(
            tetrahedrization,
            Point.make(c[0, 0], c[0, 1], c[0, 2], point_name(i)),
        )
        # plt.pause(1)
    plotTetrahedrization(tetrahedrization, ax, bigT.vertices)
    # print("\n")
    # print(xs)
    # ax.scatter(xs, ys, zs, marker="x")

    # plotTetrahedron(bigT, ax)
    # plotSphere(t.boundingSphere, ax)

    # scatterPointsDemo(t, ax)

    # plotTetrahedrization(tetrahedrization, ax, bigT.vertices)
    plt.show()
    counter = 0
    for _ in range(500):
        c = np.random.rand(1, 3) * 2.0 - 1.0

        testPoint = Point.make(c[0, 0], c[0, 1], c[0, 2], "test")
        # print(f"\ntestPoint : {testPoint.val}")

        hit = 0

        hits = []
        # ax.set_aspect("equal")
        for tet in tetrahedrization:
            if pointInTetra(testPoint, tet):
                hit += 1
                hits.append(tet)
                # plotTetrahedron(tet, ax2)
                # print(f"contained in {tet} :\n {tet.vert}\n")
        # print(hit)
        if hit > +2:
            print(f"Not ok : {hit}, hits in {hits}")
            counter += 1
    if counter == 0:
        print("SUCCESS!")
