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


def plotTetrahedron(t: Tetra, ax: Axes, ignore: list[Point] = []) -> None:
    print(f"\n{t}:\n vertices = {t.vertices}")
    for p in t.vertices:
        if p in ignore:
            print(f"point to remove detected:{p} in tetrahedron : {t}")
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


def plotTetrahedrization(tz: list[Tetra], ax: Axes, ignore: list[Point] = []):
    for t in tz:
        plotTetrahedron(t, ax, ignore)


if __name__ == "__main__":

    bigT: Tetra = Tetra(
        Point.make(0.0, 0.0, 100.0, "0"),
        Point.make(-200.0 * sqrt(2) / 3.0, 0.0, -100.0 / 3.0, "1"),
        Point.make(200.0 * sqrt(2) / 6.0, 100.0 * sqrt(2.0 / 3.0), -100.0 / 3.0, "2"),
        Point.make(200.0 * sqrt(2) / 6.0, -100.0 * sqrt(2.0 / 3.0), -100.0 / 3.0, "3"),
    )

    # print(bigT.vert)
    # (c, r) = bigT.boundingSphere
    # print(f"center: {c}, radius: {r}")
    tetrahedrization = [bigT]

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
    for i in range(100):
        c = np.random.rand(1, 3) * 2.0 - 1.0
        xs.append(c[0, 0])
        ys.append(c[0, 1])
        zs.append(c[0, 2])
        tetrahedrization = addPoint(
            tetrahedrization,
            Point.make(c[0, 0], c[0, 1], c[0, 2], chr(i + 65)),
        )
        # print("\n")
    print(xs)
    ax.scatter(xs, ys, zs, marker="x")

    # plotTetrahedron(bigT, ax)
    # plotSphere(t.boundingSphere, ax)

    # scatterPointsDemo(t, ax)

    plotTetrahedrization(tetrahedrization, ax, bigT.vertices)
    plt.show()

    c = np.random.rand(1, 3) * 2.0 - 1.0

    testPoint = Point.make(c[0, 0], c[0, 1], c[0, 2], "test")

    hit = 0

    for tet in tetrahedrization:
        if pointInTetra(testPoint, tet):
            hit += 1
    print(hit)
    assert hit < 2
