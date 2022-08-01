from math import floor, pow, sqrt
from PIL import Image
import numpy as np

from unay import Point, findLocalTetra, run
import matplotlib.pyplot as plt

# TODO: gamma correction, color spaces, accelerating structure

colors = {
    "GRASS": [127, 178, 56],
    "SAND": [247, 233, 163],
    "WOOL": [199, 199, 199],
    "FIRE": [255, 0, 0],
    "ICE": [160, 160, 255],
    "METAL": [167, 167, 167],
    "PLANT": [0, 124, 0],
    "SNOW": [255, 255, 255],
    "CLAY": [164, 168, 184],
    "DIRT": [151, 109, 77],
    "STONE": [112, 112, 112],
    "WOOD": [143, 119, 72],
    "WATER": [64, 64, 255],
    "QUARTZ": [255, 252, 245],
    "COLOR_ORANGE": [216, 127, 51],
    "COLOR_MAGENTA": [178, 76, 216],
    "COLOR_LIGHT_BLUE": [102, 153, 216],
    "COLOR_YELLOW": [229, 229, 51],
    "COLOR_LIGHT_GREEN": [127, 204, 25],
    "COLOR_PINK": [242, 127, 165],
    "COLOR_GRAY": [76, 76, 76],
    "COLOR_LIGHT_GRAY": [153, 153, 153],
    "COLOR_CYAN": [76, 127, 153],
    "COLOR_PURPLE": [127, 63, 178],
    "COLOR_BLUE": [51, 76, 178],
    "COLOR_BROWN": [102, 76, 51],
    "COLOR_GREEN": [102, 127, 51],
    "COLOR_RED": [153, 51, 51],
    "COLOR_BLACK": [25, 25, 25],
    "GOLD": [250, 238, 77],
    "DIAMOND": [92, 219, 213],
    "LAPIS": [74, 128, 255],
    "EMERALD": [0, 217, 58],
    "PODZOL": [129, 86, 49],
    "NETHER": [112, 2, 0],
    "TERRACOTTA_WHITE": [209, 177, 161],
    "TERRACOTTA_ORANGE": [159, 82, 36],
    "TERRACOTTA_MAGENTA": [149, 87, 108],
    "TERRACOTTA_LIGHT_BLUE": [112, 108, 138],
    "TERRACOTTA_YELLOW": [186, 133, 36],
    "TERRACOTTA_LIGHT_GREEN": [103, 117, 53],
    "TERRACOTTA_PINK": [160, 77, 78],
    "TERRACOTTA_GRAY": [57, 41, 35],
    "TERRACOTTA_LIGHT_GRAY": [135, 107, 98],
    "TERRACOTTA_CYAN": [87, 92, 92],
    "TERRACOTTA_PURPLE": [122, 73, 88],
    "TERRACOTTA_BLUE": [76, 62, 92],
    "TERRACOTTA_BROWN": [76, 50, 35],
    "TERRACOTTA_GREEN": [76, 82, 42],
    "TERRACOTTA_RED": [142, 60, 46],
    "TERRACOTTA_BLACK": [37, 22, 16],
    "CRIMSON_NYLIUM": [189, 48, 49],
    "CRIMSON_STEM": [148, 63, 97],
    "CRIMSON_HYPHAE": [92, 25, 29],
    "WARPED_NYLIUM": [22, 126, 134],
    "WARPED_STEM": [58, 142, 140],
    "WARPED_HYPHAE": [86, 44, 62],
    "WARPED_WART_BLOCK": [20, 180, 133],
    "DEEPSLATE": [100, 100, 100],
    "RAW_IRON": [216, 175, 147],
    "GLOW_LICHEN": [127, 167, 150],
}
multipliers = [0.71, 0.86, 1.0, 0.53]

multipliers = [0.71, 0.86, 1.0]

# multipliers = [0.86]

# TODO: Use CIELUV for closest color matching
def closestColor(
    colorList: list[tuple[str, list[float]]], color: list[float]
) -> list[float]:
    col = colorList[0][1]
    md = 100000.0

    for (_, c) in colorList:
        dx = color[0] - c[0]
        dy = color[1] - c[1]
        dz = color[2] - c[2]
        d = sqrt(dx * dx + dy * dy + dz * dz)
        if d < md:
            col = c
            md = d
    return col


def bayer(x: int, y: int, level: int) -> float:
    if level <= 0:
        return 0.5
    if level == 1:
        v = bayer.size[0] * bayer.size[1]
        x = x % bayer.size[0]
        y = y % bayer.size[1]
        return (bayer.data[x, y]) / 254.0 * (v - 1) / v
    return bayer(int(x / bayer.size[0]), int(y / bayer.size[1]), level - 1) / (
        bayer.size[0] * bayer.size[1]
    ) + bayer(x, y, 1)


def bayer_d(x: int, y: int, level: int) -> float:
    b = bayer(x, y, level)
    values = pow(bayer.size[0] * bayer.size[1], level)
    return b + 0.5 / values


def gammaInv(col: list[float]) -> list[float]:
    return rgb2ycocg([pow(c, 2.2) for c in col])


def rgb2ycocg(col: list[float]) -> list[float]:
    return [
        0.25 * col[0] + 0.5 * col[1] + 0.25 * col[2],
        0.5 * col[0] - 0.5 * col[2],
        -0.25 * col[0] + 0.5 * col[1] - 0.25 * col[2],
    ]


def gamma(col: list[float]) -> list[float]:
    col = ycocg2rgb(col)
    return [pow(max(c, 0), 1.0 / 2.2) for c in col]


def ycocg2rgb(col: list[float]) -> list[float]:
    return [
        col[0] + col[1] - col[2],
        col[0] + col[2],
        col[0] - col[1] - col[2],
    ]


def fract(x):
    return x - floor(x)


def ign(x, y):
    return fract(52.9829189 * fract(0.06711056 * float(x) + 0.00583715 * float(y)))


image = Image.open("tests/malty.png").resize((128, 199))


colorList = []
for (i, m) in enumerate(multipliers):
    colorList += [
        (
            c[0] + "|" + str(i),
            gammaInv(
                [
                    c[1][0] * m,
                    c[1][1] * m,
                    c[1][2] * m,
                ]
            ),
        )
        for c in colors.items()
    ]
# print(colorList)

palette = run(colorList)

print("palette generated")
# exit()

# ax = plt.axes(projection="3d")
# ax.set_box_aspect([1, 1, 1])
# ax.set_xlim(0, 255)
# ax.set_ylim(0, 255)
# ax.set_zlim(0, 255)

# plotTetrahedrization(palette, ax)
# plt.show()

bni = Image.open("blue.png")
bns = bni.size
blueNoise = bni.load()

bayer8 = Image.open("bayer8.png")
bayer.size = bayer8.size
bayer.data = bayer8.load()

newImageData = []

pixels = image.load()
newImage_b = Image.new(image.mode, image.size)
newImage_w = Image.new(image.mode, image.size)
newImage_ign = Image.new(image.mode, image.size)
newImage_bay = Image.new(image.mode, image.size)
newImage_close = Image.new(image.mode, image.size)

csv = []

for y in range(image.size[1]):
    csv.append([])
    for x in range(image.size[0]):
        # X = x / image.size[0]
        # color = gammaInv(
        #     [
        #         200 * X + 150 * (1.0 - X),
        #         25 * X + 160 * (1.0 - X),
        #         190 * X + 255 * (1.0 - X),
        #     ]
        # )
        color = gammaInv([float(p) for p in pixels[x, y]])
        # print(f"{pixels[x, y]} => {color}")
        p = Point.make(color[0], color[1], color[2])
        t = findLocalTetra(p, palette)
        if t is None:
            print(f"impossible color {color}, {t}")
            newImageData.append((0, 0, 0, 0))
            continue
        c = []
        xyz = t[1]
        # r = np.random.random()
        r = blueNoise[x % bns[0], y % bns[1]][0] / 255.0
        i = 0
        if r > xyz[0]:
            r -= xyz[0]
            if r > xyz[1]:
                r -= xyz[1]
                if r > xyz[2]:
                    i = 3
                else:
                    i = 2
            else:
                i = 1
        else:
            i = 0
        c = gamma(t[0].vertices[i].val.tolist())
        # print(t[0].vertices[i].name)
        newImage_b.putpixel((x, y), (int(c[0]), int(c[1]), int(c[2]), 255))

        r = np.random.random()
        i = 0
        if r > xyz[0]:
            r -= xyz[0]
            if r > xyz[1]:
                r -= xyz[1]
                if r > xyz[2]:
                    i = 3
                else:
                    i = 2
            else:
                i = 1
        else:
            i = 0
        c = gamma(t[0].vertices[i].val.tolist())
        newImage_w.putpixel((x, y), (int(c[0]), int(c[1]), int(c[2]), 255))

        r = ign(x, y)
        i = 0
        if r > xyz[0]:
            r -= xyz[0]
            if r > xyz[1]:
                r -= xyz[1]
                if r > xyz[2]:
                    i = 3
                else:
                    i = 2
            else:
                i = 1
        else:
            i = 0
        c = gamma(t[0].vertices[i].val.tolist())
        newImage_ign.putpixel((x, y), (int(c[0]), int(c[1]), int(c[2]), 255))
        name = t[0].vertices[i].name
        csv[y] += name.split("|")

        r = bayer_d(x, y, 2)
        if r > xyz[0]:
            r -= xyz[0]
            if r > xyz[1]:
                r -= xyz[1]
                if r > xyz[2]:
                    c = gamma(t[0].vertices[3].val.tolist())
                else:
                    c = gamma(t[0].vertices[2].val.tolist())
            else:
                c = gamma(t[0].vertices[1].val.tolist())
        else:
            c = gamma(t[0].vertices[0].val.tolist())
        newImage_bay.putpixel((x, y), (int(c[0]), int(c[1]), int(c[2]), 255))

        c = gamma(closestColor(colorList, color))
        newImage_close.putpixel((x, y), (int(c[0]), int(c[1]), int(c[2]), 255))
        # c = gamma((t[0].o + np.matmul(xyz, t[0].M)).tolist())
    # print("\n")
image.save("scaled_.png")
newImage_w.save("outw.png")
newImage_b.save("outb.png")
newImage_ign.save("outign.png")
newImage_bay.save("outbay.png")
newImage_close.save("out_close.png")

print("saved!\n")

# print(csv)
