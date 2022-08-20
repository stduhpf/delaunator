#!/usr/bin/env python3

from math import floor, pow, sqrt
from PIL import Image
import numpy as np

from unay import Point, findLocalTetra, run

# TODO implement NBT abd schematic generation from https://github.com/rebane2001/mapartcraft

# TODO: accelerating structures

colors = {
    "GRASS":                  [127, 178, 56 ],
    "SAND":                   [247, 233, 163],
    "WOOL":                   [199, 199, 199],
    "FIRE":                   [255, 0  , 0  ],
    "ICE":                    [160, 160, 255],
    "METAL":                  [167, 167, 167],
    "PLANT":                  [0  , 124, 0  ],
    "SNOW":                   [255, 255, 255],
    "CLAY":                   [164, 168, 184],
    "DIRT":                   [151, 109, 77 ],
    "STONE":                  [112, 112, 112],
    "WOOD":                   [143, 119, 72 ],
    "WATER":                  [64 , 64 , 255],
    "QUARTZ":                 [255, 252, 245],
    "COLOR_ORANGE":           [216, 127, 51 ],
    "COLOR_MAGENTA":          [178, 76 , 216],
    "COLOR_LIGHT_BLUE":       [102, 153, 216],
    "COLOR_YELLOW":           [229, 229, 51 ],
    "COLOR_LIGHT_GREEN":      [127, 204, 25 ],
    "COLOR_PINK":             [242, 127, 165],
    "COLOR_GRAY":             [76 , 76 , 76 ],
    "COLOR_LIGHT_GRAY":       [153, 153, 153],
    "COLOR_CYAN":             [76 , 127, 153],
    "COLOR_PURPLE":           [127, 63 , 178],
    "COLOR_BLUE":             [51 , 76 , 178],
    "COLOR_BROWN":            [102, 76 , 51 ],
    "COLOR_GREEN":            [102, 127, 51 ],
    "COLOR_RED":              [153, 51 , 51 ],
    "COLOR_BLACK":            [25 , 25 , 25 ],
    "GOLD":                   [250, 238, 77 ],
    "DIAMOND":                [92 , 219, 213],
    "LAPIS":                  [74 , 128, 255],
    "EMERALD":                [0  , 217, 58 ],
    "PODZOL":                 [129, 86 , 49 ],
    "NETHER":                 [112, 2  , 0  ],
    "TERRACOTTA_WHITE":       [209, 177, 161],
    "TERRACOTTA_ORANGE":      [159, 82 , 36 ],
    "TERRACOTTA_MAGENTA":     [149, 87 , 108],
    "TERRACOTTA_LIGHT_BLUE":  [112, 108, 138],
    "TERRACOTTA_YELLOW":      [186, 133, 36 ],
    "TERRACOTTA_LIGHT_GREEN": [103, 117, 53 ],
    "TERRACOTTA_PINK":        [160, 77 , 78 ],
    "TERRACOTTA_GRAY":        [57 , 41 , 35 ],
    "TERRACOTTA_LIGHT_GRAY":  [135, 107, 98 ],
    "TERRACOTTA_CYAN":        [87 , 92 , 92 ],
    "TERRACOTTA_PURPLE":      [122, 73 , 88 ],
    "TERRACOTTA_BLUE":        [76 , 62 , 92 ],
    "TERRACOTTA_BROWN":       [76 , 50 , 35 ],
    "TERRACOTTA_GREEN":       [76 , 82 , 42 ],
    "TERRACOTTA_RED":         [142, 60 , 46 ],
    "TERRACOTTA_BLACK":       [37 , 22 , 16 ],
    "CRIMSON_NYLIUM":         [189, 48 , 49 ],
    "CRIMSON_STEM":           [148, 63 , 97 ],
    "CRIMSON_HYPHAE":         [92 , 25 , 29 ],
    "WARPED_NYLIUM":          [22 , 126, 134],
    "WARPED_STEM":            [58 , 142, 140],
    "WARPED_HYPHAE":          [86 , 44 , 62 ],
    "WARPED_WART_BLOCK":      [20 , 180, 133],
    "DEEPSLATE":              [100, 100, 100],
    "RAW_IRON":               [216, 175, 147],
    "GLOW_LICHEN":            [127, 167, 150],
}
multipliers = [0.71, 0.86, 1.0, 0.53]

multipliers = [0.71, 0.86, 1.0]

# multipliers = [0.86]



def yCoCg2cieLab(col: list[float]) -> list[float]:
    RGBToXYZ = (np.matrix([[0.4124, 0.2126, 0.0193], 
                           [0.3576, 0.7152, 0.1192],
                           [0.1805, 0.0722, 0.9505]]))

    col = np.matmul(np.array(ycocg2rgb(col)),RGBToXYZ).tolist()[0]

    def f(t: float) -> float:
        return pow(t, 1/3) if (t > pow(6/29, 3)) else 1/3*(29/6)*(29/6) * t + 4/29

    w = [95.0489,100,108.8840]

    return [116 * f(col[1]/w[1]) - 16, 500 * (f(col[0]/w[0]) - f(col[1]/w[1])),200 * (f(col[1]/w[1]) - f(col[2]/w[2]))]


def yCoCg2cieLuv(col: list[float]) -> list[float]:

    RGBToXYZ = (np.matrix([[0.4124, 0.2126, 0.0193], 
                           [0.3576, 0.7152, 0.1192],
                           [0.1805, 0.0722, 0.9505]]))

    col = np.matmul(np.array(ycocg2rgb(col)),RGBToXYZ).tolist()[0]

    def f(t: float) -> float:
        return pow(t, 1/3) if (t > pow(6/29, 3)) else 1/3*(29/6)*(29/6) * t + 4/29

    w = [95.0489,100,108.8840]
    L = 116*f(col[1]/w[2])-16
    u = 4*col[0]/(col[0]+15*col[1]+3*col[2])
    u0 = 4*w[0]/(w[0]+15*w[1]+3*w[2])
    v = 9*col[1]/(col[0]+15*col[1]+3*col[2])
    v0 = 9*w[1]/(w[0]+15*w[1]+3*w[2])


    return [L, 13*L*(u-u0), 13*L*(v-v0)]

def closestColor(
    colorList: list[tuple[str, list[float]]], color: list[float]
) -> list[float]:
    luvColor = yCoCg2cieLuv(color)
    col = colorList[0][1]
    md = 100000.0

    for (_, c, luvc) in colorList:
        dx = luvColor[0] - luvc[0]
        dy = luvColor[1] - luvc[1]
        dz = luvColor[2] - luvc[2]
        d = sqrt(dx * dx + dy * dy + dz * dz)
        if d < md:
            # print(f'{col}->{c} ({luvColor}->{luvc})')
            col = c
            md = d
    return col


def bayer(x: int, y: int, level: int) -> float:
    if level <= 0:
        return 0.
    if level == 1:
        return fract(y*y*.75+x*.5)
    if(level < 3):
        return bayer(int(x/2), int(y/2), level-1) / 4 + bayer(x, y, 1)
    if level == 3:
        v = bayer.size[0] * bayer.size[1]
        x = x % bayer.size[0]
        y = y % bayer.size[1]
        return (bayer.data[x, y]) / 254.0 * (v - 1) / v
    return bayer(int(x / bayer.size[0]), int(y / bayer.size[1]), level - 3) / (
        bayer.size[0] * bayer.size[1]
    ) + bayer(x, y, 3)


def bayer_d(x: int, y: int, level: int) -> float:
    b = bayer(x, y, level)
    values = pow(bayer.size[0] * bayer.size[1], level)
    return b + 0.5 / values


def gammaInv(col: list[float]) -> list[float]:
    return rgb2ycocg([pow(c / 255.0, 2.2) for c in col])


def rgb2ycocg(col: list[float]) -> list[float]:
    return [
         0.25  * col[0] + 0.5 * col[1] + 0.25 * col[2],
         0.5   * col[0]                - 0.5  * col[2],
        -0.25  * col[0] + 0.5 * col[1] - 0.25 * col[2],
    ]


def gamma(col: list[float]) -> list[float]:
    col = ycocg2rgb(col)
    return [pow(max(c, 0), 1.0 / 2.2)*255.0 for c in col]


def ycocg2rgb(col: list[float]) -> list[float]:
    return [
        col[0] + col[1] - col[2],
        col[0]          + col[2],
        col[0] - col[1] - col[2],
    ]


def fract(x):
    return x - floor(x)


def ign(x, y):
    return fract(52.9829189 * fract(0.06711056 * float(x) + 0.00583715 * float(y)))


image = Image.open("tests/lena_std.tif").resize((128, 128))


colorList = []
for (i, m) in enumerate(multipliers):
    colorList += [
        (
            c[0] + "|" + str(i),
            gammaInv(
                [float(ch) * m for ch in c[1]]
            ),
        )
        for c in colors.items()
    ]
# print(colorList)

palette = run(colorList)

print(f'palette generated, {len(palette)} cells')
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
newImage_b     = Image.new(image.mode, image.size)
newImage_w     = Image.new(image.mode, image.size)
newImage_ign   = Image.new(image.mode, image.size)
newImage_bay   = Image.new(image.mode, image.size)
newImage_close = Image.new(image.mode, image.size)

csv = []

offset = 0

luvcolors = [(n, c, yCoCg2cieLuv(c)) for (n, c) in colorList]

for y in range(image.size[1]):
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
        c = gamma(closestColor(luvcolors, color))
        newImage_close.putpixel((x, y), (int(c[0]), int(c[1]), int(c[2]), 255))

newImage_close.save("out_close.png")
print('saved closest')

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
            print(f"impossible color { gamma(color) }, {t}")
            newImageData.append((0, 0, 0, 0))
            continue
        c = []
        xyz = t[1]
        r = 1.-xyz[0]-xyz[1]-xyz[2]
        while   min(xyz[0], xyz[1], xyz[2], r) < offset:
            m = min(xyz[0], xyz[1], xyz[2], r)
            d = 1 - m
            if m == xyz[0]:
                xyz[1] += xyz[0]*xyz[1]/d
                xyz[2] += xyz[0]*xyz[2]/d
                r      += xyz[0]*r     /d
                xyz[0] = 2
            elif m == xyz[1]:
                xyz[0] += xyz[1]*xyz[0]/d
                xyz[2] += xyz[1]*xyz[2]/d
                r      += xyz[1]*r     /d
                xyz[1] = 2
            elif m == xyz[2]:
                xyz[0] += xyz[2]*xyz[0]/d
                xyz[1] += xyz[2]*xyz[1]/d
                r      += xyz[2]*r     /d
                xyz[2] = 2
            else:
                xyz[0] += r     *xyz[0]/d
                xyz[1] += r     *xyz[1]/d
                xyz[2] += r     *xyz[2]/d
                r      = 2 
        xyz = [0 if v > 1 else v for v in xyz]
        r = (blueNoise[x % bns[0], y % bns[1]][0] + 0.5) / 256.0
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
        # name = t[0].vertices[i].name
        # csv[y] += name.split("|")

        r = bayer_d(x, y, 7)
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

        # c = gamma((t[0].o + np.matmul(xyz, t[0].M)).tolist())
    # print("\n")
image.save(         "scaled_.png"  )
newImage_w.save(    "outw.png"     )
newImage_b.save(    "outb.png"     )
newImage_ign.save(  "outign.png"   )
newImage_bay.save(  "outbay.png"   )

print("saved!\n")

# print(csv)
