from PIL import Image
import numpy as np
from dela import plotTetrahedrization

from unay import Point, coordsInTetra, findLocalTetra, run
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
    "WATER": [64, 64, 255],
    "QUARTZ": [255, 252, 245],
    "COLOR_ORANGE": [216, 127, 51],
    "COLOR_MAGENTA": [178, 76, 216],
    "COLOR_LIGHT_BLUE": [102, 153, 216],
    "COLOR_YELLOW": [229, 229, 51],
    "COLOR_LIGHT_GREEN": [127, 204, 25],
    "COLOR_PINK": [242, 127, 165],
    "COLOR_GRAY": [76, 76, 76],
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

colorList = []
for (i, m) in enumerate(multipliers):
    colorList += [
        (
            c[0] + str(i),
            [
                c[1][0] * m,
                c[1][1] * m,
                c[1][2] * m,
            ],
        )
        for c in colors.items()
    ]
# print(colorList)

palette = run([c for c in colors.items()])

print("palette generated")

# ax = plt.axes(projection="3d")
# ax.set_box_aspect([1, 1, 1])
# ax.set_xlim(0, 255)
# ax.set_ylim(0, 255)
# ax.set_zlim(0, 255)

# plotTetrahedrization(palette, ax)
# plt.show()

image = Image.open("tests/img.PNG").resize((128, 128))


newImageData = []
for color in image.getdata():
    p = Point.make(color[0], color[1], color[2])
    t = findLocalTetra(p, palette)
    if t is None:
        newImageData.append((0, 0, 0, 0))
        continue
    xyz = coordsInTetra(p, t)
    r = np.random.random()
    c = color
    if r > xyz[0]:
        r -= xyz[0]
        if r > xyz[1]:
            r -= xyz[1]
            if r > xyz[2]:
                c = (*(t.vertices[3].val).astype(int), 255)
            else:
                c = (*(t.vertices[2].val).astype(int), 255)
        else:
            c = (*(t.vertices[1].val).astype(int), 255)
    else:
        c = (*(t.vertices[0].val).astype(int), 255)
    # print(f"{(*t.o, 255)}  {color}")
    newImageData.append(c)
newImage = Image.new(image.mode, image.size)
newImage.putdata(newImageData)

newImage.save("out_.png")
print("saved!")
