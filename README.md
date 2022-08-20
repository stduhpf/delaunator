# Delaunator

Complicated tool to create dithered images with limited palette of colors using tetrahedrization of the colorspace.

Its intended use is to create minecraft map-art.

An implementation of this algorithm in a Minecraft mod might be a good idea in the future.

## Usage

Simply run `minecraftColors.py`to generate an image. To change the source image, and target resolution, you have to edit line `192` of this file.

By default the pallette correspond to the all the available colors for Minecraft map-art.

Tetrahedrization-related code is in `unay.py`.
