"""
Utility Functions for Detecting Red Lights
"""
import os
import numpy as np
from PIL import Image, ImageDraw
from torch import norm, normal


def rgb_to_hsv(image: Image.Image) -> Image.Image:
    """Convert Image from RGB to HSV"""
    return image.convert('HSV')


def visualize(I: Image.Image,
bounding_boxes: list[list[int]],
outline: str = "red") -> None:
    """Visualize the bounding boxes in the image"""
    img = ImageDraw.Draw(I)
    for box in bounding_boxes:
        img.rectangle(box, outline=outline)
    I.show()

def normalize(I: Image.Image) -> Image.Image:
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = np.asarray(I)
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return Image.fromarray(arr.astype("uint8"), "RGB")

def demo_normalize(filename: str) -> None:
    img = Image.open(filename).convert('RGBA')
    new_img = normalize(I)
    new_img.save('out/normalized.png')


if __name__ == "__main__":
    data_path = 'data/RedLights2011_Medium/RedLights2011_Medium'
    KERNEL_BOXES = {
        "RL-010.jpg": [[122, 13, 173, 84], [320, 26, 349, 92]],
        "RL-021.jpg": [[281, 148, 289, 167]],
        "RL-028.jpg": [[308, 202, 317, 213], [344, 204, 348, 215]],
        "RL-036.jpg": [[216, 149, 232, 171], [296, 163, 305, 183]],
        "RL-050.jpg": [[335, 123, 348, 155]],
        "RL-248.jpg": [[498, 130, 518, 172]],
        "RL-274.jpg": [[315, 232, 322, 248]],
    }
    KERNELS = []
    for img, boxes in KERNEL_BOXES.items():
        I = Image.open(os.path.join(data_path, img))
        for box in boxes:
            k_img = normalize(I.crop(tuple(box)))
            KERNELS.append(k_img)

