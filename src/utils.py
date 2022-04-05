"""
Utility Functions for Detecting Red Lights
"""
import os
import re
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw
from scipy import ndimage

CLUSTER_AREA = 10
MIN_AREA = 8
MIN_CONFIDENCE = 0.89

def rgb_to_hsv(image: Image.Image) -> Image.Image:
    """Convert Image from RGB to HSV"""
    return image.convert('HSV')


def visualize(I: Image.Image,
bounding_boxes: list[list[int]],
outline: str = "red",
save=None) -> None:
    """Visualize the bounding boxes in the image"""
    img = ImageDraw.Draw(I)
    for box in bounding_boxes:
        draw_box = (box[1], box[0], box[3], box[2])
        img.rectangle(draw_box, outline=outline)
    I.show()
    if save is not None:
        I.save(save)


def normalize_arr(arr: np.ndarray) -> np.ndarray:
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # minval = arr.min(axis=(0, 1), keepdims=True)
    # maxval = arr.max(axis=(0, 1), keepdims=True)
    # print(minval, maxval)
    # print(arr)
    # arr = (arr - minval) * 255.0 / (maxval - minval)
    # print(arr)
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr.astype("uint8")


def normalize_img(I: Image.Image) -> Image.Image:
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = np.asarray(I)
    return Image.fromarray(normalize_arr(arr).astype("uint8"), "RGB")


def demo_normalize(filename: str) -> None:
    img = Image.open(filename).convert('RGBA')
    new_img = normalize_img(I)
    new_img.save('out/normalized.png')


def mat_dot(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a.flatten(), b.flatten())


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def convolve_with_kernels(kernels: list[np.ndarray], I: np.ndarray, normalize: bool = True) -> np.ndarray:
    result = np.ones(I.shape[:2]) * -np.inf
    for k in kernels:
        H, W, n_channels = np.shape(k)
        (n_rows, n_cols, _) = np.shape(I)
        
        # Zero pad I for convolution
        left_pad = (W - 1) // 2
        right_pad = W - 1 - left_pad
        top_pad = (H - 1) // 2
        bot_pad = H - 1 - top_pad
        padded = np.zeros((n_rows + top_pad + bot_pad, n_cols + left_pad + right_pad, n_channels))
        padded[top_pad:top_pad + n_rows, left_pad:left_pad + n_cols, :] = I
        print(f"Padded shape: {padded.shape}")
        # print(f"k shape: {k.shape}")
        for row in range(0, padded.shape[0] - H + 1):
            for col in range(0, padded.shape[1] - W + 1):
                # print(f"col: {col}. W: {W}. col + W: {col + W}")
                x = padded[row:row + H, col:col + W, :]
                if normalize:
                    x = normalize_arr(x)
                # print(f"x shape: {x.shape}")
                # x = np.dot(k.flatten(), x.flatten())
                x = np.sum(k *x)
                result[row, col] = max(result[row, col], x)
    return result

def convolve_with_kernels2(kernels: list[np.ndarray], I: np.ndarray, normalize: bool = True) -> np.ndarray:
    result = np.ones(I.shape[:2]) * -np.inf
    I = I.astype("float")
    for k in kernels:
        H, W, _ = np.shape(k)

        # Zero pad I for convolution
        left_pad = (W - 1) // 2
        right_pad = W - 1 - left_pad
        top_pad = (H - 1) // 2
        bot_pad = H - 1 - top_pad
        padded = np.pad(I, ((top_pad, bot_pad), (left_pad, right_pad), (0, 0)), 'constant')

        # get all the windows into the image used for convolution
        padded = sliding_window_view(padded, k.shape)
        k = k.astype("float")

        for i in range(padded.shape[0]):
            # for j in range(padded.shape[1]):
            #     x = padded[i, j, 0, :, :, :]
            #     sim = cosine_similarity(k, x, k_norm=k_norm)
            #     result[i, j] = max(result[i, j], sim)
            result[i, :] = np.maximum(result[i, :], mass_cosine_similarity(k, padded[i, :, :, :, :]).reshape((-1)))
        # # cpmpute cosine similarity between kernel and each window
        # padded_norm = np.linalg.norm(padded)
        # k_norm = np.linalg.norm(k)
        # sim = (k * padded).sum(axis=(2, 3, 4)) / (k_norm * padded_norm)
        
        # # Update results if this kernel yields better similarity than previous ones
        # result = np.maximum(result, sim)
    return result
        

def mass_cosine_similarity(k: np.ndarray, X: np.ndarray) -> np.ndarray:
    assert k.shape == X.shape[-3:]
    m, i, l, w, c = X.shape
    X = X.reshape((m, i, l * w * c))
    X_norm = np.linalg.norm(X, axis=(-1))
    k_norm = np.linalg.norm(k)
    return X.dot(k.ravel()) / (X_norm * k_norm)


def cosine_similarity(k: np.ndarray, x: np.ndarray, k_norm=None, x_norm=None) -> float:
    if k_norm is None:
        k_norm = np.linalg.norm(k)
    if x_norm is None:
        x_norm = np.linalg.norm(x)

    return k.ravel().dot(x.ravel()) / (k_norm * x_norm)


def neighborhood(s: np.ndarray, loc: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    start_row = max(0, loc[0] - size)
    end_row = min(s.shape[0], loc[0] + size + 1)
    start_col = max(0, loc[1] - size)
    end_col = min(s.shape[1], loc[1] + size + 1)
    return (start_row, end_row, start_col, end_col)


def find_cluster(s: np.ndarray, start: tuple[int, int]) -> tuple[list[int], float]:
    tlr = brr = start[0]  # Top left row and bottom right row
    tlc = brc = start[1]  # Top left column and bottom right column
    total = 0
    num_tiles = 0

    stack = [start]
    while len(stack) > 0:
        curr = stack.pop()
        if s[curr] == 0: continue  # Already visited this

        total += s[curr]
        num_tiles += 1
        s[curr] = 0

        # Update bounding box area
        tlr = min(tlr, curr[0])
        brr = max(brr, curr[0])
        tlc = min(tlc, curr[1])
        brc = max(brc, curr[1])

        # Find nearby points to join into the cluster
        nsr, ner, nsc, nec = neighborhood(s, curr, CLUSTER_AREA)
        for i in range(nsr, ner):
            for j in range(nsc, nec):
                if s[i, j] != 0 and (i, j) not in stack:
                    stack.append((i, j))

    return [tlr, tlc, brr, brc], total / num_tiles



def score_clustering(s: np.ndarray) -> list[list[int]]:
    bounding_boxes = []

    while not (s == 0).all():
        # find highest point
        start = np.unravel_index(np.argmax(s, axis=None), s.shape)

        # find cluster around that point
        cluster_coords, confidence  = find_cluster(s, start)
        if confidence > MIN_CONFIDENCE:
        # if (cluster_coords[2] - cluster_coords[0]) \
        #     * (cluster_coords[3] - cluster_coords[1]) >= MIN_AREA:
            bounding_boxes.append(cluster_coords)

    return np.array(bounding_boxes).astype(int).tolist()


def downsample(I: Image.Image, s: int = 1) -> Image.Image:
    w, h, = I.size
    newsize = (w // s, h // s)
    return I.resize(newsize)
