"""
Utility Functions for Detecting Red Lights
"""
import os
import re
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw
from scipy import ndimage


def rgb_to_hsv(image: Image.Image) -> Image.Image:
    """Convert Image from RGB to HSV"""
    return image.convert('HSV')


def visualize(I: Image.Image,
bounding_boxes: list[list[int]],
outline: str = "red") -> None:
    """Visualize the bounding boxes in the image"""
    img = ImageDraw.Draw(I)
    for box in bounding_boxes:
        img.rectangle(tuple(box), outline=outline)
    I.show()


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


# def score_clustering(s: np.ndarray) -> list[int]:


def downsample(I: Image.Image, s: int = 1) -> Image.Image:
    w, h, = I.size
    newsize = (w // s, h // s)
    return I.resize(newsize)
