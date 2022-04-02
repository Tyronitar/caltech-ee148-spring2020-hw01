import os
import numpy as np
import json
from PIL import Image

from utils import normalize_img, convolve_with_kernels, sigmoid

KERNEL_BOXES = {
        "RL-010.jpg": [[122, 13, 173, 84], [320, 26, 349, 92]],
        "RL-021.jpg": [[281, 148, 289, 167]],
        "RL-028.jpg": [[308, 202, 317, 213], [344, 204, 348, 215]],
        "RL-036.jpg": [[216, 149, 232, 171], [296, 163, 305, 183]],
        "RL-050.jpg": [[335, 123, 348, 155]],
        "RL-248.jpg": [[498, 130, 518, 172]],
        "RL-274.jpg": [[315, 232, 322, 248]],
}

def detect_red_light(kernels: list[np.ndarray], I: np.ndarray) -> list[list[int]]:
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that the image is in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the blue channel
    I[:,:,2] is the green channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    # Get raw scores by convolving all kernels with image and taking max
    scores = convolve_with_kernels(kernels, I)

    # Normalize scores to be in [0, 1]
    scores = sigmoid(scores)

    return bounding_boxes


if __name__ == "__main__":
    # set the path to the downloaded data: 
    data_path = 'data/RedLights2011_Medium/RedLights2011_Medium'

    # Make Kernels

    kernels = []
    for img, boxes in KERNEL_BOXES.items():
        I = Image.open(os.path.join(data_path, img))
        for box in boxes:
            k_img = normalize_img(I.crop(tuple(box)))
            kernels.append(np.asarray(k_img))

    # set a path for saving predictions: 
    preds_path = 'out/hw01_preds' 
    os.makedirs(preds_path,exist_ok=True) # create directory if needed 

    # get sorted list of files: 
    file_names = sorted(os.listdir(data_path)) 

    # remove any non-JPEG files: 
    file_names = [f for f in file_names if '.jpg' in f] 

    preds = {}
    # for i in range(len(file_names)):
    for i in range(1):
        
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))
        arr = np.asarray(I)
        
        preds[file_names[i]] = detect_red_light(kernels[:1], arr)

    # visualize(data_path, file_names[0],preds[file_names[0]])

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds.json'),'w') as f:
        json.dump(preds,f)
