from run_predictions import detect_red_light, KERNEL_BOXES
from utils import *

FILE_NAMES = ["RL-010.jpg", "RL-012.jpg", "RL-061.jpg", "RL-167.jpg"]

if __name__ == '__main__':
    data_path = 'data/RedLights2011_Medium/RedLights2011_Medium'
    visual_path = 'out/report' 
    os.makedirs(visual_path,exist_ok=True) # create directory if needed 
    # Make Kernels

    kernels = []
    for img, boxes in KERNEL_BOXES.items():
        I = Image.open(os.path.join(data_path, img))
        for box in boxes:
            k_img = normalize_img(I.crop(tuple(box)))
            k_img = downsample(k_img, 2)
            kernels.append(np.asarray(k_img))
    
    kernels.append(np.asarray(downsample(Image.fromarray(kernels[0]), 2)))
    kernels.append(np.asarray(downsample(Image.fromarray(kernels[1]), 2)))

    for name in FILE_NAMES:
        # read image using PIL:
        I = Image.open(os.path.join(data_path,name))
        Id = downsample(I, 2)
        arr = np.asarray(Id)
        
        bounding_boxes = detect_red_light(kernels[:], arr)
        bounding_boxes = (np.array(bounding_boxes) * 2).tolist()
        visualize(I, bounding_boxes, save=os.path.join(visual_path,name))
 
