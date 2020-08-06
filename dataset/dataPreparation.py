import numpy as np
import math


def pad_or_crop(image, target_size, mode = 'symmetric', constant_values=(0)):
    bigger = max(image.shape[0], image.shape[1], target_size)

    pad_x = float(bigger - image.shape[0])
    pad_y = float(bigger - image.shape[1])


    pad_width_x = (int(math.floor(pad_x / 2)), int(math.ceil(pad_x / 2)))
    pad_width_y = (int(math.floor(pad_y / 2)), int(math.ceil(pad_y / 2)))

    if (target_size > image.shape[0]) & (target_size > image.shape[1]):
        return np.pad(image, (pad_width_x, pad_width_y), mode)
    else:
        if bigger > image.shape[1]:
            temp_image = np.pad(image, (pad_width_y), mode)
        else:
            if bigger > image.shape[0]:
                temp_image = np.pad(image, (pad_width_x), mode)
            else:
                temp_image = image
        return temp_image[int((temp_image.shape[0] - target_size)/2):int((temp_image.shape[0] + target_size)/2),int((temp_image.shape[1] - target_size)/2):int((temp_image.shape[1] + target_size)/2)]


