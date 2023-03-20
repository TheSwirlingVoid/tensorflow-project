import os
import cv2
import random
import numpy as np
from numpy.core.multiarray import ndarray
from numpy.random.mtrand import randint

N = 200

def main() -> None:
	train_classes = os.listdir("archive/train")
	train_classes.remove(".DS_Store")
	
	images = list(os.scandir("archive/train/" + train_classes[0]))
	num_images = len(images)
	to_add = N - num_images
	
	to_alter = []
	if to_add > 0:
		if (to_add < num_images):
			to_alter = images[:to_add]
		elif (to_add >= num_images):
			for i in range(to_add):
				base_idx = i % num_images
				to_alter.append(randomly_altered(images[base_idx]))
                
main()

def randomly_altered(image) -> "ndarray":
    alterations = get_alterations()
    random_selection = random.randint(0,len(alterations))
    filter_function = alterations[random_selection]
    return filter_function

def get_alterations() -> list:
    filters = []

    filters.append(
            lambda im: cv2.GaussianBlur(im, (5,5), 0)
            )

    filters.append(
            lambda im: cv2.dilate(
                im,
                np.ones((5,5), np.uint8),
                iterations = 1
                )
            )

    return filters
