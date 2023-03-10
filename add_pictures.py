import os
import cv2

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

def randomly_altered(image):
    pass
