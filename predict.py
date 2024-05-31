import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import math

from siamese import Siamese

if __name__ == "__main__":
    model = Siamese()

    while True:
        image_1 = input('Input image_1 filename:')
        path = os.path.join("img", image_1)
        try:
            image_1 = Image.open(path)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        path = os.path.join("members", image_2)
        try:
            image_2 = Image.open(path)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.detect_image(image_1,image_2)

        print(probability.item())
