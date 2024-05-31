import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import math

from siamese import Siamese

if __name__ == "__main__":
    model = Siamese()

    member_img = []
    members_path = "members"
    for member in os.listdir(members_path):
        member_path = os.path.join(members_path, member)
        img = Image.open(member_path)
        member_img.append(img)

    student_img = []
    students_path = "img"
    for student in os.listdir(students_path):
        student_path = os.path.join(students_path, student)
        img = Image.open(student_path)
        student_img.append(img)

    Score = np.zeros((len(student_img), len(member_img)))

    for i in range(len(student_img)):
        for j in range(len(member_img)):
            probability = model.detect_image(student_img[i],member_img[j])
            Score[i][j] = probability.item()
    # print(Score)
    
    sx = {}
    sy = {}
    while(True):
        row_col_index=np.unravel_index(np.argmax(Score),Score.shape)
        x = row_col_index[0]
        y = row_col_index[1]
        if x not in sx and y not in sy:
            sx[x] = Score[x][y]
            sy[y] = Score[x][y]
            # plt.subplot(1, 2, 1)
            # plt.imshow(np.array(student_img[x]))

            # plt.subplot(1, 2, 2)
            # plt.imshow(np.array(member_img[y]))
            # plt.text(-12, -12, 'Similarity:%.3f' % Score[x][y], ha='center', va= 'bottom',fontsize=11)
            # plt.show()
        Score[x][y] = 0
        if len(sx) == len(student_img):
            break
    # sx = dict(sorted(sx.items()))
    # sy = dict(sorted(sy.items()))
    # print(sx)
    print(sy)