import os
import imageio
import numpy as np
import sys


def main(img_dir):
    face_img = []
    img = os.listdir(img_dir)
    img.sort()
    print('Animation is creating. Please wait.')
    for i in img:
        face_img.append(imageio.imread(img_dir+i))
    face_img = np.array(face_img)
    imageio.mimsave(img_dir+"out.mp4", face_img)

if __name__ == '__main__':
    main(sys.argv[1])
