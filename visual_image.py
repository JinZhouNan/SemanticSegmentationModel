import cv2
import numpy as np
def visual_label (image, im_file):
    # imagefile="/home/zx1/project/Deeplab-v2--ResNet-101--Tensorflow-master/outimage/0.jpg"
    # image=cv2.imread(imagefile)
    # print image.shape
    image=np.array(image,np.int64)
    # print   image.shape
    im_h, im_w, _ = image.shape
    label = [[255, 255, 255],
             [255, 0, 0], [0, 0, 255], [0, 255, 0],
             [255, 255, 0], [255, 0, 255], [0, 255, 255],
             [125, 125, 125],
             [125, 125, 0], [125, 0, 125], [0, 125, 125],
             [125, 0, 0], [0, 0, 125], [0, 125, 0],
             [125, 125, 255], [125, 255, 125], [255, 125, 125],
             [125, 255, 255], [255, 255, 125], [255, 125, 255],
             [0, 255, 125], [125, 0, 255], [0, 125, 255], [125, 255, 0], [255, 0, 125], [255, 125, 0],
             [50, 100, 200],
             [0, 0, 0]
             ]
    image_1 = np.zeros([im_h, im_w, 3])
    for i_h in range(im_h):
        for i_w in range(im_w):
            # if image[i_h, i_w, 0] == 1:
            #     image_1[i_h, i_w, :] = label[image[i_h, i_w, 0]]
            # elif image[i_h, i_w, 0] == 0:
            #     image_1[i_h, i_w, :] = label[image[i_h, i_w, 0]]
            # else:
            #     image_1[i_h, i_w, :] = label[6]
            image_1[i_h, i_w, :] = label[image[i_h, i_w, 0]]
    cv2.imwrite(im_file, image_1)