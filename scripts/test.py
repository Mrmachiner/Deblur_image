
import cv2
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # test_command()
    img = cv2.imread("/home/minhhoang/Desktop/Deblur_image/out_92/000002.png")
    # cv2.imshow("abcd",img)
    imgplot = plt.imshow(img)
    plt.show()

    # cv2.waitKey(0)

