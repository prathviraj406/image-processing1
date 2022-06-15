# image-processing1
flower colur<br>
import cv2<br>
img=cv2.imread('download.jpg',1)<br>
cv2.imshow('flower1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
matplotlib<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('download.jpg')<br>
plt.imshow(img)<br>
 image rotate
 import cv2
from PIL import Image
image=Image.open("download.jpg")
img=image.rotate(90)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
