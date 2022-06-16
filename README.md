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
resize the original image<br>
import cv2<br>
img=cv2.imread('sunflowers.jpg')<br><br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('resized image',imgresize)<br>
print('resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>
![resize](https://user-images.githubusercontent.com/98145915/174047204-98560a67-dda4-446f-9164-d69fbccc2c59.png)


convert the original image to gray colour and resize<br>
import cv2<br>
img=cv2.imread('sunflowers.jpg')<br>
cv2.imshow("RGB",img)<br><br>
cv2.waitKey(0)<br>
img=cv2.imread('sunflowers.jpg',0)<br>
cv2.imshow("gray",img)<br>
cv2.waitKey(0)<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows<br>
![colour](https://user-images.githubusercontent.com/98145915/174048007-5635afdd-b8a8-4a1a-8284-2fdc17eab4ed.png)
