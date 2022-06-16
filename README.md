# image-processing1
flower colur<br>
import cv2<br>
img=cv2.imread('download.jpg',1)<br>
cv2.imshow('flower1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![p](https://user-images.githubusercontent.com/98145915/174048358-3e71801e-8c8a-4321-b19c-7a9d312001a6.png)

matplotlib<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('download.jpg')<br>
plt.imshow(img)<br>
![pr](https://user-images.githubusercontent.com/98145915/174048578-8c840dce-ce42-4bb0-8f1b-7d8df3382da4.png)<br>
v
 image rotate<br>
 import cv2<br><br>
from PIL import Image<br>
image=Image.open("download.jpg")<br>
img=image.rotate(90)<br><br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![ro](https://user-images.githubusercontent.com/98145915/174048841-ed88c70c-f9d8-4203-aa47-9d43b7a1bf50.png)<br>

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
