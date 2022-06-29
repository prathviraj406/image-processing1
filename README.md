# image-processing1
1.flower colur<br>
import cv2<br>
img=cv2.imread('download.jpg',1)<br>
cv2.imshow('flower1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![p](https://user-images.githubusercontent.com/98145915/174048358-3e71801e-8c8a-4321-b19c-7a9d312001a6.png)

2.matplotlib<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('download.jpg')<br>
plt.imshow(img)<br>
![pr](https://user-images.githubusercontent.com/98145915/174048578-8c840dce-ce42-4bb0-8f1b-7d8df3382da4.png)<br>
v
 3.image rotate<br>
 import cv2<br><br>
from PIL import Image<br>
image=Image.open("download.jpg")<br>
img=image.rotate(90)<br><br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![ro](https://user-images.githubusercontent.com/98145915/174048841-ed88c70c-f9d8-4203-aa47-9d43b7a1bf50.png)<br>

4.resize the original image<br>
import cv2<br>
img=cv2.imread('sunflowers.jpg')<br><br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('resized image',imgresize)<br>
print('resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>
![resize](https://user-images.githubusercontent.com/98145915/174047204-98560a67-dda4-446f-9164-d69fbccc2c59.png)


5.convert the original image to gray colour and resize<br>
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
6.write program to convert image using colors?<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,2<br>55,0))<br>
img.show()<br>
![n](https://user-images.githubusercontent.com/98145915/174051330-4cfdbe50-35f9-405a-9a2b-81489d593100.png)
7..Devlop a program to convert color string to RGB color value?<br>
from PIL import ImageColor<br><br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
output:<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
8.Write a program to display the image attributes?<br>
from PIL import Image<br>
image=Image.open('tree.jpg')<br>
print("filename:",image.filename)<br>
print("format:",image.format)<br>
print("mode:",image.mode)<br>
print("size:",image.size)<br>
print("width:",image.width)<br>
print("height:",image.height)<br>
image.close()<br>

output:<br>
filename: tree.jpg<br>
format: JPEG<br>
mode: RGB<br>
size: (1334, 888)<br>
width: 1334<br>
height: 888<br>
9. import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('butterflypic.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.show()<br><br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.show()<br>
![1](https://user-images.githubusercontent.com/98145915/174055098-19556aae-e648-4135-9449-282dcc4c01d4.png)
![2](https://user-images.githubusercontent.com/98145915/174055317-075011d0-d8be-4d3a-80f0-4c722e7f14a8.png)
![3](https://user-images.githubusercontent.com/98145915/174055458-a8927fd8-cbb6-4866-b4e4-4bc97c2c164b.png)
![4](https://user-images.githubusercontent.com/98145915/174055552-c4c407d3-ccc7-4e89-98fe-d4b446977011.png)
10.image using url<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRc95jShZH3NnnLDjfy5flhVeirqgcVmlH09g&usqp=CAU'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
![1](https://user-images.githubusercontent.com/98145915/175005610-e8cd413e-c194-476b-abf7-2bd19b3a86ac.png)<br>
11.maths operation<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('img1.jpg')<br>
img2=cv2.imread('sunflowers.jpg')<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()<br>
cv2.imwrite('output.jpg',fimg4)<br>
![1](https://user-images.githubusercontent.com/98145915/175261348-314322c2-8310-429a-9575-953cb1b6f046.png)<br>
![21](https://user-images.githubusercontent.com/98145915/175261469-f352612f-c10c-48ca-b83b-5c2f4b128537.png)<br>
![22](https://user-images.githubusercontent.com/98145915/175261663-efcd32c7-a2cf-4918-b681-062dbfde4e1b.png)<br>
![24](https://user-images.githubusercontent.com/98145915/175261823-042d1c95-8976-40c7-8b7d-7d030599af34.png)<br>
11.MASK AND BLUR A IMAGE<BR>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('img.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![25](https://user-images.githubusercontent.com/98145915/175262443-ded933de-72f9-41ff-9caa-73a0acc12379.png)<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![25](https://user-images.githubusercontent.com/98145915/175262807-08cf40db-fcd4-4e03-9be6-0f8c1abb2eec.png)<br>
 import cv2<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![100](https://user-images.githubusercontent.com/98145915/175264612-420feff9-1536-45fa-bb99-e80a43591370.png)<br>
final_mask=mask + mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
![111](https://user-images.githubusercontent.com/98145915/175264968-0b859a0b-9ce2-4f3e-a8b8-ad05ac8b114d.png)<br>
blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![11](https://user-images.githubusercontent.com/98145915/175265253-8f7a6f4a-e5f9-4d9a-b145-85e81f0e2888.png)<br>
12.to change the image to differnt color space<br>
 import cv2 <br>
img=cv2.imread("avenger.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>
 ![gray](https://user-images.githubusercontent.com/98145915/175271470-9ce7a0da-96ec-45cb-a233-3986510f2322.png)<br>
![hsv](https://user-images.githubusercontent.com/98145915/175271528-766fbe00-328f-4a92-918a-e3239df02117.png)<br>
![lab](https://user-images.githubusercontent.com/98145915/175271570-06713541-1304-47e0-af91-e3470f2c6e1e.png)<br>
 ![yuv](https://user-images.githubusercontent.com/98145915/175271611-3cb0bb9d-e9f7-493e-98c6-0124da05db0a.png)<br>
 ![hls](https://user-images.githubusercontent.com/98145915/175271639-a0d44a68-ee07-413c-bcf8-c92232f13d76.png)<br>
code to see the size of image<br>
 img1.shape<br>
13.2D array<br>
 import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('imgYG.png')<br>
img.show()<br>
c.waitKey(0)<br>
![12](https://user-images.githubusercontent.com/98145915/175274193-441d45c5-02dd-4846-9b51-d9a9bbe4b664.png)<br>
bitwise operation<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
im1=cv2.imread('img.jpg')<br>
im2=cv2.imread('img.jpg')<br>
ax=plt.subplots(figsize=(15,10))
bitwiseAnd=cv2.bitwise_and(im1,im2)<br>
bitwiseOr=cv2.bitwise_or(im1,im2)<br>
bitwiseXor=cv2.bitwise_xor(im1,im2)<br>
bitwiseNot_img1=cv2.bitwise_not(im1)<br>
bitwiseNot_img2=cv2.bitwise_not(im2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>
![1](https://user-images.githubusercontent.com/98145915/176422736-2b2b7fd5-9325-4e7d-8596-b0e81089f6ee.png)


