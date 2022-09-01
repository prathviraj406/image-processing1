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
14.bitwise operation<br>
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
15.gaussian<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('img1.jpg')<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
median=cv2.medianBlur(img,5)<br>
cv2.imshow('median Blurring',median)<br>
cv2.waitKey(0)<br>
bilateral=cv2.bilateralFilter(img,9,75,75)<br>
cv2.imshow(' bilateral Blurring',bilateral)<br>
cv2.waitKey(0)
cv2.destroyAllwindows()<br>
![2](https://user-images.githubusercontent.com/98145915/176426734-73d83336-a82f-49c8-9c3a-afef0458a103.png)
![22](https://user-images.githubusercontent.com/98145915/178706531-ffb705a8-33db-49a0-8682-051746f0ce9a.png)
![11](https://user-images.githubusercontent.com/98145915/178706750-cff3bcb1-7830-4754-b30b-80cce852bcaa.png)
![33](https://user-images.githubusercontent.com/98145915/178706971-7568b5cd-e0d4-4dde-b512-2d24f6574c55.png)<br>
 16.IMAGE ENHANCEMENT<br>
 from PIL import Image<br><br>
from PIL import ImageEnhance<br><br>
image=Image.open('img.jpg')<br><br>
image.show()<br><br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>
![20](https://user-images.githubusercontent.com/98145915/178708258-18b7e7bc-2aea-4611-bb16-0f8f00296f8f.png)
![30](https://user-images.githubusercontent.com/98145915/178708560-d14f3aa5-4eae-4fbf-a171-5ba0abb55c87.png)
![45](https://user-images.githubusercontent.com/98145915/178709726-3b8d27ec-be1e-497d-b674-623482189780.png)
![48](https://user-images.githubusercontent.com/98145915/178709875-0afbf797-6b70-4981-8206-a5cc3f17f1d3.png)
![44](https://user-images.githubusercontent.com/98145915/178710001-a574302b-50a3-4b88-b131-b3ddaea3390d.png)<br>
16.MORPHOPOGICAL OPERATION<br>
 import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img = cv2.imread('img.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel = np.ones((5,5),np.uint8)<br>
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN ,kernel)<br>
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)<br>
erosion = cv2.erode(img,kernel,iterations = 1)<br>
dilation = cv2.dilate(img,kernel,iterations = 1)<br>
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
 ![85](https://user-images.githubusercontent.com/98145915/178711263-d2bbeefe-ed05-421e-8a76-9f51a6320ba3.png)
17.program to save grayscale img in a specified drive<br>
 import cv2<br>
OriginalImag=cv2.imread('moutain.jpg')<br>
GrayImg=cv2.imread('moutain.jpg',0)
isSaved=cv2.imwrite('E:/I.JPG',GrayImg)<br>
cv2.imshow('display Original image',OriginalImag)<br>
cv2.imshow('display Grayscale image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAll()<br>
if isSaved:<br>
    print('the image is successfully saved.')<br>
 ![7](https://user-images.githubusercontent.com/98145915/178713081-c4dbe28f-4bb0-4f10-a97d-4c9196ed7812.png)
![8](https://user-images.githubusercontent.com/98145915/178713604-a57a9860-621b-405a-880b-69a3aecf9911.png)<br>
<br>
19.programslicing the image with  background<br>
import cv2<br>
import numpy as np
from matplotlib import pyplot as plt<br>
image=cv2.imread('moutain.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
![8](https://user-images.githubusercontent.com/98145915/178714321-348742a5-11f4-4d73-bcdb-b18c1d51defd.png)
20.without background<br>
 import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('moutain.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with out background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
 ![25](https://user-images.githubusercontent.com/98145915/178714797-39aefb74-f375-4dd0-ace6-733d170fa762.png)<br>
21.histogram using cv2<br>
import cv2<br>
import numpy as np<br>
img  = cv2.imread('moutain.jpg',0)<br>
hist = cv2.calcHist([img],[0],None,[256],[0,256])<br>
plt.hist(img.ravel(),256,[0,256])<br>
plt.show()<br>
![11](https://user-images.githubusercontent.com/98145915/178960762-67102e64-cd4a-437f-b111-4d5a9626d2c7.png)<br>
22.histogram using matplotlib<br>
import cv2  <br>
from matplotlib import pyplot as plt  <br>
img = cv2.imread('moutain.jpg',0) <br>
histr = cv2.calcHist([img],[0],None,[256],[0,256]) <br>
plt.plot(histr) <br>
plt.show()<br>
![10](https://user-images.githubusercontent.com/98145915/178961053-15a595df-6c95-4c69-9f2d-6074f600f2ec.png)<br>
 23.histogram using skimage(1)<br>
 from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('moutain.jpg')<br>

_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>
 ![1](https://user-images.githubusercontent.com/98145915/178965127-4ef38361-350e-48d9-87fc-367fb6756950.png)

24.histogram using skimage<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = io.imread('moutain.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
image = io.imread('moutain.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>
 ![50](https://user-images.githubusercontent.com/98145915/178965058-eafa5034-a196-4d66-9225-8b568d6a36b5.png)
 25.image negative<br>
 %matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('pr1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic)<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145915/179966663-0f317ab5-50a7-4006-a420-c728b77f1503.png)<br>(original)
negative=255-pic<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145915/179967126-2b0c3eb2-f0a7-4db7-925f-ab909aba8e31.png)<br>
26.log transfromation<br>
%matplotlib inline

import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('pr1.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max_=np.max(gray)<br>

def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145915/179967417-a4b2b277-bd7e-4a9e-8d89-e58d67057510.png)<br>
27.gamma correction<br>
 import imageio<br>
import matplotlib.pyplot as plt<br>
#Gamma encoding<br>
pic=imageio.imread('pr1.jpg')<br>
gamma=2.2#Gamma <1 ~ Dark ; Gamma >1 ~ Bright<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
![image](https://user-images.githubusercontent.com/98145915/179969613-f55bfe0c-376f-486c-bf78-4b6fdc10947a.png)

28.sharpeness<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
my_image=Image.open('pr1.jpg')<br>
sharp=my_image.filter(ImageFilter .SHARPEN)<br>
sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/179968296-f248b8a5-8547-4236-8867-0f4fbbee0c54.png)<br>
29.flipping<br>
import matplotlib.pyplot as plt<br><br>
img=Image.open('pr1.jpg')<br>
plt.imshow(img)<br><br>
plt.show()<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/179968550-79384c62-a61f-4f8f-a214-c8f8e56ad415.png)<br>
cropping<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
im=Image.open('pr1.jpg')<br>
width,height=im.size<br>
im1=im.crop((280,100,800,600))<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/179968711-9b1bb4bd-a83a-48fb-bf0e-36980e2ed365.png)
29 #Converting matrix to image<br><br>
from PIL import Image<br><br>
import numpy as np<br><br>
w, h = 512, 512<br><br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:256, 0:256] = [0, 255, 0] # red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
img.show()<br>
![image](https://user-images.githubusercontent.com/98145915/186376226-b405278a-bfb8-4c46-97aa-5421bdb40e58.png)<br>
30.#Grayscale gradient<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
x = np.linspace(10,6, 100)<br>
image = np.tile(x, (100, 1)).T<br>
plt.imshow(image, cmap='gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/186377120-8ce827d0-d2b1-400e-840c-68872d2990e4.png)<br>
31#Fill circle with color gradient<br>
import numpy as np<br><br>
import matplotlib.pyplot as plt<br><br>

arr = np.zeros((256,256,3), dtype=np.uint8)<br><br>
imgsize = arr.shape[:2]<br><br>
innerColor = (255, 255, 255)<br><br>
outerColor = (0, 0, 0)<br><br>
for y in range(imgsize[1]):<br><br>
    for x in range(imgsize[0]):<br><br>
        #Find the distance to the center<br><br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br><br>

        #Make it on a scale from 0 to 1innerColor<br><br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br><br>
<br><br>
        #Calculate r, g, and b values<br><br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br><br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        # print r, g, b<br>
        arr[y, x] = (int(r), int(g), int(b))<br>

plt.imshow(arr, cmap='gray')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/186385754-8fe7f078-3fb9-4f36-9042-0f6076d9148c.png)<br>
32#RGB<br>
from PIL import Image<br>
import numpy as np<br>
w, h = 600, 600
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400, 300:400] = [130, 255, 0]<br>
data[400:500, 400:500] = [0, 255, 170]<br>
data[500:600, 500:600] = [180, 255, 0]<br>
#red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/186387153-10ad1d99-fbdc-4ce7-8ca0-61234a8a0702.png)<BR>
33.# Python3 program for printing<br>
# the rectangular pattern<br>
 
# Function to print the pattern<br>
def printPattern(n):<br>
 
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         
    # Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) >
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2)) ;<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2)) ;<br>
    # Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
# Driver Code<br>
n = 4;<br>
 
printPattern(n);<br>
![image](https://user-images.githubusercontent.com/98145915/186388730-737aa0ff-82ef-4974-bd1b-4fcd747a952b.png)<br>
34.#image to matrri<br>x<br>
import matplotlib.image as image<br>
img=image.imread('img.jpg')<br><br>
print('The Shape of the image is:',img.shape)<br>
print('The image as array is:')<br>
print(img)<br>
![image](https://user-images.githubusercontent.com/98145915/186389509-5b8a679a-73a8-4bd4-b241-7fe048396b0d.png)<br>
34.program to perfrom image manipulation and edge detection<br>
import cv2<br><br>

# Read the original image<br><br>
img = cv2.imread('pr1.jpg')<br><br>

# Display original image<br><br>
cv2.imshow('Original', img)<br><br>
cv2.waitKey(0)<br><br>

# Convert to graycsale<br><br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br><br>
<br><br>
# Blur the image for better edge detection<br><br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br><br>
 <br><br>
# Sobel Edge Detection<br><br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br><br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br><br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br><br>
<br><br>
# Display Sobel Edge Detection Images<br><br>
cv2.imshow('Sobel X', sobelx)<br><br>
cv2.waitKey(0)<br><br>
cv2.imshow('Sobel Y', sobely)<br><br>
cv2.waitKey(0)<br><br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br><br>
cv2.waitKey(0)<br><br>

# Canny Edge Detection<br><br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
<br>
# Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98145915/186392223-24d9f8bc-879a-43cf-8595-b87adfea5c53.png)<br>
![image](https://user-images.githubusercontent.com/98145915/186392361-816c1ddb-4999-4840-9464-57a34285e874.png)<br>
![image](https://user-images.githubusercontent.com/98145915/186392490-85f51e5a-ac43-4a36-8d9b-5e9f776b90db.png)<br>
![image](https://user-images.githubusercontent.com/98145915/186392602-3a359f3d-2888-4a09-a232-c87378ee0c25.png)<br>
![image](https://user-images.githubusercontent.com/98145915/186392735-de3a6223-3180-440d-bac1-f04b29432ad5.png)<br><br>
 ![image](https://user-images.githubusercontent.com/98145915/186397648-f6b47db5-dd38-471d-b795-f69269bb40f2.png)<br>
from PIL import Image<br><br>

import matplotlib.pyplot as plt<br><br>
 
  
# Create an image as input:<br><br>

input_image = Image.new(mode="RGB", size=(400, 400),<br><br>
 
                        color="blue")<br><br>

  
# save the image as "input.png"<br><br>
 
#(not mandatory)<br><br>

#input_image.save("input", format="png")<br><br>
 

# Extracting pixel map:<br><br>
 
pixel_map = input_image.load()<br><br>

  
# Extracting the width and height<br><br>

# of the image:<br><br>
 
width, height = input_image.size<br><br>

z = 100<br><br>
 
for i in range(width):<br><br>

    for j in range(height):<br>
 
        <br>
 
        # the following if part will create<br>

        # a square with color orange<br>
 
        if((i >= z and i <= width-z) and (j >= z and j <= height-z)):<br>
 
            <br>
 
            # RGB value of orange<br>
 
            pixel_map[i, j] = (255, 165, 255)<br>

  
        # the following else part will fill the<br>
 
        # rest part with color light salmon.<br>
 
        else:<br>
 
            
            # RGB value of light salmon.<br>
 
            pixel_map[i, j] = (255, 160, 0)<br>

  <br>
 
# The following loop will create a cross<br>
 
# of color blue.<br>

for i in range(width):<br>

    
    # RGB value of Blue.<br>

    pixel_map[i, i] = (0, 0, 255)<br>
 
    pixel_map[i, width-i-1] = (0, 0, 255)<br>

  <br>
 
# Saving the final output<br>
 
# as "output.png":<br>

#input_image.save("output", format="png")<br>

plt.imshow(input_image)<br>
 
plt.show()  
# use input_image.show() to see the image on the<br>
#min<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('pr1.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>

print(min_channels)<br>
![image](https://user-images.githubusercontent.com/98145915/186398911-42659051-64d2-4d07-9db6-6639ac9e0d30.png)<br>
#sd<br>
from PIL import Image,ImageStat
import matplotlib.pyplot as plt<br>
im=Image.open('pr1.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
 ![image](https://user-images.githubusercontent.com/98145915/186399430-03d5c2ac-1c16-452b-81ac-2a8e77447ac9.png)<br>
 #average<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread("pr1.jpg")<br>
plt.imshow(img)<br>
np.average(img)<br>
 ![image](https://user-images.githubusercontent.com/98145915/186399690-dc530c4e-ca20-4a29-bab7-683afb87eb2a.png)<br>
 #max<br>
import imageio<br>
import numpy as np<br> as plt
img=imageio.imread('pr1.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
<br>
print(max_channels)<br>
![image](https://user-images.githubusercontent.com/98145915/186399922-7d86c751-f106-4679-ab6a-e18de356ffff.png)<br>
37.from PIL import Image,ImageChops,ImageFilter<br><br>
from matplotlib import pyplot as plt<br><br>
x=Image.open('x.png' )<br><br>
o=Image.open('o.png' )<br><br>
print('size of the image: ',x.size, 'colour mode:',x.mode)<br><br>
print('size of the image:',o.size, 'colour mode:',o.mode)<br>
plt.subplot(121), plt.imshow(x)<br><br>
plt.axis('off')<br><br>
plt.subplot(122), plt.imshow(o)<br>
plt.axis('off')<br>
merged=ImageChops.multiply(x,o)<br>
add=ImageChops.add(x,o)<br>
greyscale=merged.convert('L')<br>
greyscale<br>
![image](https://user-images.githubusercontent.com/98145915/187874524-be98b7fc-57ef-4572-94dd-8e9d5a3cdb8f.png)<br>
37 a.image=merged<br>
print('image size:',image.size,'\ncolor mode:',image.mode,'\nimage width:',image.width,'| also represented by:',image.size[0],'\nimage height:',image.height,'| also represented by:',image.size[1])<br>
output<br>
image size: (256, 256) <br>

color mode: RGB <br>

image width: 256 | also represented by: 256 <br>

image height: 256 | also represented by: 256<br>
37 b.pixel=greyscale.load()<br>
for row in range(greyscale.size[0]):<br>
    for column in range(greyscale.size[1]):<br>
       if pixel[row,column] !=(255):<br>
          pixel[row,column]=(0)<br>
greyscale<br>
![image](https://user-images.githubusercontent.com/98145915/187875812-c865bea0-61ca-4eba-996e-ae8d653c81fb.png)<br>
37 c.invert=ImageChops.invert(greyscale)<br>
bg=Image.new('L',(256,256),color=(255))<br>
subt=ImageChops.subtract(bg,greyscale)<br>
rotate=subt.rotate(45)<br>
rotate<br>
![image](https://user-images.githubusercontent.com/98145915/187876029-687871e1-0dca-412f-92ab-07b986908fe1.png)<br>
37 d.blur=greyscale.filter(ImageFilter.GaussianBlur(radius=1))<br>
edge=blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
![image](https://user-images.githubusercontent.com/98145915/187876275-ee6de9b7-1f56-4436-800e-dbfbf0f9238a.png)<br>
37e.edge=edge.convert('RGB')<br>
bg_red=Image.new('RGB',(256,256),color=(255,0,0))<br>
filled_edge=ImageChops.darker(bg_red,edge)<br>
filled_edge<br>
![image](https://user-images.githubusercontent.com/98145915/187876572-223224e3-7a9e-4978-91f8-dceedfdd7243.png)<br>
 38.import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
img =cv2.imread('dimage_damaged.png')<br>
plt.imshow(img)<br>
plt.show()<br>
mask=cv2.imread('dimage_mask.png',0)<br>
plt.imshow(mask)<br>
plt.show()<br>
dst=cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)<br>
cv2.imwrite('dimage_inpainted.png',dst)<br>
plt.imshow(dst)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145915/187877347-96df77e8-1a5f-419a-957d-fef89f1881f1.png)<br>
 39.import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
plt.rcParams['figure.figsize']=(10,8)<br>
def show_image(image,title='Image',cmap_type='gray'):<br>
    plt.imshow(iamge,cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br><br>
    
def plot_comparison(img_original,img_filtered,img_title_filtered):<br><br>
    fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(10,8),sharex=True,sharey=True)<br><br>
    ax1.imshow(img_original,cmap=plt.cm.gray)<br><br>
    ax1.set_title('original')<br><br>
    ax1.axis('off')<br><br>
    ax2.imshow(img_filtered,cmap=plt.cm.gray)<br><br>
    ax2.set_title(img_title_filtered)<br><br>
    ax1.axis('off')<br><br>
from skimage.restoration import inpaint<br><br>
from skimage.transform import resize
from skimage import color<br><br>
image_with_logo=plt.imread('imlogo.png')<br><br>
#intitalize the mask<br><br>
mask=np.zeros(image_with_logo.shape[:-1])<br><br>
#set the pixels where the logo is to 1<br><br>
mask[210:272,360:425]=1<br><br>
#apply inpainting to remove the logo<br><br>
image_logo_removed=inpaint.inpaint_biharmonic(image_with_logo,mask,multichannel=True)<br><br>
#show the original and logo removed images<br><br>
plot_comparison(image_with_logo,image_logo_removed,'image with logo removed')<br><br><br>
 ![image](https://user-images.githubusercontent.com/98145915/187878457-89635124-14c7-4200-afdd-4defb3217dba.png)<br><br><br>
40.from skimage.util import random_noise<br>
fruit_image = plt.imread('fruitts.jpeg')<br>
#Add noise to the image<br>
noisy_image = random_noise (fruit_image)<br>
#Show th original and resulting image<br>
plot_comparison (fruit_image, noisy_image, 'Noisy image')<br>
 ![image](https://user-images.githubusercontent.com/98145915/187878685-f3ad26f4-5e55-4fb1-b9d5-5364a57572c0.png)<Br>
 41.from skimage.restoration import denoise_tv_chambolle<br>
 
noisy_image = plt.imread('noisy.jpg')<br>
 
# Apply total variation filter denoising<br>
 
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)<br>
 
#show the noisy and denoised image plot_comparison (noisy_image, denoised_image, 'Denoised Image')<br>
 
plot_comparison(noisy_image,denoised_image,'Denoised Image'<br>
 ![image](https://user-images.githubusercontent.com/98145915/187879357-61dcee82-26f0-46d6-9b2c-68cd6dcbaf5b.png)<br>
 41a.from skimage.restoration import denoise_bilateral<br>
landscape_image = plt.imread('noisy.jpg')<br>
# Apply bilateral filter denoising<br>
denoised_image = denoise_bilateral(landscape_image, multichannel=True)<br>
# Show original and resulting images<br>
plot_comparison (landscape_image, denoised_image, 'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/98145915/187879644-e8601e93-1f8e-4713-801d-fb6bc0ac1632.png)<br>
 42.from skimage.segmentation import slic <br>
from skimage.color import label2rgb<br>
face_image = plt.imread('face.jpg')<br>
 
#Obtain the segmentation with 400 regions <br>
segments = slic (face_image, n_segments-400)<br>
#Put segments on top of original image to compare <br>
segmented_image = label2rgb(segments, face_image, kind='avg')<br>
#Show the segmented image <br>
plot_comparison (face_image, segmented_image, 'Segmented image, 400 superpixels')<br><br><br>
![image](https://user-images.githubusercontent.com/97940277/187892986-02557290-123f-41fc-af1a-01b0b0328ed0.png)<BR>
43. #4)contours:<br>
#a)contouriinng shapes<br>
def show_image_contour(image,contours)<br>
    plt.figure()<br>
    for n,contour in enumerate(contours):<br>
        plt.plot(contour[:,1],contour[:,0],linewidth=3)<br>
    plt.imshow(image,interpolation='nearest',cmap='gray_r')<br>
    plt.title('contour')<br>
    plt.axis('off')<br>
 43.A.from skimage import measure,data<br>
#obtain the horse image<br>
horse_image=data.horse()<br>
#find the contour with a constant level of 0.8<br>
contours=measure.find_contours(horse_image,level=0.8)<br>
#shows the image with contours found<br>
show_image_contour(horse_image,contours)<br>
 ![image](https://user-images.githubusercontent.com/98145915/187896891-e527ebaa-d1de-4aab-9d4f-ae2c577b2015.png)<br>
 44.#b)find contours of an image that is not binary<br>
from skimage.io import imread <br>
from skimage.filters import threshold_otsu<br>

image_dices = imread('diceimg.png')<br><br>

# Make the image grayscale<br><br>
image_dices = color.rgb2gray(image_dices)<br><br>

#Obtain the optimal thresh value <br>
thresh = threshold_otsu(image_dices)<br>

# Apply thresholding<br>
binary=image_dices > thresh<br>

# Find contours at a constant value of 0.8<br>
contours = measure.find_contours (binary, level=0.8)<br>

# Show the image<br>
show_image_contour (image_dices, contours)<br>
![image](https://user-images.githubusercontent.com/98145915/187897490-09161639-3776-45cf-8112-94a2ce4313c1.png)<BR>
