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









 
 
