# computer-vision-Revise
Summarized form
Techniques
Thresholding
  used in image processing
  simplify visual data for further anslysis
In image processsing we need to pre-process the image data and get imp details techniqui is imp to seperate foreground image and background image
Steps:
in starting our image is colorful and it will contain 3 values from 0-255 for every pixels
then convert iamge in grey scale 
So for a gray-scale image, we only need 1 value for every pixel ranging from 0-255.[by converting we reduced the data but still have the information] 
The next step in thresholding is to define a threshold value that will filter out the information which we don’t want.
    pixel values < threshold = zero   pixel values greater than the threshold value will become 1.
    As you can see, now our image data only consist of values 0 and 1.
    
    
    The function used for the threshold is given below:

  cv2.threshold(img , 125, 255, cv2.THRESH_BINARY)
  The first parameter is the image data, the second one is the threshold value, the third is the maximum value (generally 255) and the fourth one is the threshold technique.
  techniques of threshold
  #Applying threshold to the gray image
ret, thresh1 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_TOZERO_INV)
#Displaying resultant images
cv2.imshow('Original Image', img)
cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)
  
  
Blurring and smoothing images

image may contains lots of noise .here are few techniques through which we can reduce the amount of noise by blurring them.
 The blurring technique is used as a preprocessing step in various other algorithms. With blurring, we can hide the details when necessary. For example – the police use the blurring technique to hide the face of the criminal. Later on, when learning about edge detection in the image, we will see how blurring the image improves our edge detection experience.

There are different computer vision algorithms available to us which will blur the image using a kernel size. There is no right kernel size, according to our need, we have to try the trial and error method to see what works better in our case. 

There are four different types of functions available for blurring:

cv2.blur() – This function takes average of all the pixels surrounding the filter. It is a simple and fast blurring technique.
cv2.GaussianBlur() – Gaussian function is used as a filter to remove noise and reduce detail. It is used in graphics software and also as a preprocessing step in machine learning and deep learning models.
cv2.medianBlur() – This function uses median of the neighbouring pixels. Widely used in digital image processing as under certain conditions, it can preserve some edges while removing noise.
cv2.bilateralFilter() – In this method, sharp edges are preserved while the weak ones are discarded.

