# computer-vision-Revise

***Most important deep learning architectures like artificial neural networks (ANNs), convolutional networks (CNNs), recurrent networks (RNNs), and many more that are transferable to other domains like natural language processing (NLP) and voice user interface (VUI).***

Computer vision is a process by which we can understand the images and videos how they are stored and how we can manipulate and retrieve data from them. Computer Vision is the base or mostly used for Artificial Intelligence. Computer-Vision is playing a major role in self-driving cars, robotics as well as in photo correction apps. 
The CV pipeline is composed of 4 main steps: 1) image input, 2) image preprocessing, 3) feature extraction, and 4) ML model to interpret the image.

[CV Complete information Read then summarize here](https://livebook.manning.com/book/deep-learning-for-vision-systems/chapter-1/v-8/25)

https://xd.adobe.com/ideas/principles/emerging-technology/what-is-computer-vision-how-does-it-work/

https://data-flair.training/blogs/computer-vision-techniques/

https://www.javatpoint.com/keras

READ books on desktop

## Summarized form
Techniques:
**Thresholding**
  - used in image processing
  - simplify visual data for further anslysis

In image processsing we need to pre-process the image data and get imp details techniqui is imp to seperate foreground image and background image

Steps:

1.In starting our image is colorful and it will contain 3 values from 0-255 for every pixels

2.then convert iamge in grey scale 

3.So for a gray-scale image, we only need 1 value for every pixel ranging from 0-255.[by converting we reduced the data but still have the information] 

4.The next step in thresholding is to define a threshold value that will filter out the information which we don’t want.
    - pixel values < threshold = zero   pixel values greater than the threshold value will become 1.
    - As you can see, now our image data only consist of values 0 and 1.
    
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
  
  
**Blurring and smoothing images**

Image may contains lots of noise .here are few techniques through which we can reduce the amount of noise by blurring them.
The blurring technique is used as a preprocessing step in various other algorithms. With blurring, we can hide the details when necessary. For example – the police use the blurring technique to hide the face of the criminal. Later on, when learning about edge detection in the image, we will see how blurring the image improves our edge detection experience.

There are different computer vision algorithms available to us which will blur the image using a kernel size. There is no right kernel size, according to our need, we have to try the trial and error method to see what works better in our case. 

There are four different types of functions available for blurring:

- cv2.blur() – This function takes average of all the pixels surrounding the filter. It is a simple and fast blurring technique.
- cv2.GaussianBlur() – Gaussian function is used as a filter to remove noise and reduce detail. It is used in graphics software and also as a preprocessing step in machine learning and deep learning models.
- cv2.medianBlur() – This function uses median of the neighbouring pixels. Widely used in digital image processing as under certain conditions, it can preserve some edges while removing noise.
- cv2.bilateralFilter() – In this method, sharp edges are preserved while the weak ones are discarded.


# Some Question:

💭 How does Computer Vision Work?
Computer vision is a technique that extracts information from visual data, such as images and videos. Although computer vision works similarly to human eyes with brain work, this is probably one of the biggest open questions for IT professionals: How does the human brain operate and solve visual object recognition?

Computer Vision
On a certain level, computer vision is all about pattern recognition which includes the training process of machine systems for understanding the visual data such as images and videos, etc.

Firstly, a vast amount of visual labeled data is provided to machines to train it. This labeled data enables the machine to analyze different patterns in all the data points and can relate to those labels. E.g., suppose we provide visual data of millions of dog images. In that case, the computer learns from this data, analyzes each photo, shape, the distance between each shape, color, etc., and hence identifies patterns similar to dogs and generates a model. As a result, this computer vision model can now accurately detect whether the image contains a dog or not for each input image.

### Task Associated with Computer Vision
Although computer vision has been utilized in so many fields, there are a few common tasks for computer vision systems. These tasks are given below:

***Computer Vision***

Object classification: Object classification is a computer vision technique/task used to classify an image, such as whether an image contains a dog, a person's face, or a banana. It analyzes the visual content (videos & images) and classifies the object into the defined category. It means that we can accurately predict the class of an object present in an image with image classification.

Object Identification/detection: Object identification or detection uses image classification to identify and locate the objects in an image or video. With such detection and identification technique, the system can count objects in a given image or scene and determine their accurate location and labeling. For example, in a given image, one dog, one cat, and one duck can be easily detected and classified using the object detection technique.

Object Verification: The system processes videos, finds the objects based on search criteria, and tracks their movement.

Object Landmark Detection: The system defines the key points for the given object in the image data.

Image Segmentation: Image segmentation not only detects the classes in an image as image classification; instead, it classifies each pixel of an image to specify what objects it has. It tries to determine the role of each pixel in the image.

Object Recognition: In this, the system recognizes the object's location with respect to the image.

💭 What is the difference between computer vision and image processing?

Image processing algorithms transform images in many ways, such as sharpening, smoothing, filtering, enhancing, restoration, blurring and so on. Computer vision, on the other hand, focuses on making sense of what the machines see.

💭 What is image processing techniques in deep learning?

IDP leverages a deep learning network known as CNN (Convolutional Neural Networks) to learn patterns that naturally occur in photos. IDP is then able to adapt as new data is processed, using Imagenet, one of the biggest databases of labeled images, which has been instrumental in advancing computer vision.

💭 Which technique is best for image processing?
 
The top 6 image processing techniques for machine learning:
                 1.Image Restoration
                 2.Linear Filtering.
                 3.Independent Component Analysis.
                 4.Pixelation.
                 5.Template Matching.
                 6.Image Generation Technique (GAN)


💭 How to learn computer Vision?

Although, computer vision requires all basic concepts of machine learning, deep learning, and artificial intelligence. But if you are eager to learn computer vision, then you must follow below things, which are as follows:

Build your foundation:
Before entering this field, you must have strong knowledge of advanced mathematical concepts such as Probability, statistics, linear algebra, calculus, etc.
The knowledge of programming languages like Python would be an extra advantage to getting started with this domain.
Digital Image Processing:
It would be best if you understood image editing tools and their functions, such as histogram equalization, median filtering, etc. Further, you should also know about compressing images and videos using JPEG and MPEG files. Once you know the basics of image processing and restoration, you can kick-start your journey into this domain.
Machine learning understanding
To enter this domain, you must deeply understand basic machine learning concepts such as CNN, neural networks, SVM, recurrent neural networks, generative adversarial neural networks, etc.
Basic computer vision: This is the step where you need to decrypt the mathematical models used in visual data formulation.
What is the difference between OpenCV and cv2?
cv2 (old interface in old OpenCV versions was named as cv ) is the name that OpenCV developers chose when they created the binding generators. This is kept as the import name to be consistent with different kind of tutorials around the internet.

💭 What is the difference between OpenCV and machine learning?

Image result for difference between Image processing and opencv
OpenCV is the open-source library for computer vision and image processing tasks in machine learning. OpenCV provides a huge suite of algorithms and aims at real-time computer vision. Keras, on the other hand, is a deep learning framework to enable fast experimentation with deep learning.

💭 How many types of images are there in image processing?

The Two Types of Digital Images: Vector and Raster.

💭 What are the 3 major components of image?

Some of the most important are your ability to capture the right light, the right composition, and the right moment—the three elements of a great image.

💭 Which algorithm is best for image processing?

CNN is a powerful algorithm for image processing. These algorithms are currently the best algorithms we have for the automated processing of images. Many companies use these algorithms to do things like identifying the objects in an image. Images contain data of RGB combination.

💭 What are four different types of image processing methods?

Common image processing include image enhancement, restoration, encoding, and compression.

💭 Why is deep learning good for image classification?

Image classification with deep learning most often involves convolutional neural networks, or CNNs. In CNNs, the nodes in the hidden layers don't always share their output with every node in the next layer (known as convolutional layers). Deep learning allows machines to identify and extract features from images.

💭 Why is deep learning used for computer vision?

Image result for why is deep learning computer vision image processing
Deep learning methods can achieve state-of-the-art results on challenging computer vision problems such as image classification, object detection, and face recognition.

💭 Why deep learning is used in image processing?

Deep learning uses neural networks to learn useful representations of features directly from data. For example, you can use a pretrained neural network to identify and remove artifacts like noise from images.

💭 Which is better OpenCV or TensorFlow?

OpenCV is a library of programming functions mainly aimed at real-time computer vision. That is to say, OpenCV is a software library for performing computer vision tasks, whereas TensorFlow is a deep learning library. OpenCV supports the deep learning frameworks such as TensorFlow , Torch /PyTorch and Caffe .

💭 What is image processing in Python?

Image processing allows us to transform and manipulate thousands of images at a time and extract useful insights from them. It has a wide range of applications in almost every field. Python is one of the widely used programming languages for this purpose.

💭 What is FFT and DFT?

A fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa.

💭 What are the 3 types images format?

The PNG, JPEG, and GIF formats are most often used to display images on the Internet. Some of these graphic formats are listed and briefly described below, separated into the two main families of graphics: raster and vector.

💭 What are examples of image processing?

Examples of image processing
                  💢          Rescaling Image (Digital Zoom)
                  💢          Correcting Illumination.
                  💢          Detecting Edges.
                  💢          Mathematical Morphology.
                  💢          Evaluation and Ranking of Segmentation Algorithms.
