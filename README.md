# Plate Recognition

The goal of this project is to make a simple script that is capable of recognizing colombian car plates in a somewhat controlled enviroment

## Setup
You can run quickly the proyect in just 4 steps:
1. clone the git-hub proyect
2. install the dependencies listed in requirements.txt
3. get the image of the plate you want to recognize. Here is an example I found online <a>https://i.pinimg.com/originals/b8/33/af/b833afdfa6d050543117a76aa95dbd9c.jpg</a>
4. set the variable path in the file integration/script.py or integration/quick_run.py to the path of your image (if you are going to use the example linked above it works better with quick_run.py)
5. run the python script from step 4 with python3 and done!


## How does the script work?
I will give a high level description of the task I implemented in order to make the script work.

### 1. Estimate a bounding-box of the plate. 
For the case of a colombian car, I used the colors yellow and black to detect the plate. In a more general case there are more sophisticated DL algorithms to make object detection, for example: <a>https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006</a>

### 2. Divide the plate in the different characters in it.
I used the colors of the plate and the unsupervised algorithm kmeans to separate the characters (that are basically objects of the same color).

### 3. Clasify every caracter of the plate.
I trained a simple neural network to classify the characters. Since the characters of the plate are standarized, the model can get an almost 100% acuracy, way better than a more general OCR.

