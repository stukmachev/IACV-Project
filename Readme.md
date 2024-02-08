# Object detection in motion-blurred images
A project for Image Analysis and Computer Vision subject, the project delivery includes

## The developed code in Python with user instructions below
The alpha-matting algorithm can be found in the file `alpha.py`. In this file you can find a set of five constans defined in the first lines of the program. The following are the default values, however for each image these values can be modified to obtain better results.
```
# Constants
IMAGE_NAME = "pencilcase"               # change for the name of your image
IMAGE_EXTENSION = "jpg"                 # change for the extension of your image
THRESH_BINARY = cv2.THRESH_BINARY_INV   # if the background of your image is light don't change it. the background is dark change it to cv2.THRESH_BINARY
THRESH_VALUE_BROAD = 200                # Can be a number between 0-255, must be larger than THRESH_VALUE_STRICT
THRESH_VALUE_STRICT = 50                # Can be a number between 0-255, must be lower than THRESH_VALUE_BROAD
```
Those two last parameters must be changed checking that the broad_mask and strict_mask images result in something that it's not a plain white or plain black image, otherwise it won't work.

## Experimental files:
 - input images can be found in the folder called `source`
 - We enumerate some values for THRESH_BINARY, THRESH_VALUE_BROAD and THRESH_VALUE_STRICT for the images mostly used:
    - `banana.jpg` -> `THRESH_BINARY = cv2.THRESH_BINARY`, `THRESH_VALUE_BROAD=150` and `THRESH_VALUE_STRICT=50`
    - `pencilcase.jpg` -> `THRESH_BINARY = cv2.THRESH_BINARY_INV`, `THRESH_VALUE_BROAD=200` and `THRESH_VALUE_STRICT=50`
    - `orange.png` -> `THRESH_BINARY = cv2.THRESH_BINARY_INV`, `THRESH_VALUE_BROAD=200` and `THRESH_VALUE_STRICT=50`
 - intermediate results of your processing algorithm are generated when you run the files according to the name of your file named: image_name_result.png, for example "pencilcase_trimap.png"
 - final results of your processing algorithm can be generated as well in a pdf at the end of file run.

## A well-written self-sufficient pdf report:
It can be found in the `docs` folder and it includes
 - the problem formulation (with a short motivation of its relevance)
 - a short outline of the state of the art
 - the solution approach
 - the description of your implementation
 - the experimental activity and the analysis of experimental results
 - final conclusions, also including some problems left open

## Slides for an oral presentation: 
The slides can be found in the `docs` folder and the have the same structure as the report structure