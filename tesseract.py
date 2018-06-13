# Copyright 2017 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from PIL import Image, ImageOps
import sys
import pyocr
import pyocr.builders
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
import time
import argparse
import scatteract_logger

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]
scatteract_logger.get_logger().info("Will use tool '%s'" % (tool.get_name()))


def compute_skew(image):
    """
    Method which computes the skew of a label.
    Inputs:
    image (numpy array): Numpy array of the image label.
    Output:
    angle (float) Angle needed to unskew the label.
    contours_len: Number of independent object detected
    (if the image is clean, this is the number of digitsminus signs, and points)
    """

    image = ImageOps.invert(image)

    thresh = cv2.bitwise_not(cv2.adaptiveThreshold(np.array(image),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,3))
    thresh2 = cv2.threshold(np.array(image), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if cv2.__version__[0]=='2':
        contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    try:
        contours_len = len([j for j in hierarchy[0] if j[-1]==-1])
    except TypeError:
        contours_len = 0


    coords = np.column_stack(np.where(thresh > 0))
    try:
        rectangle = cv2.minAreaRect(coords)
    except cv2.error:
        rectangle = ((0,0),(0,0),0)
    angle = rectangle[-1]

    if angle <= -45:
        angle = -(90 + angle)
        height = rectangle[1][1]
        width = rectangle[1][0]
    else:
        angle = -angle
        height = rectangle[1][0]
        width = rectangle[1][1]


    if width>0 and angle!=0.0:
        if height/width>1.0 and contours_len >= 2:
            if angle<0:
                angle = +90+angle
            else:
                angle = angle - 90
        elif width/height>1.0 and contours_len < 2:
            if angle<0:
                angle = +90+angle
            else:
                angle = angle - 90

    return angle, contours_len


def deskew(image):
    """
    Method to unskew a image.
    Inputs:
    image (numpy array): Numpy array of the image label.
    Output:
    image (numpy array): Numpy array of the image label unskewed.
    contours_len: Number of independent object detected
    """

    angle, contours_len = compute_skew(image)

    image = Image.fromarray(ndimage.rotate(image, angle, mode='constant', cval=(np.median(image)+np.max(image))/2))

    return image, contours_len


def crop(image):
    """
    Method to crop out the uncessary white parts of the image.
    Inputs:
    image (numpy array): Numpy array of the image label.
    Outputs:
    image (numpy array): Numpy array of the image label, cropped.
    """

    image = ImageOps.invert(image)
    imageBox = image.getbbox()

    image = image.crop(imageBox)

    return ImageOps.invert(image)


def extract_number(digit_string):
    """
    Method used to clean up the output from tesseract.
    Inputs:
    digit_string (string): String output from tesseract.
    Outputs:
    pred (float) : If the input was a number, this is the float version of this number,
    else the output is None.
    """

    try:
        pred = float(digit_string.replace(" ", ""))
    except ValueError:
        scatteract_logger.get_logger().error("Output from tesseract is not a float " + digit_string)
        pred = None

    return pred


def get_label(image, size = 130):
    """
    Main method, used to preprocess the image and extract the label values using tesseract.
    Inputs:
    image (numpy array): Numpy array of the image label.
    size: (int) We resize the image such that its height is of length "size".
    Outputs
    pred (float): Value of the label detected via ocr.
    """

    image = image.convert('L')
    image = image.resize((size,size*np.shape(image)[0]/np.shape(image)[1]), Image.ANTIALIAS)
    image, contours_len =  deskew(image)
    image = crop(image)
    image = image.resize((size,size*np.shape(image)[0]/np.shape(image)[1]), Image.ANTIALIAS)

    digits = tool.image_to_string(
        image,
        lang="eng+osd",
        builder=pyocr.tesseract.DigitBuilder(tesseract_layout=6)
        )

    if contours_len==2 and len(digits)==1:  #Hacky, but helps tesseract on negative single digits
        digits = '-' + digits

    pred = extract_number(digits)

    return pred



if __name__ == "__main__":

    """
    Example of command line usage:

    python tesseract.py --tsv_truth data/plot_test/label_image_values.tsv --image_dir data/plot_test/label_images
    """

    mylogger = scatteract_logger.get_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--tsv_truth', help='Directory of the tsv which contains the ground truth (string)', required=True)
    parser.add_argument('--image_dir', help='Image directory (string)', required=True)
    args = vars(parser.parse_args())


    df = pd.read_csv(args['tsv_truth'], sep="\t", names=['image','label'], header = None)

    true = 0
    index = 0
    start = time.time()


    for j in range(0,len(df)):
        image = Image.open('{}/{}'.format(args['image_dir'],df.ix[j,'image']))
        pred = get_label(image)
        correct = pred==float(df.ix[j,'label'])

        if (j+1) % 20 == 0 or j==0:
            mylogger.debug(df.ix[j,'image'] + " found: {0} Real: {1} Correct?: {2}".format(pred, float(df.ix[j,'label']), correct))

        if correct:
            true+=1
        index+=1

    mylogger.info("Accuracy {0}, Number of images seen: {1}".format(float(true)/index, index))
    mylogger.info("Speed: {} image/sec".format(float(index)/(time.time()-start)))
