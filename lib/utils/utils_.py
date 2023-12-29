import math
from transformers import VivitModel, VisionEncoderDecoderModel, AutoTokenizer
from config import *

def getPath(x):
    """Extract the path of the pose data from BosphorusSign22k.csv

    Args:
        x (str): The row of the csv file

    Returns:
        str: The path of the pose data
    """
    class_id = ('0' * (3-math.floor(math.log10(x[0])))) + str(x[0])
    repeat_id = ('0' * (2-math.floor(math.log10(x[2])))) + str(x[2])
    return f"{POSE_DATA_PATH}/{class_id}/{x[1]}_{repeat_id}.pickle"
