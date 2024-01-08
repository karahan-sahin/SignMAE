import math
from transformers import VivitModel, VisionEncoderDecoderModel, AutoTokenizer
from ..config import *

def getPath(x):
    """Extract the path of the pose data from BosphorusSign22k.csv

    Args:
        x (str): The row of the csv file

    Returns:
        str: The path of the pose data
    """

