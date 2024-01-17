import torch
import numpy as np
import pandas as pd
from lib.config import *
from fastdist import fastdist

import numpy as np
from tqdm import tqdm
from scipy.linalg import inv


SAMPLE = pd.read_parquet(POSE_DATA_PATH+'/1019715464.parquet')
SAMPLE = SAMPLE.reset_index()
SAMPLE = SAMPLE[SAMPLE.columns[~SAMPLE.columns.str.contains('face')]]
SAMPLE.replace(np.NaN, 0, inplace=True)

x_cols = SAMPLE.columns[SAMPLE.columns.str.contains('^x_', regex=True)]
y_cols = SAMPLE.columns[SAMPLE.columns.str.contains('^y_', regex=True)]
z_cols = SAMPLE.columns[SAMPLE.columns.str.contains('^z_', regex=True)]

del SAMPLE

def getEuc(x, y): return np.linalg.norm(x - y)

def getCos(x, y): return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def extractPoseGraphs(
    paths: list,
):
    
    VIDEOS = []
    
    for doc_path, seq_id in tqdm(paths):
        
        POSE_DF = pd.read_parquet(POSE_DATA_PATH+'/'+doc_path)
        POSE_DF = POSE_DF.reset_index()
        POSE_DF = POSE_DF[POSE_DF.columns[~POSE_DF.columns.str.contains('face')]]
        POSE_DF.replace(np.NaN, 0, inplace=True)

        POSE_DF = POSE_DF[POSE_DF.sequence_id == seq_id]
        
        timestamps = POSE_DF.frame.max()

        VIDEO = []
        for timestamp in tqdm(range(NUM_CHANNELS)):
            try:
                merge_pose = POSE_DF.iloc[timestamp]
            except:
                merge_pose = POSE_DF.iloc[0]
                merge_pose = 0
            A = getPoseGraph(merge_pose, distF=DISTANCE, dim=NUM_CHANNELS)
            VIDEO.append(A)

        video_array = np.array(VIDEO)
        video_array = torch.tensor(video_array, dtype=torch.double)
                
        VIDEOS.append(video_array)          

    return VIDEOS


def getPoseGraph(
    merge_pose, 
    distF='cos', 
    dim=2
):

    dist, dist_id = (fastdist.euclidean, 'euclidean') if distF == 'euc' else (fastdist.cosine, 'cosine')
    
    if dim == 1:
        try:
            a = np.array([merge_pose[joint][:2]  for joint in list(merge_pose.columns[2:])])
            A = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)
            A = A.flatten()
        except:
            A = np.zeros((len(x_cols), len(x_cols)))

    elif dim == 2:
        try:
            a = np.array(merge_pose[x_cols].iloc[0].to_list())
            A_x = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)

            a = np.array(merge_pose[y_cols].iloc[0].to_list())
            A_y = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)

            A = np.stack([A_x, A_y], axis=2)
        except:
            A = np.zeros((len(x_cols), len(x_cols), 2))

    elif dim == 3:
        try:
            a = np.array(merge_pose[x_cols].iloc[0].to_list())
            A_x = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)

            a = np.array(merge_pose[y_cols].iloc[0].to_list())
            A_y = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)

            a = np.array(merge_pose[z_cols].iloc[0].to_list())
            A_z = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)

            A = np.stack([A_x, A_y, A_z], axis=2)
        except:
            A = np.zeros((len(x_cols), len(x_cols), 3))
            
            
    return A