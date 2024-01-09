import torch
import pickle
import numpy as np
from ..config import *
from fastdist import fastdist

def getMahalonobisDistance(x, y, cov): return np.linalg.norm(x - y, ord=1)

def getEuc(x, y): return np.linalg.norm(x - y)

def getCos(x, y): return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def getPoseGraphBSign22k(merge_pose, 
                 timestamp, 
                 distF='cos', 
                 dim=1):
    
    dist, dist_id = (fastdist.euclidean, 'euclidean') if distF == 'euc' else (fastdist.cosine, 'cosine')
    
    if dim == 1:
        try:
            a = np.array([merge_pose[joint][timestamp][:2]  for joint in list(merge_pose.keys())])
            A = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)
        except:
            A = np.zeros((len(merge_pose.keys()), len(merge_pose.keys())))

    elif dim == 2:
        
        try:
            a = np.array([merge_pose[joint][timestamp][:1]  for joint in list(merge_pose.keys())])
            A_x = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)
            
            a = np.array([merge_pose[joint][timestamp][1:2]  for joint in list(merge_pose.keys())])
            A_y = fastdist.matrix_pairwise_distance(a, dist, dist_id, return_matrix=True)
                
            A = np.stack([A_x, A_y], axis=2)
        except:
            A = np.zeros((len(merge_pose.keys()), len(merge_pose.keys()), 2))
        
    return A


def extractPoseGraphs(
    paths: list,
    dataset_path: str = 'datasets/train',
    distF: str = 'cos',
    dim: int = 1,
):
    
    # TODO: Add Hand Up-Down Crop

    VIDEOS = []
    
    for doc_path in paths:
        
        try:
            with open(doc_path, 'rb') as f:

                x = pickle.load(f)

                POSE_DF = []
                merge_pose = {}
                merge_pose.update(x['hand_right'])
                merge_pose.update(x['hand_left'])
                merge_pose.update(x['pose'])

                timestamps = list(merge_pose.values())[0].shape[0]

                VIDEO = []
                for timestamp in range(NUM_FRAMES):
                    A = getPoseGraphBSign22k(merge_pose, timestamp, distF, dim)
                    VIDEO.append(A)
                
                video_array = np.array(VIDEO)
                video_array = torch.tensor(video_array, dtype=torch.double)
                    
                VIDEOS.append(video_array)
        except: print(f' ** {doc_path=} cannot be readed ** ')

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