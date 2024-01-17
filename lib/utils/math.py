import itertools
import threading
import torch
import pickle
import numpy as np
from lib.config import *
from ..utils.utils_ import *
from fastdist import fastdist
from tqdm import tqdm
import numpy as np
from scipy.linalg import inv

def mahalanobis_distance(x, y, S):
    """
    Calculate the Mahalanobis distance between two points with the covariance matrix derived from data.
    
    Parameters:
    x (array): Coordinates of the first point.
    y (array): Coordinates of the second point.
    data (array): Dataset to calculate the covariance matrix from.
    
    Returns:
    float: Mahalanobis distance.
    """
    # Calculating the Mahalanobis distance
    x_minus_y = np.array(x) - np.array(y)
    x_minus_y = x_minus_y.reshape(-1, 1)  # Reshape x_minus_y to a column vector
    distance = np.sqrt(np.dot(np.dot(x_minus_y.T, inv(S)), x_minus_y))
    return distance.item()  # Return the distance as a scalar    


def calculate_distance(comb, merge_pose, data):
    A, (joint_0, idx_0), (joint_1, idx_1), (timestamp, total_timestamps) = comb
    J0 = np.array([merge_pose[joint_0][timestamp][:2]])
    J1 = np.array([merge_pose[joint_1][timestamp][:2]])
    # Calculating the covariance matrix of the data
    data = np.concatenate([
            np.array([merge_pose[joint_0][t][:2] for t in range(total_timestamps)]), 
            np.array([merge_pose[joint_1][t][:2] for t in range(total_timestamps)])
        ], axis=0)
    
    S = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-5
    dist = mahalanobis_distance(J0, J1, S)
    A[idx_0, idx_1] = dist

def getEuc(x, y): return np.linalg.norm(x - y)

def getCos(x, y): return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def getPoseGraphBSign22k(merge_pose, 
                         timestamp, 
                         distF='cos', 
                         dim=1,
                         total_timestamps=100):
    
    if distF == 'cos':
        dist, dist_id = fastdist.cosine, 'cosine'
    if distF == 'euc':
        dist, dist_id = fastdist.euclidean, 'euclidean'
    # if distF == 'mah':
    #     dist, dist_id = mahalanobis_distance, 'mahalanobis'

    if dim == 1:
        try:
            # if distF == 'mah':
            #     A = np.zeros((len(merge_pose.keys()), len(merge_pose.keys())))
                
            #     threads = []    
            #     lists = list(enumerate(list(merge_pose.keys())))
            #     comb = list(itertools.combinations(lists, 2))
            #     for (idx_0, joint_0), (idx_1, joint_1) in comb:
                    
            #         def calculate_distance(idx_0, joint_0, idx_1, joint_1, timestamp, total_timestamps):
            #             J0 = np.array([merge_pose[joint_0][timestamp][:2]])
            #             J1 = np.array([merge_pose[joint_1][timestamp][:2]])
            #             # Calculating the covariance matrix of the data
            #             data = np.concatenate([
            #                     np.array([merge_pose[joint_0][t][:2] for t in range(total_timestamps)]), 
            #                     np.array([merge_pose[joint_1][t][:2] for t in range(total_timestamps)])
            #                 ], axis=0)
                        
            #             S = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-5
            #             dist = mahalanobis_distance(J0, J1, S)
                        
            #             A[idx_0, idx_1] = dist                
            #             A[idx_1, idx_0] = dist                
        
            #         thread = threading.Thread(target=calculate_distance, args=(idx_0, joint_0, idx_1, joint_1, timestamp, total_timestamps))
            #         threads.append(thread)
            #         thread.start()

            #     for thread in threads:
            #         thread.join()
            # else:
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
    
    VIDEOS = []

    for doc_path in paths:
        
        with open(doc_path, 'rb') as f:

            x = pickle.load(f)
            nodes = list(x['pose'].keys()) + list(x['hand_right'].keys()) + list(x['hand_left'].keys())
           
            _, activate_indices = process_skeleton(x, nodes=list(set(nodes)))

            POSE_DF = []
            merge_pose = {}
            merge_pose.update(x['hand_right'])
            merge_pose.update(x['hand_left'])
            merge_pose.update(x['pose'])
            
            total_timestamps = len(merge_pose[list(merge_pose.keys())[0]])
            VIDEO = []
            timestamp = 0
            # print(f'{doc_path=}: {activate_indices}')
            
            for timestamp in tqdm(list(np.arange(total_timestamps))):
                if timestamp in activate_indices and len(VIDEO) < NUM_FRAMES:
                    A = getPoseGraphBSign22k(merge_pose, timestamp, distF, dim, total_timestamps=total_timestamps)
                    VIDEO.append(A)
                    
            if len(VIDEO) < NUM_FRAMES:
                for _ in range(NUM_FRAMES - len(VIDEO)):
                    A = getPoseGraphBSign22k(merge_pose, -1, distF, dim, total_timestamps=total_timestamps)
                    VIDEO.append(A)            
                        
            video_array = np.array(VIDEO)
            video_array = torch.tensor(video_array, dtype=torch.double)
                
            VIDEOS.append(video_array)          

    return VIDEOS