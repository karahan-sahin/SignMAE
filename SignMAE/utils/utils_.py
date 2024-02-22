import math
import numpy as np
from SignMAE.config import *

def process_skeleton(input_raw, nodes):
    active_frame_indices = get_active_frames(input_raw)

    input_raw = {
        **input_raw["pose"],
        **input_raw["face"],
        **input_raw["hand_left"],
        **input_raw["hand_right"],
    }
    
    input = np.array([input_raw[jn] for jn in nodes]).transpose((1, 0, 2))
    active_frame_indices = (
        active_frame_indices
        if active_frame_indices.size > 10
        else np.arange(0, len(input))
    )

    input = input[active_frame_indices, ...]
    return input, active_frame_indices

def get_active_frames(input_raw) -> np.ndarray:
    threshold = (
        (
            (
                (
                    input_raw["pose"]["left_hip"][:, 1]
                    + input_raw["pose"]["right_hip"][:, 1]
                )
                / 2
            )
            * 7
        )
        + input_raw["pose"]["nose"][:, 1]
    ) / 10

    active_frames = (
        np.minimum(
            input_raw["hand_left"]["left_lunate_bone"][:, 1],
            input_raw["hand_right"]["right_lunate_bone"][:, 1],
        )
        < threshold
    )

    active_frame_indices = np.argwhere(active_frames).squeeze()
    return active_frame_indices