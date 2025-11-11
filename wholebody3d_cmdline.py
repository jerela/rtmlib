import time

import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

from rtmlib import PoseTracker, Wholebody3d, draw_skeleton
from rtmlib.visualization.visualizer_3d import Visualizer3D
import os
from anytree import RenderTree
from rtmlib.visualization.skeleton.coco133 import coco133


def apply_height_rebase(keypoints, disable_rebase=False):
    """Apply height rebasing to ground the pose; This is same as mmpose's post-processing function.
    
    Args:
        keypoints: 3D keypoints array
        disable_rebase: Whether to skip rebasing
        
    Returns:
        Height-rebased keypoints with lowest point at height 0
    """
    if not disable_rebase and keypoints is not None and len(keypoints) > 0:
        keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)
    return keypoints


def make_trc_with_trc_data(trc_data, trc_path, fps=30):
    '''
    Write a TRC file from a DataFrame of time and coordinates

    INPUTS:
    - trc_data: pd.DataFrame. The time and coordinates of the keypoints. 
                    The column names must be 'time', 'kpt1', 'kpt1', 'kpt1', 'kpt2', 'kpt2', 'kpt2', ...
    - trc_path: Path. The path to the TRC file to save
    - fps: float. The framerate of the video

    OUTPUT:
    - None
    '''

    DataRate = CameraRate = OrigDataRate = fps
    NumFrames = len(trc_data)
    NumMarkers = (len(trc_data.columns)-1)//3
    keypoint_names = trc_data.columns[1::3]
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + str(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, 0, NumFrames])), 
            # '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'simcc or px', OrigDataRate, 0, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoint_names) + '\t\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoint_names))])]

    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        trc_data.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')


def _save_trc_if_any(trc_rows_dict, trc_columns, fps_val, video, saved_trc):
    """Save collected TRC rows to file if any; sets saved_trc to True after saving."""
    if saved_trc:
        return
    
    if len(trc_rows_dict) > 0 and trc_columns is not None:
        # Save TRC for each person
        for person_id, trc_rows in trc_rows_dict.items():
            if len(trc_rows) > 0:
                df_trc = pd.DataFrame(trc_rows, columns=trc_columns)
                if video:
                    base_path = os.path.splitext(video)[0]
                    trc_path_local = f'{base_path}_person{person_id}.trc'
                else:
                    trc_path_local = f'output_person{person_id}.trc'
                make_trc_with_trc_data(df_trc, trc_path_local, fps=fps_val)
                print(f'TRC saved for person {person_id} to: {trc_path_local}')
        saved_trc = True
    else:
        print('No TRC data collected.')

# for cmd line args
import sys


def main():

    print(f'args: {len(sys.argv)}')

    path_input = sys.argv[1]

    device = 'cuda'
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino

    # choose input source: if `video` is set to a filepath, use it; otherwise use webcam (0)
    video = path_input
    if video:
        print('Video detected')
        cap = cv2.VideoCapture(video)
    else:
        print('No video detected')
        cap = cv2.VideoCapture(0)

    # prepare TRC logging
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if not fps_val or np.isnan(fps_val) or fps_val <= 0:
        fps_val = 30.0

    # Get total frame count for progress bar
    if video:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames = None  # Unknown for webcam

    trc_rows_dict = {}  # person_id -> list of rows
    trc_columns = None
    num_joints = None
    saved_trc = False

    wholebody3d = PoseTracker(
        Wholebody3d,
        det_frequency=1,
        tracking=False,  # Enable tracking for multi-person
        backend=backend,
        device=device)

    # try to extract keypoint names from the pose model
    keypoints_names = None
    keypoints_ids = None
    pose_model = getattr(wholebody3d, 'pose_model', None)
    print(f"pose model: {pose_model}")
    if pose_model is not None:
        try:
            keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if getattr(node, 'id', None) is not None]
            keypoints_names = [node.name for _, _, node in RenderTree(pose_model) if getattr(node, 'id', None) is not None]
        except Exception:
            keypoints_names = None
            keypoints_ids = None

    frame_idx = 0

    # Single person 3D visualizer (person 0 only)
    visualizer_3d = Visualizer3D(person=0)

    # Initialize progress bar
    if total_frames:
        pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    else:
        pbar = tqdm(desc="Processing frames", unit="frames")

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break
        
        # Update progress bar
        pbar.update(1)
        s = time.time()

        keypoints, scores, keypoints_simcc, keypoints_2d, track_ids = wholebody3d(frame)

        # Apply coordinate transformations to match MMPose format
        # Use keypoints_simcc to maintain original scale (like MMPose)
        if keypoints_simcc is not None:
            keypoints = keypoints_simcc.copy()
            keypoints = apply_height_rebase(keypoints, disable_rebase=False)

        # collect 3D keypoints for TRC
        kp_arr = np.array(keypoints)
        # initialize columns once we know number of joints
        if trc_columns is None and kp_arr is not None and len(kp_arr) > 0:
            # if RenderTree didn't give names, try coco133 mapping as fallback
            if keypoints_names is None:
                try:
                    max_id = max([v['id'] for v in coco133['keypoint_info'].values()])
                    coco_names = [None] * (max_id + 1)
                    for v in coco133['keypoint_info'].values():
                        coco_names[v['id']] = v['name']
                    # only accept coco names if joint count matches
                    if len(coco_names) == kp_arr.shape[1]:
                        keypoints_names = coco_names
                except Exception:
                    keypoints_names = None
            # prefer model-provided keypoint names when available
            if keypoints_names is not None and len(keypoints_names) == kp_arr.shape[1]:
                num_joints = len(keypoints_names)
                trc_columns = ['time'] + [name for name in keypoints_names for _ in range(3)]
            else:
                num_joints = kp_arr.shape[1]
                # fallback numeric names if model names unavailable
                trc_columns = ['time'] + [f'kpt{i+1}' for i in range(num_joints) for _ in range(3)]

        time_sec = frame_idx / float(fps_val)
        
        # Process each detected person
        if kp_arr is not None and len(kp_arr) > 0 and track_ids is not None:
            for person_idx in range(len(kp_arr)):
                if person_idx >= len(track_ids):
                    break
                    
                person_id = track_ids[person_idx]  # Use actual track ID
                
                # Initialize TRC rows for this person if not exists
                if person_id not in trc_rows_dict:
                    trc_rows_dict[person_id] = []
                
                # Create row for this person
                row = [time_sec]
                if num_joints is not None:
                    person = kp_arr[person_idx]
                    # person expected shape (J, C)
                    for j in range(num_joints):
                        if person.shape[1] >= 3:
                            x, y, z = person[j][:3]
                            row.extend([float(x), float(y), float(z)])
                        else:
                            row.extend([np.nan, np.nan, np.nan])
                    trc_rows_dict[person_id].append(row)
        else:
            # No detection in this frame -> fill with NaNs for existing persons
            for person_id in trc_rows_dict.keys():
                row = [time_sec]
                if num_joints is not None:
                    for _ in range(num_joints):
                        row.extend([np.nan, np.nan, np.nan])
                    trc_rows_dict[person_id].append(row)

        img_show = frame.copy()

        img_show = draw_skeleton(img_show,
                                 keypoints_2d,
                                 scores,
                                 kpt_thr=0.3,
                                 line_width=3)

        cv2.imshow('img', img_show)

        # Update 3D visualizer with person 0 only
        if keypoints is not None and len(keypoints) > 0 and track_ids is not None:
            # Find person with track_id = 0
            person0_idx = None
            for idx, track_id in enumerate(track_ids):
                if track_id == 0:
                    person0_idx = idx
                    break
            
            if person0_idx is not None:
                # Extract person 0 data
                person0_keypoints = keypoints[person0_idx:person0_idx+1]  # Keep as (1, J, 3)
                person0_scores = scores[person0_idx:person0_idx+1] if scores is not None else None
                visualizer_3d.update(person0_keypoints, person0_scores, kpt_thr=0.3)
            else:
                # Person 0 not detected, show empty visualization
                visualizer_3d.update(None, None, kpt_thr=0.3)
        else:
            # No detections
            visualizer_3d.update(None, None, kpt_thr=0.3)

        # Only support quitting with 'q'
        # First press of 'q' will close 3D visualization
        # Second press of 'q' will close 2D visualization (slightly delay) -> finish inference and save results as .trc file in the same directory as input video/image
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            # save collected data so far and then quit
            _save_trc_if_any()
            break

    # cleanup
    pbar.close()
    visualizer_3d.close()
    cap.release()
    cv2.destroyAllWindows()

    # write TRC if we collected data
    if not saved_trc:
        _save_trc_if_any(trc_rows_dict, trc_columns, fps_val, video, saved_trc)
    
if __name__ == '__main__':
    main()

