'''
Test 'total curvature' using real data
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from total_curvature import total_curvature_2d, total_curvature_3d
from resample import resample_splines
from derivative import calculate_velocity
from filter import butter_lowpass_filter
from movement_onset_detection import onset_detection


# VR-S1 Data
p_ = 1
participant_ = r"\P" + str(p_).zfill(2)
path_ = os.path.dirname(os.getcwd()) + r"\VR-S1" + participant_ + r"\S001"

path_results = path_ + r"\trial_results.csv"
results = pd.read_csv(path_results, usecols=['start_time', 'initial_time'])
start_time = results['start_time'].to_numpy()
initiation_time = results['initial_time'].to_numpy()
t_threshold = initiation_time - start_time

for trial_number in range(17, 284 + 1, 1):

    path_trial = path_ + r"\trackers" + r"\controllertracker_movement_T" + str(trial_number).zfill(3) + ".csv"

    # Load Raw Data
    raw_data = pd.read_csv(path_trial, usecols=['time', 'pos_x', 'pos_y', 'pos_z'])

    # Adjust to Zero
    t_raw = raw_data['time'].to_numpy() - start_time[trial_number - 1]
    x_raw = raw_data['pos_x'].to_numpy()
    y_raw = raw_data['pos_y'].to_numpy()
    z_raw = raw_data['pos_z'].to_numpy()

    # Resampling
    resampled_data = resample_splines(t_raw, x_raw, y_raw, z_raw)

    t = resampled_data['t'].to_numpy()
    x_res = resampled_data['x'].to_numpy()
    y_res = resampled_data['y'].to_numpy()
    z_res = resampled_data['z'].to_numpy()

    # Filtering.
    # This is a test. A filter might be helpful but this step is not necessary
    cutoff_fq = 10
    x = butter_lowpass_filter(x_res, cutoff_fq, 90, 2)
    y = butter_lowpass_filter(y_res, cutoff_fq, 90, 2)
    z = butter_lowpass_filter(z_res, cutoff_fq, 90, 2)
    # x, y, z = x_res, y_res, z_res

    # Note: step is typically the same for all trials (e.g. if 90 Hz, step=1/90)
    # Although step is similar across trials, step is quickly calculated here
    step = t[1] - t[0]
    vx = calculate_velocity(step, x)
    vy = calculate_velocity(step, y)
    vz = calculate_velocity(step, z)

    # Movement Onset Time Detection
    delta_T = 0.1  # 100 ms.
    Ts = step
    m = int(delta_T / Ts) - 1
    tm = m * Ts

    res = onset_detection(m, x, z, t, vx, vz, t_th=t_threshold[trial_number - 1], vel_th=0.6)
    to = res[0]

    # Interpolation to find the corresponding location and velocity
    idx_ub = np.argwhere(t > to).T[0][0]
    idx_lb = np.argwhere(t < to).T[0][-1]

    # Interpolation to find the corresponding location and velocity
    if abs(t[idx_lb] - to) < abs(t[idx_ub] - to):
        idx = idx_lb
    else:
        idx = idx_ub

    print("Trial", trial_number)

    # Total Curvature (2D)
    tot_cur2 = total_curvature_2d(x[idx:], z[idx:])

    print("Total Curvature (2D)", tot_cur2)

    # Total Curvature (3D)
    tot_cur3 = total_curvature_3d(x[idx:], z[idx:], y[idx:])

    print("Total Curvature (3D)", tot_cur3)
