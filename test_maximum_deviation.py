'''
Test 'maximum deviation' using real data
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from maximum_deviation import maximum_deviation_2d, maximum_deviation_3d
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

for trial_number in range(17, 20 + 1, 1):

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

    idx_ub = np.argwhere(t > to).T[0][0]
    idx_lb = np.argwhere(t < to).T[0][-1]

    # Interpolation to find the corresponding location and velocity
    if abs(t[idx_lb] - to) < abs(t[idx_ub] - to):
        idx = idx_lb
    else:
        idx = idx_ub

    # Maximum Perpendicular Deviation (2D)
    max_dev2, max_dis2, x_max2, z_max2, idx_max2 = maximum_deviation_2d(x[idx:],
                                                                        z[idx:])

    # Maximum Perpendicular Deviation (3D)
    max_dev3, max_dis3, x_max3, z_max3, y_max3, idx_max3 = maximum_deviation_3d(x[idx:],
                                                                                z[idx:],
                                                                                y[idx:])
    # Plotting results
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(True)
    ax.plot(x[idx:], z[idx:], 'b.', label='trajectory')
    ax.plot([x[idx], x[-1]], [z[idx], z[-1]], '--', color='grey', label='direct line')
    ax.plot(x_max2, z_max2, 'ro', label='max per dev')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title("Max Dis: " + str(max_dis2))
    ax.legend()

    ax = fig.add_subplot(gs[0, 1], projection='3d')
    ax.plot(x[idx:], z[idx:], y[idx:], 'b.', label='trajectory')
    # NOTE: Remember that the sign is defined with respect to a plane
    ax.plot([x[idx], x[-1]], [z[idx], z[-1]], [y[idx], y[-1]], '--', color='grey', label='plane')
    ax.plot([x[idx], x[-1]], [z[idx], z[-1]], [y[idx], y[idx]], '--', color='grey')
    ax.plot([x[-1], x[-1]], [z[-1], z[-1]], [y[idx], y[-1]], '--', color='grey')
    ax.plot(x_max3, z_max3, y_max3, 'ro', label='max per dev')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title("Max Dis: " + str(max_dis3))
    ax.legend()
    ax.set_xlim(-0.3, 0.3)
    plt.show()
