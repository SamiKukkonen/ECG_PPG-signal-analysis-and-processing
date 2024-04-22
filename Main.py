import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # This is needed at the end to save multiple figures into a single PDF file.
from scipy.signal import find_peaks

# The following line creates interactive graphs.
%matplotlib widget

def parse_file(filename, fs):
    """
    Reads data in a CSV file into a 2D NumPy array.

    Parameters
    ----------
    filename : str
        The name of the data file.
    fs : int
        Desired sampling frequency in the resampled data.

    Returns
    -------
    timestamps_s : ndarray
        An array of timestamps in seconds, measured from the 
        beginning of the measurement.
    ppg : ndarray
        Resampled PPG data.
    ecg : ndarray
        Resampled ECG data.
    """
    # Read the data into 2D array.

    data = np.loadtxt(filename, delimiter=',')
    #

    # An array of timestamps with the starting timestamp
    # having the largest timestamp of the first timestamps
    # (i.e. here ECG because its samples are read after PPG) 
    # and the last timestamp having the smallest timestamp
    # of the last timestamps. The difference between the 
    # timestamps is defined by the sampling rate, i.e.
    # 1_000_000 / sampling rate. NOTE: Here the timestamps are
    # assumed to be in microseconds.
    ts_raw = np.arange(data[0, 2], data[-1, 0], 1_000_000 / fs)
    # The same array but in seconds from the beginning of 
    # the measurement.
    # ts = ...
    # 
    ts = ts_raw / 1e6  # Convert timestamps to seconds
    #

    # Resampling. Interpolate both PPG and ECG data using the 
    # created timestamp array. The extrapolation fill_value
    # actually does not matter if you create the timestamp
    # array as above (which is better than relying on 
    # extrapolation)
    ppg = interpolate.interp1d(data[:, 0], data[:, 1], 
        'cubic', fill_value='extrapolate')(ts_raw)
    ecg = interpolate.interp1d(data[:, 2], data[:, 3], 
        'cubic', fill_value='extrapolate')(ts_raw)
    
    return ts, ppg, ecg

fs_200 = 200
fs = 500
ts_200, ppg_raw_200, ecg_raw_200 = parse_file("data_while_sitting.csv", fs_200)
ts, ppg_raw, ecg_raw = parse_file("data_while_sitting.csv", fs)

fig, ax = plt.subplots()
ax.plot(ts_200, ecg_raw_200, label='ECG (200 Hz)', alpha=0.7)
ax.plot(ts, ecg_raw, label='ECG (500 Hz)', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('ECG Signal')
ax.legend()


ax.spines[['right', 'top']].set_visible(False) # Remove unnecessary spines to make the figure "more airy".
fig.tight_layout()

# Show plot
plt.show()

def plot_signals(ts, ss, fid_locs=None, titles=None, y_labels=None, fig_title=None):
    """
    Plots multiple signals as a subplot.

    Parameters
    ----------
    ts : array_like
        An array of timestamps that will be used as the x-axis.
    ss : array_like
        A 2D array of signals where each row is a signal (s).
    fid_locs : array_like
        A 2D array (array-like) of fiducial point locations
        where each row corresponds to the
        fiducial points on the respective row of ss.
    titles : array_like
        An array of titles for each s in ss.
    y_labels : array_like
        An array of y-axis labels for each s in ss.

    Returns
    -------
    fig : matplotlib.figure
        Figure object.
    """
    fig, axes = plt.subplots(len(ss), 1, sharex=True)

    for i, (ax, s) in enumerate(zip(axes, ss)):
        ax.plot(ts, s, c='tab:blue')
        if fid_locs is not None:
            ax.plot(ts[fid_locs[i]], s[fid_locs[i]], linestyle='none', marker='o', c='r')
        if titles is not None:
            ax.set_title(titles[i])
        if y_labels is not None:
            ax.set_ylabel(y_labels[i])
        ax.spines[['right', 'top']].set_visible(False)

    axes[-1].set_xlabel('Time [s]')
    if fig_title is not None:
        fig.suptitle(fig_title)
    fig.tight_layout()

    return fig

_ = plot_signals(ts, np.array([ecg_raw, ppg_raw]), titles=['ECG Signal', 'PPG Signal'], y_labels=['ECG', 'PPG'], fig_title='Raw ECG and PPG Signals')
def butterworth_filter(s, fs, wn, btype='bandpass'):
    """
    Performs Butterworth filtering.

    Parameters
    ----------
    arr : array_like
        Signal which will be filtered.
    fs : int
        Sampling frequency of the signal.
    wn : array_like
        Cutoff frequencies ([lower, upper]).
    btype : str
        Type of the filter. Default is 'bandpass'.

    Returns
    -------
    s_filt : ndarray
        Filtered signal.
    """

    sos = signal.butter(10, wn, btype=btype, fs=fs, output='sos')
    s_filt = signal.sosfiltfilt(sos, s)
    return s_filt

ppg_filt = butterworth_filter(-ppg_raw, fs, [0.5, 8], btype='bandpass')
ecg_filt = butterworth_filter(ecg_raw, fs, [0.5, 40], btype='bandpass')

_ = plot_signals(ts, [ecg_filt, ppg_filt], titles=['ECG', 'PPG'], y_labels=['ADC value [counts]'] * 2, fig_title='Filtered signals')

def normalize(s, min_val=0.0, max_val=1.0):
    """
    Performs normalization into the given interval.

    Parameters
    ----------
    s : array_like
        Signal which will be normalized.
    min_val : float
        Minimum value of the desired interval.
    max_val : float
        Maximum value of the desired interval.

    Returns
    -------
    s_norm : ndarray
        Normalized signal.    
    """

    s_norm = (s - np.min(s)) / (np.max(s) - np.min(s)) * (max_val - min_val) + min_val


    return s_norm

ppg_norm = normalize(ppg_filt)
ecg_norm = normalize(ecg_filt)

_ = plot_signals(ts, [ecg_norm, ppg_norm], titles=['ECG', 'PPG'], 
                 y_labels=['ADC value [counts]'] * 2, fig_title='Normalized signals')

ppg_feet, _ = find_peaks(-ppg_norm, height=-0.25)  
ecg_r_peaks, _ = find_peaks(ecg_norm, height=0.8, distance=100, prominence=0.2)  

_ = plot_signals(ts, [ecg_norm, ppg_norm], [ecg_r_peaks, ppg_feet], titles=['ECG', 'PPG'], 
                 y_labels=['ADC value [counts]'] * 2, fig_title='Normalized signals with fiducial points')

def compute_hr_hrv(timestamps, fid_points):
    """
    Computes HR and HRV from ECG R peaks and PPG feet.

    Parameters
    ----------
    timestamps : array_like
        An array of signal timestamps in seconds.
    fid_points : array_like
        An array of fiducial points (e.g. ECG R peaks or PPG feet).
    """
    hr = 60 / np.mean(np.diff(timestamps[fid_points]))
    hrv = np.std(np.diff(timestamps[fid_points])) * 1000

    print(f'HR of {hr} bpm, HRV of {hrv} ms')

compute_hr_hrv(ts, ppg_feet)
compute_hr_hrv(ts, ecg_r_peaks)

def plot_nn_intervals(timestamps, ecg_r_peaks, ppg_feet):
    """
    Plot NN intervals.

    Parameters
    ----------
    timestamps : ndarray
        An array of timestamps in seconds.
    ecg_r_peaks : ndarray
        An array of ECG R peaks.
    ppg_feet : ndarray
        An array of PPG waveform feet.

    Returns
    -------
    fig : matplotlib.figure
        Figure object.
    """

    fig, ax = plt.subplots()

    ax.plot(timestamps[ecg_r_peaks[:-1]], np.diff(timestamps[ecg_r_peaks]), label='ECG NN intervals')
    ax.plot(timestamps[ppg_feet[:-1]], np.diff(timestamps[ppg_feet]), label='PPG NN intervals')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NN Intervals (s)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    fig.suptitle('NN intervals')
    fig.tight_layout()

    return fig

_ = plot_nn_intervals(ts, ecg_r_peaks, ppg_feet)

def extract_waveforms(arr, fid_points, mode='fid_to_fid'):
    """
    Extracts waveforms from a signal using an array of 
    fiducial points.

    Parameters
    ----------
    arr : ndarray
        Signal from which the waveforms are extracted.
    fid_points : ndarray
        Fiducial points that are used to extract the
        waveforms.
    mode : str
        How the fiducial points are used to extract the waveforms:
        - fid_to_fid: from one fiducial point to the next one.
        For example, from one PPG foot to another one.
        - nn_interval: the waveform is extracted around each
        fiducial point by taking half of the NN interval before
        and after.

    Returns
    -------
    waveforms : ndarray
        An array of extracted waveforms where each row corresponds
        to one waveform.
    mean_waveform : ndarray
        The calculated mean waveform.
    """
    # Max NN interval.
    nn_max = np.max(np.diff(fid_points))
    if mode == 'fid_to_fid':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points) - 1, int(nn_max)), np.nan)
        # Loop through the fiducial points in pairs.
        for i, fds in enumerate(zip(fid_points[:-1], fid_points[1:])):
            waveforms[i, :int(fds[1] - fds[0])] = arr[int(fds[0]):int(fds[1])]
    
    elif mode == 'nn_interval':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points) - 2, int(nn_max)), np.nan)
        # Center point of the longest NN interval.
        nn_max_center = nn_max // 2
        # Loop through the fiducial points starting from the second
        # until the second last.
        for i in range(1, len(fid_points) - 1):
            # Number of samples to take from left and right.

            samples_left = (fid_points[i] - fid_points[i - 1]) // 2
            samples_right = (fid_points[i + 1] - fid_points[i]) // 2

            # Place the waveform into the matrix.
            waveforms[i - 1, nn_max_center - samples_left:nn_max_center + samples_right] = \
                arr[fid_points[i] - samples_left:fid_points[i] + samples_right]

        # Remove columns with just NaN values. These columns could happen due to
        # integer divisions used above. This line is just a way to get rid of 
        # nanmean's "Mean of empty slice" warning.
        waveforms = waveforms[:, ~np.isnan(waveforms).all(axis=0)]
    
    # Compute the mean waveform.
    mean_waveform = np.nanmean(waveforms, 0)

    return waveforms, mean_waveform

ppg_wfs, ppg_mean_wf = extract_waveforms(ppg_norm, ppg_feet)
ecg_wfs, ecg_mean_wf = extract_waveforms(ecg_norm, ecg_r_peaks, 'nn_interval')

def plot_waveforms(wfs, wfs_mean, fs, title=''):
    """
    Plots signal waveforms along with their mean.

    Parameters
    ----------
    wfs : ndarray
        A 2D array of waveforms where each row is an individual waveform.
    wfs_mean : ndarray
        Mean waveform.
    fs : int
        Sampling frequency.
    title : str, optional
        Title of the figure.

    Returns
    -------
    fig : matplotlib.figure
        Figure object.
    """
    # Create an array of timestamps with a length equal to the length
    # of the longest waveform.
    ts_wf = np.arange(wfs.shape[1]) / fs

    fig, ax = plt.subplots()

    for wf in wfs:
        ax.plot(ts_wf, wf, alpha=0.5)
    ax.plot(ts_wf, wfs_mean, color='black', linewidth=2, label='Mean Waveform')


    # Set x-ticks to have an interval of 0.1 seconds.
    x_labels = np.arange(0, ts_wf[-1], 0.1).round(1)
    ax.set_xticks(x_labels, x_labels)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[arb. unit]')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    if title != '':
        fig.suptitle(title)
    fig.tight_layout()

    return fig


# ECG waveforms
_ = plot_waveforms(ecg_wfs, ecg_mean_wf, fs, title='ECG Waveforms')
# PPG waveforms
_ = plot_waveforms(ppg_wfs, ppg_mean_wf, fs, title='PPG Waveforms')

def compute_pat(timestamps, ecg_r_peaks, ppg_feet):
    """
    Computes pulse arrival time (PAT) from ECG R peaks and PPG feet.

    Parameters
    ----------
    timestamps : ndarray
        An array of signal timestamps in seconds.
    ecg_r_peaks : ndarray
        ECG R peak locations.
    ppg_feet : ndarray
        PPG feet locations.

    Returns
    -------
    pats : ndarray
        An array of PAT values for each R peak.
    """
    pats = np.empty(len(ecg_r_peaks))
    pats.fill(np.nan)
    for i, r in enumerate(ecg_r_peaks):
        # Find the PPG feet after the R peak.
        f_idxs = np.argwhere(ppg_feet - r >= 0).flatten()
        f_idxs = np.argwhere(ppg_feet >= r).flatten()
        if f_idxs.size > 0:
            # Potential feet exist, take the first one.
            # NOTE: This is a very simple approach, in reality some
            # thresholds might be a good idea.
            f = ppg_feet[f_idxs[0]]
            # The time difference between the foot and the R peak
            # is the PAT.

            pats[i] = timestamps[f] - timestamps[r]
    
    # Let's convert the seconds into milliseconds.
    pats *= 1000
    
    return pats

pat = compute_pat(ts, ecg_r_peaks, ppg_feet)

def plot_pat(timestamps, ecg_r_peaks, pat):
    """
    Plots PAT time series.

    Parameters
    ----------
    timestamps : ndarray
        An array of timestamps in seconds.
    ecg_r_peaks : ndarray
        An array of ECG R peaks.
    pat : ndarray
        An array of PAT values corresponding to
        the locations of the R peaks.

    Returns
    -------
    fig : matplotlib.figure
        Figure object.
    """
    fig, ax = plt.subplots()
    
    ax.plot(timestamps[ecg_r_peaks], pat, linestyle='-')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('PAT [ms]')

    ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('PAT')
    fig.tight_layout()

    return fig

_ = plot_pat(ts, ecg_r_peaks, pat)