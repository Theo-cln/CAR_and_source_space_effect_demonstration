import numpy as np
from scipy.signal import welch, csd
from tqdm import tqdm


def PSD(signal, fs, low_limit, high_limit):
    """
    Calculate the Power Spectral Density (PSD) mean of the signal.

    Parameters:
    - signal (numpy array): Input signal data, expected shape (n_trials, n_channels, n_samples).
    - fs (float): Sampling rate of the signal.
    - low_limit (float): Lower frequency limit for PSD analysis.
    - high_limit (float): Upper frequency limit for PSD analysis.

    Returns:
    - f (numpy array): Frequency values corresponding to the PSD.
    - PSD_dB (numpy array): PSD values in dB.
    - low_freq_idx (int): Index of the low frequency limit.
    - high_freq_idx (int): Index of the high frequency limit.
    """
    window_size = int(0.5 * fs)  # Window size: 0.5 seconds
    overlap_size = int(0.25 * fs)  # Overlap size: 0.25 seconds
    PSD_all_trials = []

    for trial in signal:
        PSD_trial = []
        for channel in trial:
            f, Pxx = welch(channel, fs=fs, window='hamming', nperseg=window_size, noverlap=overlap_size,
                           scaling='density', nfft=500, detrend='constant')
            PSD_trial.append(Pxx)
        PSD_all_trials.append(PSD_trial)

    PSD_all_trials = np.array(PSD_all_trials)
    PSD_dB = 10 * np.log10(PSD_all_trials)

    # Select frequency range of interest
    low_freq_idx = np.where(f >= low_limit)[0][0]
    high_freq_idx = np.where(f <= high_limit)[0][-1]
    f = f[low_freq_idx:high_freq_idx + 1]
    PSD_dB = PSD_dB[:, :, low_freq_idx:high_freq_idx + 1]

    return f, PSD_dB



def compute_rsquare_map_welch(power_trials_1, power_trials_2, feature_name="PSD"):
    """
    Compute the R-square map between two sets of power spectral densities.

    Parameters:
    - power_trials_1 (numpy array): PSD data for condition 1 (e.g., motor imagery).
    - power_trials_2 (numpy array): PSD data for condition 2 (e.g., rest).
    - feature_name (str): Name of the feature being analyzed (e.g., "PSD" or "NS").

    Returns:
    - Rsquare_map (numpy array): The R-square values indicating the strength of the difference between conditions.
    """
    n_channels, n_freqs = power_trials_1.shape[1], power_trials_1.shape[2]
    Rsquare_map = np.zeros((n_channels, n_freqs))

    for ch in range(n_channels):
        for freq in range(n_freqs):
            sum_q = np.sum(power_trials_1[:, ch, freq])
            sum_r = np.sum(power_trials_2[:, ch, freq])
            n1, n2 = len(power_trials_1[:, ch, freq]), len(power_trials_2[:, ch, freq])
            sumsq1 = np.sum(np.square(power_trials_1[:, ch, freq]))
            sumsq2 = np.sum(np.square(power_trials_2[:, ch, freq]))
            G = ((sum_q + sum_r) ** 2) / (n1 + n2)
            Rsquare_map[ch, freq] = (sum_q ** 2 / n1 + sum_r ** 2 / n2 - G) / (sumsq1 + sumsq2 - G)


    return Rsquare_map



def imaginary_coherence(data, freq, low_limit, high_limit, activated_dipoles=None):
    """
    Compute the imaginary coherence of a signal.

    Parameters:
    - data (numpy array): Input signal data, expected shape (n_trials, n_channels, n_samples).
    - freq (float): Sampling rate of the signal.
    - low_limit (float): Lower frequency limit for coherence analysis.
    - high_limit (float): Upper frequency limit for coherence analysis.

    Returns:
    - im_coherence (numpy array): Imaginary coherence matrix for each pair of channels and frequencies.
    """

    if activated_dipoles is not None:
        data = data[:, activated_dipoles, :]

    window_size = int(0.5 * freq)  # 0.5 seconds
    overlap_size = int(0.25 * freq)  # 0.25 seconds
    n_trials, n_channels = data.shape[0], data.shape[1]
    im_coherence = np.zeros((n_trials, n_channels, n_channels, 251))


    for trial_idx in range(n_trials):
        progress_bar = tqdm(total=n_channels, desc=f'Processing trial {trial_idx+1}/{n_trials}', unit='%')
        for ch1 in range(n_channels):
            for ch2 in range(ch1 + 1, n_channels):
                freqs, Pxx = welch(data[trial_idx, ch1], fs=freq, window='hamming', nperseg=window_size, noverlap=overlap_size,
                               scaling='density', nfft=500)
                _, Pyy = welch(data[trial_idx, ch2], fs=freq, window='hamming', nperseg=window_size, noverlap=overlap_size,
                               scaling='density', nfft=500)
                _, Pxy = csd(data[trial_idx, ch1], data[trial_idx, ch2], fs=freq, window='hamming', nperseg=window_size,
                             noverlap=overlap_size, scaling='density', nfft=500)
                Icoh = np.abs(np.imag(Pxy)) / (Pxx * Pyy)
                im_coherence[trial_idx, ch1, ch2] = Icoh
                im_coherence[trial_idx, ch2, ch1] = Icoh
            progress_bar.update(1)
            progress_bar.set_postfix({"progress": f"{progress_bar.n/progress_bar.total:.0%}"})
        progress_bar.close()
    low_freq_idx = np.where(freqs >= low_limit)[0][0]
    high_freq_idx = np.where(freqs <= high_limit)[0][-1]
    im_coherence = im_coherence[:, :, :, low_freq_idx:high_freq_idx + 1]

    return im_coherence

def node_strength(coherence_matrix):
    """
    Calculate the node strength of the coherence matrix by summing the coherence values across all channels.

    Parameters:
    - coherence_matrix (numpy array): The coherence matrix, expected shape (n_trials, n_channels, n_channels, n_freqs).

    Returns:
    - node_strengths (numpy array): Node strength values for each channel and frequency.
    """
    n_trials, n_channels, _, n_freqs = coherence_matrix.shape
    node_strengths = np.zeros((n_trials, n_channels, n_freqs))

    for trial in range(n_trials):
        for channel in range(n_channels):
            node_strengths[trial, channel] = np.sum(coherence_matrix[trial, channel], axis=0)
            node_strengths[trial, channel] += np.sum(coherence_matrix[trial, :, channel], axis=0)

    return node_strengths
