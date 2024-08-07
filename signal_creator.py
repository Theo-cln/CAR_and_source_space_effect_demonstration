import numpy as np
import matplotlib.pyplot as plt
import random
import mne
import os.path as op
from mne.datasets import fetch_fsaverage

def generate_car_signals(freq1, freq2, amp_mi, amp_rest, duration, sfreq, noise_amp, n_trials, n_channels, target_channels):
    """
    Generate synthetic signals for Motor Imagery (MI) and Rest conditions, with added noise.

    Parameters:
    - freq1, freq2: Frequencies of the sinusoidal components (Hz)
    - amp_mi, amp_rest: Amplitudes of the MI and Rest signals
    - duration: Duration of the signal in seconds
    - sfreq: Sampling frequency (Hz)
    - noise_amp: Standard deviation of the noise
    - n_trials: Number of trials
    - n_channels: Number of EEG channels
    - target_channels: Indices of channels where specific signals will be applied

    Returns:
    - mi_signals: Synthetic MI signals (trials x channels x time points)
    - rest_signals: Synthetic Rest signals (trials x channels x time points)
    - noise_signals: Noise-only signals (trials x channels x time points)
    """
    n_points = int(duration * sfreq)
    time = np.linspace(0, duration, n_points)

    # Create base sinusoids
    sinusoid_8hz = amp_rest * np.sin(2 * np.pi * freq1 * time)
    sinusoid_25hz = amp_rest * np.sin(2 * np.pi * freq2 * time)
    sinusoid_8hz_weak = amp_mi * np.sin(2 * np.pi * freq1 * time)
    sinusoid_25hz_weak = amp_mi * np.sin(2 * np.pi * freq2 * time)

    # Initialize signal matrices
    mi_signals = np.zeros((n_trials, n_channels, n_points))
    rest_signals = np.zeros((n_trials, n_channels, n_points))
    noise_signals = np.zeros((n_trials, n_channels, n_points))

    # Generate signals for each trial and channel
    for trial in range(n_trials):
        for ch in range(n_channels):
            noise = np.random.normal(0, scale=noise_amp, size=n_points)
            if ch in target_channels:
                rest_signals[trial, ch, :] = noise + sinusoid_8hz + sinusoid_25hz
                mi_signals[trial, ch, :] = noise + sinusoid_8hz_weak + sinusoid_25hz_weak
            else:
                noise_sinusoid = np.sum([
                    random.choice([0.1, 0.3, 0.6, 0, 0, 0, 0, 0, 0, 0]) * np.sin(2 * np.pi * random.randint(4, 40) * time)
                    for _ in range(random.randint(0, 20))
                ], axis=0)
                rest_signals[trial, ch, :] = noise + sinusoid_8hz + noise_sinusoid
                mi_signals[trial, ch, :] = noise + sinusoid_8hz_weak

            noise_signals[trial, ch, :] = noise

    # Plot example signals
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, mi_signals[4, 4], label='MI Signal')
    plt.plot(time, rest_signals[4, 4], 'r--', label='Rest Signal')
    plt.title('Example Signal - TP7')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    plt.plot(time, mi_signals[29, target_channels[0]], label='MI Signal (C3)')
    plt.plot(time, rest_signals[29, target_channels[0]], 'r--', label='Rest Signal (C3)')
    plt.title('C3 Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    return mi_signals, rest_signals, noise_signals

def generate_channel_labels(n_channels, ground, ref):
    """
    Generate channel labels for EEG electrodes based on the number of channels and specific ground/reference electrodes.

    Parameters:
    - n_channels: Number of channels (32 or 64)
    - ground: Ground electrode label
    - ref: Reference electrode label

    Returns:
    - channels: List of channel labels with modified ground and reference electrodes
    """
    if n_channels == 32:
        channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
            'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
            'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
        ]
    elif n_channels == 64:
        channels = [
            'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1',
            'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4',
            'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5',
            'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
            'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8',
            'AF4', 'F2', 'Iz'
        ]

    # Modify ground and reference electrode labels
    if ground in channels:
        channels[channels.index(ground)] = 'AFz' if n_channels == 32 else 'Fpz'
    if ref in channels:
        channels[channels.index(ref)] = 'FCz'

    return channels

def generate_source_space_signals(duration, noise_std, active_dipoles, visualize_raw=False):
    """
    Create synthetic source space signals and project them onto the scalp using forward modeling.

    Parameters:
    - duration: Duration of the signals (seconds)
    - noise_std: Standard deviation of added noise
    - active_dipoles: List of dipole indices with significant activity
    - visualize_raw: Whether to visualize the raw EEG data (default: False)

    Returns:
    - source_signals_mi: Source space signals for MI condition
    - source_signals_rest: Source space signals for Rest condition
    - noise_signals: Noise-only signals (trials x channels x time points)
    - scalp_signals_mi: Projected scalp signals for MI condition
    - scalp_signals_rest: Projected scalp signals for Rest condition
    """
    # Fetch the fsaverage subject data
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)
    subject = "fsaverage"
    trans = "fsaverage"
    sfreq = 500

    n_points = int(duration * sfreq)
    time = np.linspace(0, duration, n_points)

    # Set up source space and BEM model
    src = mne.setup_source_space(subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir)
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    n_sources = 2052
    n_samples = 30

    # Generate synthetic source activity
    np.random.seed(42)
    source_signals_mi = np.random.randn(n_samples, n_sources, n_points)
    source_signals_rest = np.random.randn(n_samples, n_sources, n_points)

    # Introduce significant activity in selected dipoles
    for dipole in active_dipoles:
        source_signals_rest[:, dipole] += 4 * np.sin(2 * np.pi * 8 * time) + 4 * np.sin(2 * np.pi * 25 * time)

    # Create source estimates
    stc_mi_list = [
        mne.SourceEstimate(source_signals_mi[i], vertices=[src[0]['vertno'], src[1]['vertno']], tmin=0., tstep=1/sfreq)
        for i in range(n_samples)
    ]
    stc_rest_list = [
        mne.SourceEstimate(source_signals_rest[i], vertices=[src[0]['vertno'], src[1]['vertno']], tmin=0., tstep=1/sfreq)
        for i in range(n_samples)
    ]

    # Channel information
    ch_names = generate_channel_labels(64, 'AFz', 'FCz')
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Create forward solution
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1, meg=False)

    # Apply forward model to project source signals onto the scalp
    scalp_signals_mi = np.array([mne.apply_forward(fwd, stc, info).data for stc in stc_mi_list])
    scalp_signals_rest = np.array([mne.apply_forward(fwd, stc, info).data for stc in stc_rest_list])

    # Visualize raw data if required
    if visualize_raw:
        raw_mi_list = [mne.io.RawArray(data, info) for data in scalp_signals_mi]
        raw_rest_list = [mne.io.RawArray(data, info) for data in scalp_signals_rest]
        raw_mi_list[0].plot()
        raw_rest_list[0].plot()

    # Generate noise signals
    noise_signals = np.random.normal(0, scale=noise_std, size=(n_samples, 64, n_points))

    return source_signals_mi, source_signals_rest, noise_signals, scalp_signals_mi, scalp_signals_rest
