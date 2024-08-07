import matplotlib.pyplot as plt
import numpy as np



def PSD_visualisation(Freqs_MI, Freqs_rest, Psd_mean_MI, Psd_mean_rest, electrode_name, phase_name):
    """
    Visualize the Power Spectral Density (PSD) for Motor Imagery (MI) and rest phases.

    Parameters:
        Freqs_MI (ndarray): Frequencies corresponding to the MI phase.
        Freqs_rest (ndarray): Frequencies corresponding to the rest phase.
        Psd_mean_MI (ndarray): PSD values for the MI phase.
        Psd_mean_rest (ndarray): PSD values for the rest phase.
        electrode_name (str): Name of the electrode being analyzed.
        phase_name (str): Name of the phase being visualized.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(Freqs_MI, Psd_mean_MI, color='red', label='MI')
    plt.plot(Freqs_rest, Psd_mean_rest, color='blue', label='Rest')
    plt.title(f'Average Power Spectral Density (PSD) of MI vs Rest ({phase_name} {electrode_name})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectrum (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()



def FC_visualisation(FC_MI, FC_rest, frequency, channel_names, phase_name):
    """
    Visualize the Functional Connectivity (FC) for MI and rest phases.

    Parameters:
        FC_MI (ndarray): Functional connectivity matrix for the MI phase.
        FC_rest (ndarray): Functional connectivity matrix for the rest phase.
        frequency (int): The specific frequency to visualize.
        channel_names (list): List of channel names.
        phase_name (str): Name of the phase being visualized.

    Returns:
        None
    """
    FC_mean_MI = np.mean(FC_MI, axis=0)
    FC_mean_rest = np.mean(FC_rest, axis=0)
    channels = range(len(channel_names))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    im_MI = axes[0].imshow(FC_mean_MI[:, :, frequency], cmap='bwr')
    axes[0].set_title(f'Functional Connectivity MI {phase_name} {frequency} Hz')
    axes[0].set_xticks(channels)
    axes[0].set_xticklabels(channel_names, rotation=90, fontsize=5)
    axes[0].set_yticks(channels)
    axes[0].set_yticklabels(channel_names, fontsize=5)

    im_rest = axes[1].imshow(FC_mean_rest[:, :, frequency], cmap='bwr')
    axes[1].set_title(f'Functional Connectivity Rest {phase_name} {frequency} Hz')
    axes[1].set_xticks(channels)
    axes[1].set_xticklabels(channel_names, rotation=90, fontsize=5)
    axes[1].set_yticks(channels)
    axes[1].set_yticklabels(channel_names, fontsize=5)

    fig.colorbar(im_MI, ax=axes, orientation='vertical')
    plt.show()



def R_squared_map_visualisation(R_squared_map, frequency, channel_names=None, phase_name="", feature_name=""):
    """
    Visualize the R-squared map for a specific feature across channels and frequencies.

    Parameters:
        R_squared_map (ndarray): The R-squared values matrix.
        frequency (list): List of frequency values.
        channel_names (list): List of channel names.
        phase_name (str): Name of the phase being visualized.
        feature_name (str): Name of the feature being visualized.

    Returns:
        None
    """
    if channel_names is None:
        channel_names = []

    channels = range(len(channel_names))
    frequency_indices = range(len(frequency))

    plt.figure(figsize=(16, 16))
    plt.imshow(R_squared_map, cmap='jet', aspect='auto')
    plt.title(f'{phase_name} R-squared Map of {feature_name}')

    if channel_names:
        plt.yticks(channels, channel_names)
    plt.xticks(frequency_indices, frequency)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channels')
    plt.colorbar()
    plt.show()



def visualize_complete_PSD(psd_MI, psd_rest, R_squared_map, freqs_MI, channel_names, phase_name, feature_name):
    """
    Visualizes the Power Spectral Density (PSD) and R-squared map for given motor imagery (MI) and resting state data.

    Parameters:
    - psd_MI (ndarray): The PSD data for motor imagery, shape (trials, channels, frequencies).
    - psd_rest (ndarray): The PSD data for resting state, shape (trials, channels, frequencies).
    - R_squared_map (ndarray): R-squared values for the channels and frequencies.
    - freqs_MI (ndarray): Array of frequency values corresponding to the PSD data.
    - channel_names (list): List of channel names corresponding to the data.
    - phase_name (str): The name of the phase being analyzed (e.g., 'Motor Imagery').
    - feature_name (str): The name of the feature being analyzed (e.g., 'PSD').

    Returns:
    - selected_channels (list): List of indices of the channels with the highest R-squared values.
    """

    # Compute the mean PSD for MI and resting state
    psd_mean_MI = np.mean(psd_MI, axis=0)
    psd_mean_rest = np.mean(psd_rest, axis=0)

    # Visualize the R-squared map
    R_squared_map_visualisation(R_squared_map, freqs_MI, channel_names=channel_names,
                                phase_name=phase_name, feature_name=feature_name)

    # Identify channels with the highest R-squared values
    max_R_squared_per_channel = R_squared_map.max(axis=1)
    top_channel_indices = np.argpartition(max_R_squared_per_channel, -2)[-2:]

    # Or use a custom selection
    sorted_indices = [7, 17]

    # Visualize the PSD for the selected channels
    for channel_index in sorted_indices:
        PSD_visualisation(freqs_MI, freqs_MI, psd_mean_MI[channel_index],
                          psd_mean_rest[channel_index], channel_names[channel_index], phase_name)

    return sorted_indices
