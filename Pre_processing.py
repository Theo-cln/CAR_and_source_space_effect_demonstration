import numpy as np



def CAR(data):
    """
    Compute the Common Average Reference (CAR) for the data.

    Parameters:
    - data (numpy array): The input data. Expected shape: (n_epochs, n_channels, n_samples).

    Returns:
    - car_data (numpy array): The data after applying the Common Average Reference (CAR).
    """
    # Compute the mean across all channels for each epoch
    channel_mean = np.mean(data, axis=1, keepdims=True)

    # Subtract the mean from each channel in the epoch
    car_data = data - channel_mean

    return car_data