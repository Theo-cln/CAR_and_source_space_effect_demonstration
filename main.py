import mne
from Data_importation import import_edf_files
from Pre_processing import CAR
from Features_extraction import PSD, compute_rsquare_map_welch, imaginary_coherence, node_strength
from Visualisation import FC_visualisation, visualize_complete_PSD
from signal_creator import generate_car_signals, generate_source_space_signals
from source import source

print("Importation done")

# Variables and Parameters
fe = 500  # Sampling frequency
tmin, tmax = 1, 4  # Time window in seconds
lowLimit, highLimit = 4, 40  # Frequency band limits

# Signal Parameters
freq1, freq2 = 8, 25  # Frequencies for the signals
amplitudeMI, amplitudeRest = 1, 4  # Signal amplitudes for MI and rest
amplitudeNoise = 0.5  # Noise standard deviation
nb_channels, nb_trials = 64, 30  # Number of channels and trials

phase_name = "Synthetic Signal"

# Import EDF file
folder = "/Users/theo.coulon/data/sub-02/ses-01/"
edf_files = import_edf_files(folder)
raw = mne.io.read_raw_edf(edf_files[0][0], preload=False)
channel_names = raw.info['ch_names']
electrodes = [channel_names.index('CP1'), channel_names.index('C3')]
print(f"Selected electrodes: {electrodes}")

# CAR Effect Simulation
MI, rest, noise = generate_car_signals(freq1, freq2, amplitudeMI, amplitudeRest, tmax - tmin, fe, amplitudeNoise, nb_trials, nb_channels, electrodes)
print(f"Shape of MI and rest signals: {MI.shape}, {rest.shape}")

# Apply Common Average Reference (CAR)
car_MI = CAR(MI)
car_rest = CAR(rest)

# Compute PSD and R-squared maps before and after CAR
freqs_MI, psd_MI = PSD(MI, fe, lowLimit, highLimit)
freqs_rest, psd_rest = PSD(rest, fe, lowLimit, highLimit)
R_squared_map = compute_rsquare_map_welch(psd_MI, psd_rest)

freqs_car_MI, psd_car_MI = PSD(car_MI, fe, lowLimit, highLimit)
freqs_car_rest, psd_car_rest = PSD(car_rest, fe, lowLimit, highLimit)
R_squared_car_map = compute_rsquare_map_welch(psd_car_MI, psd_car_rest)

# Visualize PSD and R-squared maps
visualize_complete_PSD(psd_MI, psd_rest, R_squared_map, freqs_MI, channel_names, phase_name, "PSD")
visualize_complete_PSD(psd_car_MI, psd_car_rest, R_squared_car_map, freqs_car_MI, channel_names, phase_name, "PSD")

# Compute Functional Connectivity before and after CAR
FC_MI = imaginary_coherence(MI, fe, lowLimit, highLimit)
FC_rest = imaginary_coherence(rest, fe, lowLimit, highLimit)
FC_visualisation(FC_MI, FC_rest, 25, channel_names, phase_name)

FC_car_MI = imaginary_coherence(car_MI, fe, lowLimit, highLimit)
FC_car_rest = imaginary_coherence(car_rest, fe, lowLimit, highLimit)
FC_visualisation(FC_car_MI, FC_car_rest, 25, channel_names, phase_name)

# Node Strength Analysis before and after CAR
node_strength_MI = node_strength(FC_MI)
node_strength_rest = node_strength(FC_rest)
R_squared_map_NS = compute_rsquare_map_welch(node_strength_MI, node_strength_rest)
visualize_complete_PSD(node_strength_MI, node_strength_rest, R_squared_map_NS, freqs_MI, channel_names, phase_name, "Node Strength")

node_strength_car_MI = node_strength(FC_car_MI)
node_strength_car_rest = node_strength(FC_car_rest)
R_squared_map_car_NS = compute_rsquare_map_welch(node_strength_car_MI, node_strength_car_rest)
visualize_complete_PSD(node_strength_car_MI, node_strength_car_rest, R_squared_map_car_NS, freqs_car_MI, channel_names, phase_name, "Node Strength")

# Source Space Analysis
activated_dipoles = [5, 12, 17, 24, 29]  # Indices of activated dipoles
MI_src, rest_src, noise_src, MI_fwd, rest_fwd = generate_source_space_signals(tmax - tmin, amplitudeNoise, activated_dipoles, visualize_raw=False)

MI_fwd_src, rest_fwd_src, _, _ = source(MI_fwd, noise_src, rest_fwd, clean=False, parcellation=True)

# Compute PSD and R-squared maps for source and forward models
freqs_src_MI, psd_src_MI = PSD(MI_src, fe, lowLimit, highLimit)
freqs_src_rest, psd_src_rest = PSD(rest_src, fe, lowLimit, highLimit)
freqs_fwd_MI, psd_fwd_MI = PSD(MI_fwd, fe, lowLimit, highLimit)
freqs_fwd_rest, psd_fwd_rest = PSD(rest_fwd, fe, lowLimit, highLimit)
freqs_fwd_src_MI, psd_fwd_src_MI = PSD(MI_fwd_src, fe, lowLimit, highLimit)
freqs_fwd_src_rest, psd_fwd_src_rest = PSD(rest_fwd_src, fe, lowLimit, highLimit)

print(f"Shape of PSD fwd src MI: {psd_fwd_src_MI.shape}")

R_squared_src_map = compute_rsquare_map_welch(psd_src_MI, psd_src_rest)
R_squared_fwd_map = compute_rsquare_map_welch(psd_fwd_MI, psd_fwd_rest)
R_squared_fwd_src_map = compute_rsquare_map_welch(psd_fwd_src_MI, psd_fwd_src_rest)

print(f"Shape of R-squared src map: {R_squared_src_map.shape}")

# Define channels for visualization
src_channel = list(range(60)) # look only at the first 60 dipoles for more clarity
src_channel_2 = list(range(psd_fwd_src_MI.shape[1]))

# Visualize PSD for source and forward models
visualize_complete_PSD(psd_src_MI, psd_src_rest, R_squared_src_map[:60], freqs_src_MI, src_channel, phase_name, "PSD")
visualize_complete_PSD(psd_fwd_MI, psd_fwd_rest, R_squared_fwd_map, freqs_fwd_MI, channel_names, phase_name, "PSD")
visualize_complete_PSD(psd_fwd_src_MI, psd_fwd_src_rest, R_squared_fwd_src_map[:60], freqs_fwd_src_MI, src_channel, phase_name, "PSD")

# Functional Connectivity Analysis for Source Space
FC_src_MI = imaginary_coherence(MI_src, fe, lowLimit, highLimit, activated_dipoles)
FC_src_rest = imaginary_coherence(rest_src, fe, lowLimit, highLimit, activated_dipoles)
FC_fwd_MI = imaginary_coherence(MI_fwd, fe, lowLimit, highLimit, activated_dipoles)
FC_fwd_rest = imaginary_coherence(rest_fwd, fe, lowLimit, highLimit, activated_dipoles)
FC_fwd_src_MI = imaginary_coherence(MI_fwd_src, fe, lowLimit, highLimit, activated_dipoles)
FC_fwd_src_rest = imaginary_coherence(rest_fwd_src, fe, lowLimit, highLimit, activated_dipoles)

print(FC_MI)
print(FC_src_MI.shape)


print(f"Shape of source FC: {FC_src_MI.shape}, {FC_src_rest.shape}")

# Visualize Functional Connectivity for specific frequencies
for frequency in [16, 33]:
    FC_visualisation(FC_src_MI, FC_src_rest, frequency, activated_dipoles, phase_name)
    FC_visualisation(FC_fwd_MI, FC_fwd_rest, frequency, activated_dipoles, phase_name)
    FC_visualisation(FC_fwd_src_MI, FC_fwd_src_rest, frequency, activated_dipoles, phase_name)
