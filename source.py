from mne.minimum_norm import make_inverse_operator,apply_inverse_epochs
import mne
from mne.datasets import fetch_fsaverage
import os.path as op
import numpy as np


def channel_generator(number_of_channel, Ground, Ref):
    if number_of_channel == 32:
        electrodes = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
        for i in range(len(electrodes)):
            if (electrodes[i] == Ground):
                index_gnd = i
            if (electrodes[i] == Ref):
                index_ref = i
        electrodes[index_gnd] = 'AFz'
        electrodes[index_ref] = 'FCz'

    if number_of_channel == 64:
        #electrodes = ['FP1','FP2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10','C5','C1','C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7','PO3','POz','PO4','PO8']
        electrodes = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','Iz']
        for i in range(len(electrodes)):
            if (electrodes[i] == Ground):
                index_gnd = i
            if (electrodes[i] == Ref):
                index_ref = i
        electrodes[index_gnd] = 'Fpz'
        electrodes[index_ref] = 'FCz'

    return electrodes

def source(Signal_MI, Signal_MI_ba, Signal_Rest, clean=False, parcellation=True):
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # The files live in:
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    irm_path = op.join(fs_dir,"mri","aparc+aseg.mgz")

    # Créer une info synthétique
    ch_names = ['EEG %03d' % i for i in range(1, 65)]  # Exemple pour 64 canaux EEG
    ch_types = ['eeg'] * 64
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)

    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    car_bool = False
    # Download fsaverage files
    src = mne.setup_source_space(
        subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir
    )
    parc = "aparc_sub"
    labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)

    """ 
   mne.viz.plot_alignment(
        info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        src=src,
        surfaces=['white'],  # Vous pouvez ajouter plus de surfaces comme 'pial', 'inner_skull', etc.
        coord_frame='mri'  # Utilisez le cadre de référence MRI pour une meilleure précision
    )
    """
    electrodes = channel_generator(64, 'TP9', 'TP10')
    biosemi_montage_inter = mne.channels.make_standard_montage('standard_1020')

    ind = [i for (i, channel) in enumerate(biosemi_montage_inter.ch_names) if channel in electrodes]
    biosemi_montage = biosemi_montage_inter.copy()
    # Keep only the desired channels
    biosemi_montage.ch_names = [biosemi_montage_inter.ch_names[x] for x in ind]
    print(biosemi_montage.ch_names)
    kept_channel_info = [biosemi_montage_inter.dig[x + 3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    biosemi_montage.dig = biosemi_montage_inter.dig[0:3] + kept_channel_info
    # biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    n_channels = len(biosemi_montage.ch_names)
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=500 / 2,
                           ch_types='eeg')
    pos = np.stack([biosemi_montage.get_positions()['ch_pos'][ch] for ch in electrodes])
    print(biosemi_montage)
    info.set_montage(biosemi_montage)


    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=-1, meg=False
    )

    leadfield = fwd['sol']['data']

    # Create an events array
    n_events = Signal_MI.shape[0]  # Number of events
    event_id = {'event{}'.format(i): i for i in range(n_events)}  # Event IDs
    events = np.column_stack((np.arange(n_events), np.zeros(n_events, dtype=int), np.arange(n_events)))
    # # use the CSDs and the forward model to build the DICS beamformer


    #fwd = mne.convert_forward_solution(fwd, surf_ori=True)
    # Create the Epochs object
    Sig_MI_Reorder = np.zeros((Signal_MI.shape[0],Signal_MI.shape[1],Signal_MI.shape[2]))
    Sig_Rest_Reorder = np.zeros((Signal_Rest.shape[0],Signal_Rest.shape[1],Signal_Rest.shape[2]))
    Signal_MI_ba_reorder = np.zeros((Signal_MI_ba.shape[0],Signal_MI_ba.shape[1],Signal_MI_ba.shape[2]))
    for k in range(len(biosemi_montage.ch_names)):
          for l in range(len(electrodes)):
              if biosemi_montage.ch_names[k] == electrodes[l]:
                  Sig_MI_Reorder[:,k,:] = Signal_MI[:,l,:]
                  Sig_Rest_Reorder[:, k, :] = Signal_Rest[:, l, :]
                  Signal_MI_ba_reorder[:,k,:] = Signal_MI_ba[:, l, :]
    tmin = 0.0  # Start time of each epoch
    epochs_MI = mne.EpochsArray(Sig_MI_Reorder[:,:,:], info, events, tmin, event_id)
    mne.set_eeg_reference(epochs_MI, ref_channels='average', projection=True)
    epochs_Rest = mne.EpochsArray(Sig_Rest_Reorder[:,:,:], info, events, tmin, event_id)
    mne.set_eeg_reference(epochs_Rest, ref_channels='average', projection=True)
    epoch_noise = mne.EpochsArray(Signal_MI_ba_reorder, info, events, tmin, event_id)
    mne.set_eeg_reference(epoch_noise, ref_channels='average', projection=True)
    noise_cov = mne.compute_covariance(epoch_noise, method='auto', rank=None)
    inverse_operator = make_inverse_operator(
          info, fwd, noise_cov,  depth=None
      )

    method = "MNE"
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    fmin = 13.
    fmax = 26.
    sfreq = 500  # the sampling frequency

    stc_mi = apply_inverse_epochs(epochs_MI,inverse_operator,lambda2,method=method,pick_ori=None)
    print("stc_mi : \n", stc_mi)
    label_ts_mi = np.asarray(mne.extract_label_time_course(
        stc_mi, labels_parc, src, mode="mean",allow_empty = True
    ))
    stc_Rest = apply_inverse_epochs(epochs_Rest,inverse_operator,lambda2,method=method,pick_ori=None)
    print("stc_rest : \n", stc_Rest)
    label_ts_Rest = np.asarray(mne.extract_label_time_course(
        stc_Rest, labels_parc, src, mode="mean",allow_empty = True
    ))

    stc_MI = np.asarray([stc.data for stc in stc_mi])
    stc_rest = np.asarray([stc.data for stc in stc_Rest])

    # Prendre en compte les 3 composantes directionnelles
    source_data_3d_MI = np.zeros((Signal_MI.shape[0], leadfield.shape[1], stc_MI.shape[2]))
    for trial in range(Signal_MI.shape[0]):
        source_data_3d_MI[trial, ::3, :] = stc_MI[trial]  # Composante x
        source_data_3d_MI[trial, 1::3, :] = stc_MI[trial]  # Composante y
        source_data_3d_MI[trial, 2::3, :] = stc_MI[trial]  # Composante z

    source_data_3d_rest = np.zeros((Signal_Rest.shape[0], leadfield.shape[1], stc_rest.shape[2]))
    for trial in range(Signal_Rest.shape[0]):
        source_data_3d_rest[trial, ::3, :] = stc_rest[trial]  # Composante x
        source_data_3d_rest[trial, 1::3, :] = stc_rest[trial]  # Composante y
        source_data_3d_rest[trial, 2::3, :] = stc_rest[trial]  # Composante z

    MI_fwd = leadfield @ source_data_3d_MI
    rest_fwd = leadfield @ source_data_3d_rest


    motor_labels = [label.name for label in labels_parc if 'postcentral' in label.name or 'precentral' in label.name]
    motor_labels_lh = [label for label in motor_labels if 'lh' in label]
    motor_labels_rh = [label for label in motor_labels if 'rh' in label]

    parcel_indices_motor_lh = []
    parcel_indices_motor_rh = []

    # Find indices of each parcel name in labels_parc
    for parcel_name in motor_labels_lh:
        index = [label.name for label in labels_parc].index(parcel_name)
        parcel_indices_motor_lh.append(index)
    for parcel_name in motor_labels_rh:
        index = [label.name for label in labels_parc].index(parcel_name)
        parcel_indices_motor_rh.append(index)

    parcel_indices_motor = parcel_indices_motor_lh + parcel_indices_motor_rh

    label_mi = label_ts_mi[:, parcel_indices_motor, :]

    label_rest = label_ts_Rest[:, parcel_indices_motor, :]

    if clean:
        positions = [5, 12, 17, 24, 29]
        for i in range(label_mi.shape[1]):
            if i not in positions:
                label_mi[:, i, :] = label_rest[:, i, :]
        return np.asarray(label_mi), np.asarray(label_rest), MI_fwd, rest_fwd

    elif parcellation:
        return np.asarray(label_mi), np.asarray(label_rest), MI_fwd, rest_fwd

    else :
        return stc_MI,stc_rest, MI_fwd, rest_fwd



