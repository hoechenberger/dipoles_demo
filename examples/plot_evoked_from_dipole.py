# Author: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)
import numpy as np
import os.path as op
import nibabel as nib

import mne

from mne.datasets import sample
from mne.viz import plot_dipole_locations
from mne.transforms import apply_trans, invert_transform

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
t1w_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
evoked_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
bem_fname = op.join(subjects_dir, subject, 'bem',
                    'sample-5120-5120-5120-bem-sol.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
trans = mne.read_trans(trans_fname)

preset_coords = {
    'Preset 1': dict(pos=[2.94, -76.54, -0.38],
                     ori=[1., 1., 1.]),
    'Preset 2': dict(pos=[-50.70, -23.55, 53.65],
                     ori=[0., 1., 0.]),
    'Preset 3': dict(pos=[21.60, 80.03, 34.01],
                     ori=[1., 0., 0.])}

t1_img = nib.load(t1w_fname)


def ras_to_head(pos_ras, move=True):
    pos_vox = apply_trans(t1_img.header.get_ras2vox(), pos_ras, move=move)
    pos_mri = apply_trans(t1_img.header.get_vox2ras_tkr(), pos_vox, move=move)
    pos_mri_m = pos_mri / 1000.
    pos_head = apply_trans(invert_transform(trans), pos_mri_m, move=move)
    return pos_head


for name, coords in preset_coords.items():
    # Make dipole object
    times = np.array([0])
    dipole_pos = ras_to_head(np.array(coords['pos']).reshape(1, 3))
    dipole_ori = np.array(coords['ori']).reshape(1, 3)
    dipole_ori = ras_to_head(dipole_ori, move=False)
    dipole_ori /= np.linalg.norm(dipole_ori)
    amplitude = np.array([1e-9])
    gof = np.array([100])
    dip = mne.Dipole(times=times, pos=dipole_pos, ori=dipole_ori,
                     amplitude=amplitude, gof=gof)

    # Plot dipole in 3D
    plot_dipole_locations([dip], trans=trans, subject=subject,
                          subjects_dir=subjects_dir, title=name)

    # Make forward model using sample evoked info
    info = mne.io.read_info(evoked_fname)
    fwd, stc = mne.make_forward_dipole(dip, bem_fname, info, trans_fname)

    # Simulate evoked
    evoked = mne.simulation.simulate_evoked(
        fwd, stc, info, cov=None, nave=np.inf)

    # Plot the resulting topomap
    for ch_type in ('mag', 'grad', 'eeg'):
        evoked.plot_topomap(times=times, ch_type=ch_type, title=ch_type)
