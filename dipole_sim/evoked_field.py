import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.transforms import apply_trans, invert_transform

from slice import create_head_grid
from math_ import find_closest
from download import download_fwd_from_github, download_bem_from_github
from forward import gen_forward_solution


def _update_topomap_label(widget, state, ch_type):
    label = widget['label']['topomap_' + ch_type]
    label_text = state['label_text']['topomap_' + ch_type]
    label.value = label_text


def gen_evoked(dipole_ori, dipole_amplitude, info, fwd):
    dipole_ori /= np.linalg.norm(dipole_ori)
    dipole_ori = dipole_ori.reshape(3, 1)

    # Apply the correct weights to each dimension of the leadfield, which is
    # based on a "free" orientation forward model. This essentially collapses
    # the three "free" orientation dimensions into a single "fixed" orientation
    # dimension.
    leadfield_free = fwd['sol']['data']
    leadfield_fixed = leadfield_free @ dipole_ori

    # Now do the actual forward projection (which simply means: scale the
    # leadfield by the dipole amplitude), and generate an Evoked object.
    meeg_data = leadfield_fixed * dipole_amplitude
    evoked = mne.EvokedArray(meeg_data, info)
    return evoked


def plot_evoked(widget, state, fwd_path, subject, info, ras_to_head_t,
                exact_solution, bem_path=None, head_to_mri_t=None,
                fwd_lookup_table=None, t1_img=None):
    if fwd_lookup_table is None:
        raise ValueError('Must prodive fwd_lookup_table')

    dipole_pos = (state['dipole_pos']['x'],
                  state['dipole_pos']['y'],
                  state['dipole_pos']['z'])
    dipole_ori = (state['dipole_ori']['x'],
                  state['dipole_ori']['y'],
                  state['dipole_ori']['z'])

    # dipole_pos = np.array(dipole_pos).reshape(1, 3)
    # dipole_pos = apply_trans(trans=ras_to_head_t, pts=dipole_pos)
    # dipole_pos /= 1000

    # dipole_ori = np.array(dipole_ori).reshape(1, 3)
    # dipole_ori = apply_trans(trans=ras_to_head_t, pts=dipole_ori, move=False)
    # dipole_ori /= 1000
    # dipole_ori /= np.linalg.norm(dipole_ori)

    # dipole_pos = np.array(dipole_pos).reshape(1, 3).round(3)
    # dipole_ori = np.array(dipole_ori).reshape(1, 3).round(3)

    dipole_pos_ras = np.array(dipole_pos).reshape(1, 3)
    dipole_pos_vox = apply_trans(t1_img.header.get_ras2vox(), dipole_pos_ras)
    dipole_pos_mri = apply_trans(t1_img.header.get_vox2ras_tkr(),
                                 dipole_pos_vox)
    dipole_pos_mri_m = dipole_pos_mri / 1000.
    dipole_pos_head = apply_trans(invert_transform(head_to_mri_t),
                                  dipole_pos_mri_m)
    dipole_pos = dipole_pos_head

    dipole_ori_ras = np.array(dipole_ori).reshape(1, 3)
    dipole_ori_vox = apply_trans(t1_img.header.get_ras2vox(), dipole_ori_ras,
                                 move=False)
    dipole_ori_mri = apply_trans(t1_img.header.get_vox2ras_tkr(),
                                 dipole_ori_vox, move=False)
    dipole_ori_mri_m = dipole_ori_mri / 1000.
    dipole_ori_head = apply_trans(invert_transform(head_to_mri_t),
                                  dipole_ori_mri_m, move=False)
    dipole_ori = dipole_ori_head
    dipole_ori /= np.linalg.norm(dipole_ori)

    dipole_amplitude = state['dipole_amplitude']

    if exact_solution:
        if (bem_path).exists():
            print(f'\nUsing existing BEM solution: {bem_path}\n')
        else:
            print('Retrieving BEM solution from GitHub.')
            try:
                download_bem_from_github(data_path=bem_path.parent,
                                         subject=subject,
                                         overwrite=False)
            except RuntimeError as e:
                msg = (f'Failed to retrieve the BEM solution. '
                       f'The error was: {e}\n')
                raise RuntimeError(msg)
        bem = mne.read_bem_solution(bem_path)
        fwd = gen_forward_solution(pos=dipole_pos, bem=bem, info=info,
                                   trans=head_to_mri_t)
    else:
        # Retrieve the dipole pos closest to the one we have a pre-calculated
        # fwd for.
        pos_head_grid = create_head_grid(info=info)
        dipole_pos_for_fwd = (find_closest(pos_head_grid[0], dipole_pos[0, 0]),
                              find_closest(pos_head_grid[1], dipole_pos[0, 1]),
                              find_closest(pos_head_grid[2], dipole_pos[0, 2]))

        print(f'Requested calculations for dipole located at:\n'
              f'    x={dipole_pos[0, 0]}, y={dipole_pos[0, 1]}, '
              f'z={dipole_pos[0, 2]} [m, MNE Head]\n'
              f'Using a forward solution for the following location:\n'
              f'    x={dipole_pos_for_fwd[0]}, y={dipole_pos_for_fwd[1]}, '
              f'z={dipole_pos_for_fwd[2]} [m, MNE Head]')

        fwd_fname = (f'{subject}-'
                     f'{dipole_pos_for_fwd[0]:.3f}-'
                     f'{dipole_pos_for_fwd[1]:.3f}-'
                     f'{dipole_pos_for_fwd[2]:.3f}-fwd.fif')
        if (fwd_path / fwd_fname).exists():
            print(f'\nUsing existing forward solution: {fwd_fname}\n')
        elif not fwd_lookup_table.loc[str(dipole_pos_for_fwd[0]),
                                      str(dipole_pos_for_fwd[1]),
                                      str(dipole_pos_for_fwd[2])].iloc[0]:
            msg = ('No pre-calculated foward solution available for this '
                   'dipole. Please select a dipole origin clearly inside the '
                   'brain.')
            print(msg)
            return
        else:
            print('Retrieving forward solution from GitHub.\n\n')
            try:
                download_fwd_from_github(fwd_path=fwd_path, subject=subject,
                                         dipole_pos=dipole_pos_for_fwd)
            except RuntimeError as e:
                msg = (f'Failed to retrieve pre-calculated forward solution. '
                       f'The error was: {e}\n\n'
                       f'Please try again with another dipole origin inside '
                       f'the brain.')
                raise RuntimeError(msg)

        fwd = mne.read_forward_solution(fwd_path / fwd_fname)
        del fwd_fname, pos_head_grid, dipole_pos_for_fwd

    evoked = gen_evoked(fwd=fwd,
                        dipole_ori=dipole_ori,
                        dipole_amplitude=dipole_amplitude,
                        info=info)

    for ch_type, fig in widget['topomap_fig'].items():
        ax_topomap = fig.axes[0]
        ax_colorbar = fig.axes[1]
        ax_topomap.clear()
        ax_colorbar.clear()
        ax_topomap.set_aspect('equal')

        if ch_type == 'eeg':
            outlines = 'head'
        else:
            outlines = 'skirt'

        evoked.plot_topomap(ch_type=ch_type,
                            colorbar=False,
                            outlines=outlines,
                            times=evoked.times[-1],
                            show=False,
                            axes=ax_topomap)
        ax_topomap.set_title(None)
        # ax_topomap.format_coord = _create_format_coord('topomap')
        cb = fig.colorbar(ax_topomap.images[-1], cax=ax_colorbar,
                          orientation='horizontal')

        if ch_type == 'mag':
            label = 'fT'
        elif ch_type == 'grad':
            label = 'fT/cm'
        elif ch_type == 'eeg':
            label = 'µV'
            # Scale the topomap to approx. the same size as the MEG topomaps.
            mag_topomap_ax = widget['topomap_fig']['mag'].axes[0]
            ax_topomap.set_xlim(np.array(mag_topomap_ax.get_xlim()) * 1.05)
            ax_topomap.set_ylim(np.array(mag_topomap_ax.get_ylim()) * 1.05)

        cb.set_label(label, fontweight='bold')
        fig.canvas.draw()


def create_topomap_fig():
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.1]},
                           figsize=(2, 2.5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.resizable = False
    fig.canvas.callbacks.callbacks.clear()
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].set_aspect('equal')
    fig.tight_layout()
    fig.set_tight_layout(True)
    return fig


def reset_topomaps(widget, evoked):
    for ch_type in ['mag', 'grad', 'eeg']:
        # Clear topomap.
        widget['topomap_fig'][ch_type].axes[0].clear()

        # Clear colorbar.
        widget['topomap_fig'][ch_type].axes[1].clear()
        widget['topomap_fig'][ch_type].axes[1].set_axis_off()

        plot_sensors(widget=widget, evoked=evoked, ch_type=ch_type)
        widget['topomap_fig'][ch_type].canvas.draw()


def plot_sensors(widget, evoked, ch_type):
    ax = widget['topomap_fig'][ch_type].axes[0]
    evoked.plot_sensors(ch_type=ch_type,
                        title='',
                        show=False,
                        axes=ax)
    ax.collections[0].set_sizes([0.05])
    ax.figure.canvas.draw()
