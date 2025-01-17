"""Auxiliary functions pertaining to the manipulation of neuroimaging files. """

import numpy as np
import pandas as pd
import nibabel as nib
from .. import constants, config
from . import plots
from os import system, remove
from matplotlib import colors as clrs
from matplotlib import cm
import xml.etree.cElementTree as eT
from nibabel.cifti2.parse_cifti2 import Cifti2Parser
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib import colorbar
import matplotlib.image as mpimg
import matplotlib.colorbar as mcbar


def map_unilateral_to_bilateral(pscalars, hemisphere):
    """
    Map 180 unilateral pscalars to 360 bilateral pscalars, padding contralateral
    hemisphere with zeros.

    Parameters
    ----------
    pscalars : numpy.ndarray
        unilateral parcellated scalars
    hemisphere : 'left' or 'right' or None

    Returns
    -------
    numpy.ndarray

    """
    hemisphere = check_parcel_hemi(pscalars=pscalars, hemisphere=hemisphere)
    if hemisphere is None:
        return pscalars
    pscalars_lr = np.zeros(360)
    if hemisphere == 'right':
        pscalars_lr[:180] = pscalars
    elif hemisphere == 'left':
        pscalars_lr[180:] = pscalars
    return pscalars_lr


def check_pscalar_dict(pscalar_dict, dlabel_df):
    """
    Check that the pscalar dictionary matches header in dlabel file.

    Parameters
    ----------
    pscalar_dict : dict
        keys=parcel label
        vals=number to plot

    dlabel_df: pd.DataFrame
        Dataframe with 'roi_num' and 'roi' columns.

    Returns
    -------
    None

    Raises
    ------
    ValueError : pscalars is not one-dimensional and length 59412
    """
    if not all(dlabel_df.roi.isin(pscalar_dict.keys())):
        raise ValueError("The keys of 'pscalar_dict' do not match the parcels in dlabel.nii file")


def check_vol_view(vol_view):
    """
    Check that the pscalar dictionary matches header in dlabel file.

    Parameters
    ----------
    vol_view

    Returns
    -------
    vol_view

    Raises
    ------
    ValueError : pscalars is not one-dimensional and length 59412
    """
    valid_vol_views = ['sagittal', 'coronal', 'axial']
    if type(vol_view) is not list:
        if vol_view in valid_vol_views:
            return [vol_view]
        else:
            raise ValueError("Valid parameters for 'vol_view' are ".format(valid_vol_views))
    elif type(vol_view) is list:
        if all(pd.Series(vol_view).isin(valid_vol_views)):
            return vol_view
        else:
            raise ValueError("Valid parameters for 'vol_view' are ".format(valid_vol_views))


def add_colormap(png_file, fig_title, cbar_title, vmin, vmax, cmap):

    # get image size 
    im = Image.open(png_file)
    w, h = im.size
    aspect = w / h

    # create mpl figure
    fig = plt.figure(figsize=(3, 3/aspect))
    img_ax    = fig.add_axes([0.075, 0.075, .85, .85])
    footer_ax = fig.add_axes([.4, .02, .2, .03])
    img = mpimg.imread(png_file)
    im = img_ax.imshow(img)
    img_ax.set_title(fig_title, y=0.95, family='avenir')
    img_ax.axis('off')

    cnorm = clrs.Normalize(vmin=vmin, vmax=vmax)  # only important for tick placing
    cmap = plt.get_cmap(cmap)
    cbar = colorbar.ColorbarBase(
        footer_ax, cmap=cmap, norm=cnorm, orientation='horizontal')
    cbar.set_ticks([-2, 2])  # don't need to do this since we're going to hide them
    cbar.outline.set_visible(False)
    footer_ax.get_xaxis().set_tick_params(length=0, pad=-2)
    cbar.set_ticklabels([])
    footer_ax.text(-0.025, 0.4, vmin, ha='right', va='center', family='avenir', transform=footer_ax.transAxes,
            fontsize=6)
    footer_ax.text(1.025, 0.4, vmax, ha='left', va='center', family='avenir', transform=footer_ax.transAxes,
            fontsize=6)
    #footer_ax.text(.4, 1.025, 'title', ha='left', va='center', transform=footer_ax.transAxes,
    #        fontsize=6)
    cbar.ax.set_title(cbar_title, y=0.5, family='avenir', fontsize=6)
    #cbar.set_label('sadf', loc='center', verticalalignment='top')
    plt.savefig(png_file.replace('.png', '_cbar.png'), dpi=500)
    plt.close()


def check_pscalars_unilateral(pscalars):
    """
    Check that unilateral pscalars have the expected size and shape.

    Parameters
    ----------
    pscalars : numpy.ndarray
        parcellated scalars

    Returns
    -------
    None

    Raises
    ------
    TypeError : pscalars is not array_like
    ValueError : pscalars is not one-dimensional and length 180

    """
    if not isinstance(pscalars, np.ndarray):
        raise TypeError(
            "pscalars: expected array_like, got {}".format(type(pscalars)))
    if pscalars.ndim != 1 or pscalars.size != 180:
        e = "pscalars must be one-dimensional and length 180"
        e += "\npscalars.shape: {}".format(pscalars.shape)
        raise ValueError(e)


def check_pscalars_bilateral(pscalars):
    """
    Check that bilateral pscalars have the expected size and shape.

    Parameters
    ----------
    pscalars : numpy.ndarray
        parcellated scalars

    Raises
    ------
    TypeError : pscalars is not array_like
    ValueError : pscalars is not one-dimensional and length 360

    """
    if not isinstance(pscalars, np.ndarray):
        raise TypeError(
            "pscalars: expected array_like, got {}".format(type(pscalars)))
    if pscalars.ndim != 1 or pscalars.size != 360:
        e = "pscalars must be one-dimensional and length 180"
        e += "\npscalars.shape: {}".format(pscalars.shape)
        raise ValueError(e)


def check_dscalars(dscalars):
    """
    Check that dscalars have the expected size and shape.


    Parameters
    ----------
    dscalars : numpy.ndarray
        dense scalars

    Returns
    -------
    None

    Raises
    ------
    TypeError : pscalars is not array_like
    ValueError : pscalars is not one-dimensional and length 59412

    """
    if not isinstance(dscalars, np.ndarray):
        raise TypeError(
            "dscalars: expected array_like, got {}".format(type(dscalars)))
    if dscalars.ndim != 1 or dscalars.size != 59412:
        e = "dscalars must be one-dimensional and length 59412"
        e += "\ndscalars.shape: {}".format(dscalars.shape)
        raise ValueError(e)


def check_parcel_hemi(pscalars, hemisphere):
    """
    Check hemisphere argument for package compatibility.

    Parameters
    ----------
    pscalars : numpy.ndarray
        parcels' scalar quantities
    hemisphere : 'left' or 'right' or None
        if bilateral, use None

    Returns
    -------
    'left' or 'right' or None

    Raises
    ------
    RuntimeError : pscalars is not length-360 but hemisphere not indicated
    ValueError : invalid hemisphere argument

    """
    if pscalars.size != 360 and hemisphere is None:
        raise RuntimeError(
            "you must indicate which hemisphere these pscalars correspond to")
    options = ['left', 'l', 'L', 'right', 'r', 'R', None, 'lr', 'LR']
    if hemisphere not in options:
        raise ValueError("{} is not a valid hemisphere".format(hemisphere))
    if hemisphere in ['left', 'l', 'L']:
        return 'left'
    if hemisphere in ['right', 'r', 'R']:
        return 'right'
    if hemisphere in ['None', 'lr', 'LR']:
        return None


def check_dense_hemi(hemisphere):
    """
    Check hemisphere argument for compatibility.

    Parameters
    ----------
    hemisphere : 'left' or 'right' or None
        if bilateral, use None

    Returns
    -------
    'left' or 'right' or None

    Raises
    ------
    ValueError : invalid hemisphere argument

    """
    options = ['left', 'l', 'L', 'right', 'r', 'R', None, 'lr', 'LR']
    if hemisphere not in options:
        raise ValueError("{} is not a valid hemisphere".format(hemisphere))
    if hemisphere in ['left', 'l', 'L']:
        return 'left'
    if hemisphere in ['right', 'r', 'R']:
        return 'right'
    if hemisphere in ['None', 'lr', 'LR']:
        return None


def extract_nifti_data(of):
    """Extract array of scalar quantities from a NIFTI2 image.

    Parameters
    ----------
    of : :class:~`nibabel.Nifti2Image` instance
        the NIFTI2 image from which to extract scalar data

    Returns
    -------
    data : numpy.ndarray

    """
    return np.asanyarray(of.dataobj).squeeze()


def extract_gifti_data(of):
    """Extract array of scalar quantities from a GIFTI image.

    Parameters
    ----------
    of : :class:~`nibabel.gifti.GiftiImage` instance
        the GIFTI image from which to extract scalar data

    Returns
    -------
    data : numpy.ndarray

    """
    return np.asanyarray(of.darrays[0].data).squeeze()


def write_parcellated_image(
        data, fout, hemisphere=None, cmap='magma', vrange=None, dlabel=None):
    """
    Change the colors for parcels in a dlabel file to illustrate pscalar data.

    Parameters
    ----------
    data : numpy.ndarray
        scalar map values
    fout : str
        absolute path to output neuroimaging file with *.dlabel.nii* extension
        (if an extension is provided)
    hemisphere : 'left' or 'right' or None, default None
        which hemisphere `pscalars` correspond to. for bilateral data use None
    cmap : str
        a valid MATPLOTLIB colormap used to plot the data
    vrange : tuple
        data (min, max) for plotting; if None, use (min(data), max(data))

    Returns
    -------
    None

    Notes
    -----
    The file defined by wbplot.config.PARCELLATION_FILE is used as a template to
    achieve this. Thus the data provided to this function must be in the same
    parcellation as that file. By default, this is the HCP MMP1.0 parcellation;
    thus, `data` must be ordered as (R_1, R_2, ..., R_180, L_1, L_2, ..., L_180)
    if bilateral. If unilateral, they must be ordered from area V1 (parcel 1) to
    area p24 (parcel 180).
    """
    cmap = plots.check_cmap_plt(cmap)
    # Check provided inputs and pad contralateral hemisphere with 0 if necessary
    if dlabel is None:
        check_parcel_hemi(pscalars=data, hemisphere=hemisphere)
        pscalars_lr = map_unilateral_to_bilateral(
            pscalars=data, hemisphere=hemisphere)
    else:
        pscalars_lr = data

    # Change the colors assigned to each parcel and save to `fout`
    c = Cifti(dlabel=dlabel)
    c.set_cmap(data=pscalars_lr, cmap=cmap, vrange=vrange)
    c.save(fout)


def dlabel_to_dscalar(dlabel, dscalar_out, dataobj=None):

    # read dlabel file
    dlabel_cii = nib.load(dlabel)

    # identify the brainmodels header axis
    for idx in dlabel_cii.header.mapped_indices:
        cur_axis = dlabel_cii.header.get_axis(idx)
        if type(cur_axis) == nib.cifti2.cifti2_axes.BrainModelAxis:
            brain_model_axis = cur_axis

    # create a new scalar axis    
    scalar_axis = nib.cifti2.ScalarAxis(['scalar'])
    # create new header
    new_hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))

    # create new dscalar cifti
    if dataobj is None: 
        new_cii = nib.cifti2.cifti2.Cifti2Image(dataobj=dlabel_cii.dataobj, header=new_hdr)
    else: 
        new_cii = nib.cifti2.cifti2.Cifti2Image(dataobj=dataobj, header=new_hdr)
    nib.save(new_cii, dscalar_out)


def write_dense_image(dscalars, fout, palette='magma', palette_params=None, dscalar_override=False):
    """
    Create a new DSCALAR neuroimaging file.

    Parameters
    ----------
    dscalars : numpy.ndarray
        dense (i.e., whole-brain vertex/voxel-wise) scalar array of length 91282
    fout : str
        absolute path to output neuroimaging file with *.dscalar.nii* extension
        (if an extension is provided)
    palette : str, default 'magma'
        name of color palette
    palette_params : dict or None, default None
        additional (key: value) pairs passed to "wb_command -cifti-palette". for
        more info, see
        https://humanconnectome.org/software/workbench-command/-cifti-palette

    Returns
    -------
    None

    Notes
    -----
    For a list of available color palettes, see the wbplot.constants module or
    visit:
    https://www.humanconnectome.org/software/workbench-command/-metric-palette.

    The file defined by wbplot.config.DSCALAR_FILE is used as a template to
    achieve this. Thus the `dscalars` array provided to this function must be
    contain 59412 elements (i.e., it must include both cortical hemispheres).
    Note the subcortex is not currently supported. In standard bilateral
    cortical dscalar files, elements [0,29695] correspond to the left
    hemisphere and elements [29696,59411] correspond to the right hemisphere.
    Thus, you can simply pad the other elements with NaN if you want only a
    single hemisphere to be plotted.

    Example usage of `palette_params`:
        palette_params = dict()
        palette_params["disp-zero"] = True
        palette_params["disp-positive"] = True
        palette_params["disp-negative"] = False
        palette_params["inversion"] = "POSITIVE_WITH_NEGATIVE"
    The above, passed to this function, would invert the color palette and
    display only positive- and zero-valued scalars (when `fout` is opened in
    wb_view).

    Note that if you wish to define vmin and vmax by hand, you should do one of
    the following:

    >> palette_params = {
        "pos-user": (pos_min, pos_max), "neg-user": (neg_min, neg_max)}
    where pos_min is the minimum positive value shown, pos_max is the maximum
    positive value shown, neg_min is the minimum negative (ie, most negative)
    value shown, and neg_max is the maximum negative (ie, least negative) value
    shown

    or

    >> palette_params = {
        "pos-percent": (pos_min, pos_max), "neg-percent": (neg_min, neg_max)}
    where pos_min, pos_max, neg_min, and neg_max are the same as before but
    expressed as *percentages* of the positive and negative values

    Raises
    ------
    ValueError : palette_params contains an invalid key,value pair

    """
    # TODO: add function for users to map from 32k unilateral to CIFTI subset
    # TODO: implement subcortex

    if fout[-12:] != ".dscalar.nii":  # TODO: improve input handling
        fout += ".dscalar.nii"

    new_data = np.copy(dscalars)

    # Load template dscalar file
    of = nib.load(constants.DSCALAR_FILE)
    temp_data = np.asanyarray(of.dataobj)

    # Write new data to file
    if dscalar_override == False: 
        data_to_write = new_data.reshape(np.shape(temp_data))
        new_img = nib.Cifti2Image(
            dataobj=data_to_write, header=of.header, nifti_header=of.nifti_header)
    else: 
        new_img = nib.load(dscalar_override)

    prefix = fout.split(".")[0]
    cifti_palette_input = prefix + "_temp.dscalar.nii"
    nib.save(new_img, cifti_palette_input)

    # Use Workbench's command line utilities to change color palette
    mode = "MODE_AUTO_SCALE"  # default mode (not DMN, haha)
    disp_zero = disp_neg = disp_pos = True
    if palette_params:
        args = list(palette_params.keys())
        if "pos-percent" in args and "neg-percent" in args:
            mode = "MODE_AUTO_SCALE_PERCENTAGE"
        elif "pos-user" in args and "neg-user" in args:
            mode = "MODE_USER_SCALE"
        if "disp-pos" in args:
            disp_pos = palette_params["disp-pos"]
        if "disp-neg" in args:
            disp_neg = palette_params["disp-neg"]
        if "disp-zero" in args:
            disp_zero = palette_params["disp-zero"]
    cmd = "wb_command -cifti-palette {} {} {}".format(
        cifti_palette_input, mode, fout)
    cmd += " -palette-name {}".format(palette)
    cmd += " -disp-zero {}".format(disp_zero)
    cmd += " -disp-pos {}".format(disp_pos)
    cmd += " -disp-neg {}".format(disp_neg)

    # Update command with provided parameters. NOTE these must be consistent
    # with the format expected by "wb_command -cifti-palette": see
    # https://www.humanconnectome.org/software/workbench-command/-cifti-palette
    if palette_params:
        for k, v in palette_params.items():
            if k in ["disp-zero", "disp-pos", "disp-neg"]:
                continue
            if hasattr(v, '__iter__'):
                if len(v) != 2:
                    raise ValueError(
                        "palette params must be a dict with values which are "
                        "either strings, numbers, or tuples")
                cmd += " -{} {} {}".format(k, v[0], v[1])
            else:
                cmd += " -{} {}".format(k, v)

    # We're ready to change palette and save new file to `fout`
    system(cmd)

    # Remove file which was only used temporarily
    remove(cifti_palette_input)


class Cifti(object):
    """
    A class for changing the colors inside the metadata of a DLABEL neuroimaging
    file. Some of this code was contributed by Dr. Murat Demirtas while he was
    a post-doctoral researcher at Yale.
    """

    def __init__(self, dlabel=None):
        if dlabel == None: 
            of = nib.load(config.PARCELLATION_FILE)  # must be a DLABEL file!!
            self.dlabel = config.PARCELLATION_FILE
        else: 
            of = nib.load(dlabel)
            self.dlabel = dlabel
        self.data = np.asanyarray(of.dataobj)
        self.header = of.header
        self.nifti_header = of.nifti_header
        # self.extensions = eT.fromstring(  BROKEN AS OF NIBABEL 3.2
        #     self.nifti_header.extensions[0].get_content().to_xml())
        self.tree = eT.fromstring(self.header.to_xml())
        self.vrange = None
        self.ischanged = False

    def set_cmap(self, data, cmap='magma', vrange=None, mappable=None):
        """
        Map scalar data to RGBA values and update file header metadata.

        Parameters
        ----------
        data : numpy.ndarray
            scalar data
        cmap : str or None, default 'magma'
            colormap to use for plotting
        vrange : tuple or None, default None
            data (min, max) for illustration; if None, use (min(data),max(data))
        mappable : Callable[float] or None, default None
            can be used to override arguments `cmap` and `vrange`, e.g. by
            specifying your own map from scalar input to RGBA output

        Returns
        -------
        None

        """
        if (data.size != 360) and (self.dlabel == config.PARCELLATION_FILE):
            raise RuntimeError(
                "pscalars must be length 360 for :class:~wbplot.images.Cifti")

        # Check input arguments
        cmap = plots.check_cmap_plt(cmap)
        self.vrange = (
            np.min(data), np.max(data)) if vrange is None else vrange
        self.vrange = plots.check_vrange(self.vrange)

        # Map scalar data to colors (R, G, B, Alpha)
        if mappable is None:
            cnorm = clrs.Normalize(vmin=self.vrange[0], vmax=self.vrange[1])
            clr_map = cm.ScalarMappable(cmap=cmap, norm=cnorm)
            colors = clr_map.to_rgba(data)
        else:
            colors = np.array([mappable(d) for d in data])

        # Update file header metadata
        for ii in range(1, len(self.tree[0][1][0][0])):
            self.tree[0][1][0][0][ii].set(
                'Red', str(colors[ii - 1, 0]))
            self.tree[0][1][0][0][ii].set(
                'Green', str(colors[ii - 1, 1]))
            self.tree[0][1][0][0][ii].set(
                'Blue', str(colors[ii - 1, 2]))
            self.tree[0][1][0][0][ii].set(
                'Alpha', str(colors[ii - 1, 3]))
        self.ischanged = True

    def save(self, fout):
        """
        Write self.data to image `fout`.

        Parameters
        ----------
        fout : str
            absolute path to output neuroimaging file. must be a DLABEL file!!

        Returns
        -------
        None

        """
        if self.ischanged:
            cp = Cifti2Parser()
            cp.parse(string=eT.tostring(self.tree))
            header = cp.header
        else:
            header = self.header
        if fout[-11:] != ".dlabel.nii":  # TODO: improve input handling
            fout += ".dlabel.nii"
        new_img = nib.Cifti2Image(
            self.data, header=header, nifti_header=self.nifti_header)
        nib.save(new_img, fout)


# Pythonic version of this workbench command (primarily so I don't forget)
def cifti_parcellate(cifti_in, dlabel_in, cifti_out, direction='COLUMN'):
    cmd = "wb_command -cifti-parcellate {} {} {} {}".format(
        cifti_in, dlabel_in, direction, cifti_out)
    system(cmd)
