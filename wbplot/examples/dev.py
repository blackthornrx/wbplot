import numpy as np
import pandas as pd
import nibabel as nb
import wbplot
import quilt3
import tempfile

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib import colorbar
import matplotlib.image as mpimg


# Read GBC data
neuro_p    = quilt3.Package.browse("embarc/neuroimaging", registry="s3://trestle-curated")
gbc_file = '/Users/kevin.anderson/task-rest_concatenated_proc-Atlas_s_hpss_res-mVWMWB1d_lpss_frameCensor_study-EMBARC_atlas-YeoPlus_stat-pearsonRtoZ_GBC.csv.gz'
neuro_p['functional/funccon/GBC/task-rest_concatenated_proc-Atlas_s_hpss_res-mVWMWB1d_lpss_frameCensor_study-EMBARC_atlas-YeoPlus_stat-pearsonRtoZ_GBC.csv.gz'].fetch(gbc_file)
gbc_df   = pd.read_csv(gbc_file, compression='gzip')

# GBC dict to plot
gbc_mean = gbc_df.filter(regex='LH|RH').mean()
gbc_dict = dict(gbc_mean)

# dlabel reference
dlabel = wbplot.config.WHOLEBRAIN_PARCELLATIONS[0]

# define pallete parameters
vrange = (np.percentile(gbc_mean, 2), np.percentile(gbc_mean, 98))
vmin  = -0.06
vmax = 0.06
palette_params = {'pos-user':(0,vmax), 'neg-user':(0,vmin)}

file_out = '/Users/kevin.anderson/GBC_YeoPlus.png'
png_list = wbplot.wbplot.pscalar_from_dict(file_out, gbc_dict, dlabel, orientation='landscape', 
            vol_view=['sagittal','axial','coronal'], palette_params=palette_params,
            hemisphere=None, palette='cool-warm', transparent=False)

for png_file in png_list:
    wbplot.wbplot.images.add_colormap(png_file, 'GBC', 'GBC', vmin, vmax, 'coolwarm')




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
    cax.get_xaxis().set_tick_params(length=0, pad=-2)
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



png_file = '/Users/kevin.anderson/GBC_YeoPlus_Sagittal.png'
vmin = -.06
vmax = .06
cmap = 'coolwarm'
def add_colormap(png_file, vmin, vmax, cmap):
    # get image size 
    im = Image.open(png_file)
    w, h = im.size
    aspect = w / h

    # create mpl figure
    fig = plt.figure(figsize=(3, 3/aspect))
    ax  = fig.add_axes([0.075, 0.05, 0.85, 0.85])
    cax = fig.add_axes([0.44, 0.03, 0.12, 0.03])

    img = mpimg.imread(png_file)
    im = ax.imshow(img)
    ax.axis('off')

    # Now let's add a colorbar and we're done. We'll use -2 and +2 as the limits,
    # since that's what we used for vrange when generating the image.
    cnorm = clrs.Normalize(vmin=vmin, vmax=vmax)  # only important for tick placing
    cmap = plt.get_cmap(cmap)
    cbar = colorbar.ColorbarBase(
        cax, cmap=cmap, norm=cnorm, orientation='horizontal')
    cbar.set_ticks([-2, 2])  # don't need to do this since we're going to hide them
    cax.get_xaxis().set_tick_params(length=0, pad=-2)
    cbar.set_ticklabels([])

    cax.text(-0.025, 0.4, vmin, ha='right', va='center', transform=cax.transAxes,
            fontsize=6)
    cax.text(1.025, 0.4, vmax, ha='left', va='center', transform=cax.transAxes,
            fontsize=6)
    cbar.outline.set_visible(False)
    ax.text(0.5, 1.0, "Human T1w/T2w", transform=ax.transAxes,
            va='bottom', ha='center', fontsize=9)
    ax.text(0.5, -0.01, "z-score", transform=ax.transAxes,
            va='bottom', ha='center', fontsize=6)
    plt.savefig(png_file.replace('.png', '_cbar.png'), dpi=500)
    plt.close()


import matplotlib.colorbar as mcbar
cax, cbar_kwds = mcbar.make_axes(img_ax, location = 'botton',
                            fraction=0.15, shrink=0.5, aspect=20)









# Now we have to load the map into matplotlib. One way to do this is the
# following:
img = mpimg.imread(file_out)
im = ax.imshow(img)
ax.axis('off')

# Now let's add a colorbar and we're done. We'll use -2 and +2 as the limits,
# since that's what we used for vrange when generating the image.
cnorm = clrs.Normalize(vmin=-2, vmax=2)  # only important for tick placing
cmap = plt.get_cmap('magma')
cbar = colorbar.ColorbarBase(
    cax, cmap=cmap, norm=cnorm, orientation='horizontal')
cbar.set_ticks([-2, 2])  # don't need to do this since we're going to hide them
cax.get_xaxis().set_tick_params(length=0, pad=-2)
cbar.set_ticklabels([])
cax.text(-0.025, 0.4, "-2", ha='right', va='center', transform=cax.transAxes,
         fontsize=6)
cax.text(1.025, 0.4, "+2", ha='left', va='center', transform=cax.transAxes,
         fontsize=6)
cbar.outline.set_visible(False)
ax.text(0.5, 1.015, "Human T1w/T2w", transform=ax.transAxes,
        va='bottom', ha='center', fontsize=9)
ax.text(0.5, 0.1, "z-score", transform=ax.transAxes,
        va='bottom', ha='center', fontsize=6)
plt.savefig(join(output_dir, "colorbar_test.png"), dpi=500)
plt.close()
# voila!











neuro_p    = quilt3.Package.browse("embarc/neuroimaging", registry="s3://trestle-curated")
cabnp_tsnr = pd.read_csv('/Users/kevin.anderson/task-restrun_1_proc-Atlas_study-EMBARC_atlas-CABNP_stat-TSNR.csv.gz', compression='gzip')
parcel_columns = cabnp_tsnr.columns[cabnp_tsnr.columns.str.contains('_L|_R')]
avg_tsnr       = cabnp_tsnr[parcel_columns].mean()

pscalar_dict = dict(avg_tsnr)
dlabel       = wbplot.config.WHOLEBRAIN_PARCELLATIONS[1]

file_out = '/Users/kevin.anderson/test.png'
orientation='landscape'
hemisphere=None
vrange=None
cmap='magma'
transparent=False

vol_view = 'sagittal'
    

file_out = '/Users/kevin.anderson/test.png'
wbplot.wbplot.pscalar_from_dict(file_out, pscalar_dict, dlabel, orientation='landscape', 
            vol_view=['sagittal','axial','coronal'], 
            hemisphere=None, vrange=None, cmap='magma', transparent=False)


x = nb.load(wbplot.config.WHOLEBRAIN_PARCELLATIONS[1])
y = nb.load(wbplot.config.WHOLEBRAIN_PARCELLATIONS[0])

gbc_file = '/Users/kevin.anderson/task-rest_concatenated_proc-Atlas_s_hpss_res-mVWMWB1d_lpss_frameCensor_study-EMBARC_atlas-YeoPlus_stat-pearsonRtoZ_GBC.csv.gz'
neuro_p['functional/funccon/GBC/task-rest_concatenated_proc-Atlas_s_hpss_res-mVWMWB1d_lpss_frameCensor_study-EMBARC_atlas-YeoPlus_stat-pearsonRtoZ_GBC.csv.gz'].fetch(gbc_file)
gbc_df = pd.read_csv(gbc_file, compression='gzip')
gbc_mean = gbc_df.filter(regex='LH|RH').mean()

dlabel       = wbplot.config.WHOLEBRAIN_PARCELLATIONS[0]
gbc_dict = dict(gbc_mean)

file_out = '/Users/kevin.anderson/GBC_YeoPlus.png'
vrange = (np.percentile(gbc_mean, 2), np.percentile(gbc_mean, 98))
wbplot.wbplot.pscalar_from_dict(file_out, gbc_dict, dlabel, orientation='landscape', 
            vol_view=['sagittal','axial','coronal'], 
            hemisphere=None, vrange=vrange, cmap='coolwarm', transparent=False)






dlabel_df = extract_dlabel_dict(dlabel)


file_out = '/Users/kevin.anderson/test.png'
orientation='landscape'
hemisphere=None
vrange=None
cmap='magma'
transparent=False

vol_view = 'sagittal'
    
import nibabel as nib
dlabel = wbplot.constants.DLABEL_FILE
dlabel = wbplot.config.WHOLEBRAIN_PARCELLATIONS[1]
dscalar = wbplot.constants.DSCALAR_FILE
def dlabel_to_dscalar(dlabel):

    dlabel_cii = nib.load(dlabel)
    dscalar_cii = nib.load(dscalar)
    
    scalar_axis = nib.cifti2.ScalarAxis(['scalar'])
    brain_axis  = dlabel_cii.header.get_axis(1)
    new_hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))

    new_cii = nib.ci    

    new_cii = nib.cifti2.cifti2.Cifti2Image(dataobj=dlabel_cii.dataobj, header=new_hdr)
    nb.save(new_cii, '/Users/kevin.anderson/test.dscalar.nii')

    dlabel_cii.dataobj[0]

    temp_dir = tempfile.gettempdir()
    vol_nii  = split(dlabel)[1].replace('.dlabel.nii', '.nii')
    vol_temp = join(temp_dir, vol_nii)

    lh_gii  = split(dlabel)[1].replace('.dlabel.nii', '_L.func.gii')
    lh_tmp = join(temp_dir, lh_gii)

    rh_gii  = split(dlabel)[1].replace('.dlabel.nii', '_R.func.gii')
    rh_tmp = join(temp_dir, rh_gii)

    split_cmd = [
        'wb_command', '-cifti-separate', dlabel,
        'COLUMN',
        '-volume-all', vol_temp, 
        '-metric', 'CORTEX_LEFT', lh_tmp,
        '-metric', 'CORTEX_RIGHT', rh_tmp
    ]
    subprocess.call(dlabel)

def make_pscalar_arr(pscalar_dict, dlabel, dlabel_df):
    """
    Map the dlabel roi number to the new data to plot

    Parameters
    ----------
    pscalar_dict : dict
        keys=parcel label
        vals=value to plot

    dlabel_df: pd.DataFrame
        Dataframe with 'roi_num' and 'roi' columns.

    dlabel: str
        Full path to the reference dlabel cifti

    Returns
    -------
    plot_pscalar: np.arr
        An array of values to plot
    """
    dlabel_cii  = nb.load(dlabel)
    dlabel_data = dlabel_cii.dataobj[0]
    
    dlabel_df['plot_vals'] = dlabel_df.roi.map(pscalar_dict)
    map_dict = dict(zip(dlabel_df.roi_num, dlabel_df.plot_vals))
    
    plot_pscalar = np.array(list(map_dict.values()))
    return plot_pscalar


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


def pscalar_from_dict(file_out, pscalar_dict, dlabel, orientation='landscape', vol_view=None, 
            hemisphere=None, vrange=None, cmap='magma', transparent=False):
    """
    Save an image of parcellated scalars using Connnectome Workbench.

    Parameters
    ----------
    file_out : str
        absolute path to filename where image is saved. if `filename` has an
        extension, it must be .png, e.g. fout="/Users/jbb/Desktop/test.png"
    pscalar_dict : numpy.ndarray # TODO 
        parcel scalar values
    orientation : 'portrait' or 'landscape', default 'landscape'
        orientation of the output image. if hemisphere is None (i.e., if data
        are bilateral), this argument is ignored.
    hemisphere : 'left' or 'right' or None, default None
        which hemisphere `pscalars` correspond to. if bilateral, use None
    vrange : tuple or None, default None
        data (min, max) for plotting
    cmap : str, default 'magma'
        MATPLOTLIB colormap to use for plotting
    transparent : bool, default False
        make all white pixels in resultant image transparent

    Returns
    -------
    None

    Raises
    ------
    RuntimeError : scene file was not copied to temp directory

    """
    # Check file extension
    if file_out[-4:] != ".png":  # TODO: improve input handling
        file_out += ".png"

    # Perform checks on inputs
    cmap = plots.check_cmap_plt(cmap)
    orientation = plots.check_orientation(orientation)

    # extract label parcel information from cifti header
    dlabel_df = extract_dlabel_dict(dlabel)
    check_pscalar_dict(pscalar_dict, dlabel_df)

    # replace dlabel integers with data to plot
    pscalars = make_pscalar_arr(pscalar_dict, dlabel, dlabel_df)

    # Write `pscalars` to the neuroimaging file which is pre-loaded into the
    # scene file, and update the colors for each parcel using the file metadata
    temp_dir   = tempfile.gettempdir()
    temp_cifti = join(temp_dir, split(constants.DLABEL_FILE)[1])
    images.write_parcellated_image(
        data=pscalars, fout=temp_cifti, cmap=cmap, vrange=vrange, dlabel=dlabel)

    # This is just to prevent error messages written to console because
    # ImageDense.dscalar.nii doesn't exist in the scene directory
    dscalar_out = join(temp_dir, split(constants.DSCALAR_FILE)[1])
    cmd = "cp {} {}".format(constants.DSCALAR_FILE, dscalar_out)
    system(cmd)
    mni_out = join(temp_dir, split(constants.MNI152_FILE)[1])
    cmd = "cp {} {}".format(constants.MNI152_FILE, mni_out)
    system(cmd)

    # Now copy the scene file & HumanCorticalParcellations directory to the
    # temp directory as well
    with ZipFile(config.SCENE_ZIP_FILE, "r") as z:  # unzip to temp dir
        z.extractall(temp_dir)
    scene_file = join(temp_dir, "Human.scene")
    if not exists(scene_file):
        raise RuntimeError(
            "scene file was not successfully copied to {}".format(scene_file))

    # Map the input parameters to the appropriate scene in the scene file
    scene, width, height = plots.map_params_to_scene(
        dtype='pscalars', orientation=orientation, hemisphere=hemisphere)

    # Call Connectome Workbench's command-line utilities to generate an image
    cmd = 'wb_command -show-scene "{}" {} "{}" {} {}'.format(
        scene_file, scene, file_out, width, height)
    system(cmd)
    if transparent:  # Make background (defined as white pixels) transparent
        plots.make_transparent(file_out)

    # plot requested volumetric viewss
    if vol_view is not None: 
        vol_view = check_vol_view(vol_view)
        for cur_vol_view in vol_view: 
            cur_file_out = file_out.replace('.png', f'_{cur_vol_view}.png')
            scene, width, height = plots.map_params_to_scene(
                dtype='pscalars', orientation=orientation, hemisphere=hemisphere, vol_view=cur_vol_view)
            # Call Connectome Workbench's command-line utilities to generate an image
            cmd = 'wb_command -show-scene "{}" {} "{}" {} {}'.format(
                scene_file, scene, cur_file_out, width, height)
            # cmd += " >/dev/null 2>&1"
            system(cmd)
            if transparent:  # Make background (defined as white pixels) transparent
                plots.make_transparent(cur_file_out)


