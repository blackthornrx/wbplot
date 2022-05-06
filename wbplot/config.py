from os.path import join
from .constants import DATA_DIR

# TODO: add descriptions

PARCELLATIONS_DIR = join(DATA_DIR, "HumanCorticalParcellations")

PARCELLATION_FILE = join(
    PARCELLATIONS_DIR, "Q1-Q6_RelatedValidation210.CorticalAreas_dil_"
                       "Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii")


WHOLEBRAIN_PARCELLATION_FILE = join(
    DATA_DIR, 'HumanWholeBrainParcellations', 
    'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')


CORTEX_PARCELLATIONS = [
    join(DATA_DIR, 'HumanCorticalParcellations', 'desikan_atlas_32k.dlabel.nii'),
    join(DATA_DIR, 'HumanCorticalParcellations', 'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'),
]
WHOLEBRAIN_PARCELLATIONS = [
    join(DATA_DIR, 'HumanWholeBrainParcellations', 'YeoPlus_Schaefer17Net200_BucknerCBL17Net_MelbourneS3_VentralDC_Brainstem.dlabel.nii'),
    join(DATA_DIR, 'HumanWholeBrainParcellations', 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')
]


SCENE_FILE = join(DATA_DIR, "Human.scene")
SCENE_ZIP_FILE = join(DATA_DIR, "scene.zip")
