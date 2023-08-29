
if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import os, glob
import pandas as pd

from autogluon.tabular import TabularPredictor

# from utils.io.io_utils import load_compressed_pickle

COBDOCK_MODEL_PATH = "models/cobdock"
COBDOCK_COMMERCIAL_USE_MODEL_PATH = "models/cobdock_commercial_use"


# POCKET_PREDICTION_MODEL_FILENAME = "models/model.pkl.gz"

# POCKET_PREDICTION_MODEL_FEATURES = [
#     'vina_rmsd_lb', 
#     'number_fpocket_poses_at_location', 
#     'number_vina_poses_at_location', 
#     'number_galaxydock_poses_at_location', 
#     'vina_min_pose_id', 
#     'vina_distance', 
#     'galaxydock_min_pose_id', 
#     'galaxydock_score', 
#     'galaxydock_autodock', 
#     'galaxydock_internal_energy', 
#     'plants_distance', 
#     'plants_total_score',
#     'plants_score_rb_pen',
#     'plants_score_norm_hevatoms', 
#     'plants_score_norm_weight', 
#     'plants_score_rb_pen_norm_crt_hevatoms', 
#     'zdock_distance', 
#     'zdock_score', 
#     'p2rank_min_pose_id', 
#     'p2rank_distance', 
#     'p2rank_pocket_score', 
#     'p2rank_sas_points', 
#     'p2rank_surf_atoms', 
#     'fpocket_min_pose_id', 
#     'fpocket_pocket_score', 
#     'fpocket_number_of_alpha_spheres', 
#     'fpocket_mean_alpha-sphere_radius', 
#     'fpocket_hydrophobicity_score', 
#     'fpocket_polarity_score', 
#     'fpocket_amino_acid_based_volume_score', 
#     'fpocket_pocket_volume_(convex_hull)', 
#     'fpocket_charge_score', 
#     'fpocket_local_hydrophobic_density_score', 
#     'fpocket_number_of_apolar_alpha_sphere',
# ]

# def load_model(
#     model_filename: str = POCKET_PREDICTION_MODEL_FILENAME,
#     verbose: bool = True,
#     ):
#     if verbose:
#         print ("Loading COBDock model from", model_filename)
#     return load_compressed_pickle(model_filename, verbose=verbose)

COBDOCK_MODEL_FEATURE_NAMES = [
    'sampled_pose_number_vina_poses_at_location', 
    'sampled_pose_number_galaxydock_poses_at_location', 
    'sampled_pose_number_plants_poses_at_location', 
    'sampled_pose_number_zdock_poses_at_location', 
    'vina_min_pose_id', 
    'vina_distance', 
    'vina_rmsd_lb', 
    'vina_rmsd_ub', 
    'galaxydock_min_pose_id', 
    'galaxydock_distance', 
    'galaxydock_score', 
    'galaxydock_autodock', 
    'galaxydock_drug_score', 
    'plants_min_pose_id', 
    'plants_distance', 
    'plants_score_rb_pen', 
    'plants_score_norm_crt_hevatoms', 
    'plants_score_norm_crt_weight', 
    'plants_score_rb_pen_norm_crt_hevatoms', 
    'zdock_distance', 
    'p2rank_min_pose_id', 
    'p2rank_distance', 
    'p2rank_pocket_score', 
    'p2rank_sas_points', 
    'p2rank_surf_atoms', 
    'fpocket_distance', 
    'fpocket_pocket_score', 
    'fpocket_drug_score', 
    'fpocket_number_of_alpha_spheres', 
    'fpocket_polarity_score', 
    'fpocket_pocket_volume_(monte_carlo)',
    'fpocket_pocket_volume_(convex_hull)', 
    'fpocket_local_hydrophobic_density_score', 
    'fpocket_number_of_apolar_alpha_sphere',
]

COBDOCK_COMMERCIAL_USE_MODEL_FEATURE_NAMES = [
    'sampled_pose_number_vina_poses_at_location', 
    'vina_min_pose_id', 
    'vina_distance', 
    'vina_rmsd_lb', 
    'vina_rmsd_ub',
    'p2rank_min_pose_id', 
    'p2rank_distance', 
    'p2rank_pocket_score', 
    'p2rank_sas_points', 
    'p2rank_surf_atoms',
    'fpocket_distance', 
    'fpocket_pocket_score', 
    'fpocket_drug_score', 
    'fpocket_number_of_alpha_spheres',
    'fpocket_polarity_score', 
    'fpocket_pocket_volume_(monte_carlo)', 
    'fpocket_pocket_volume_(convex_hull)',
    'fpocket_local_hydrophobic_density_score', 
    'fpocket_number_of_apolar_alpha_sphere',
]

def load_model(
    commercial_use_only: bool = False,
    verbose: bool = True,
    ):

    if commercial_use_only:
        model_path = COBDOCK_COMMERCIAL_USE_MODEL_PATH
    else:
        model_path = COBDOCK_MODEL_PATH

    if verbose:
        print ("Loading model from", model_path)

    return TabularPredictor.load(model_path, require_py_version_match=False)

def prepare_for_model_prediction(
    dataframe: pd.DataFrame,
    commercial_use_only: bool = False,
    verbose: bool = True,
    ):

    if verbose:
        print ("Preparing data of shape", dataframe.shape, "for input into machine learning model")
    if commercial_use_only:
        feature_names = COBDOCK_COMMERCIAL_USE_MODEL_FEATURE_NAMES
    else:
        feature_names = COBDOCK_MODEL_FEATURE_NAMES

    return dataframe[feature_names]
