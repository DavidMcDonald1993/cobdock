
if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import os, glob

from utils.io.io_utils import load_compressed_pickle

POCKET_PREDICTION_MODEL_FILENAME = "models/model.pkl.gz"

POCKET_PREDICTION_MODEL_FEATURES = [
    'vina_rmsd_lb', 
    'number_fpocket_poses_at_location', 
    'number_vina_poses_at_location', 
    'number_galaxydock_poses_at_location', 
    'vina_closest_pose_id', 
    'vina_distance', 
    'galaxydock_closest_pose_id', 
    'galaxydock_score', 
    'galaxydock_autodock', 
    'galaxydock_internal_energy', 
    'plants_distance', 
    'plants_total_score',
    'plants_score_rb_pen',
    'plants_score_norm_hevatoms', 
    'plants_score_norm_weight', 
    'plants_score_rb_pen_norm_crt_hevatoms', 
    'zdock_distance', 
    'zdock_score', 
    'p2rank_closest_pose_id', 
    'p2rank_distance', 
    'p2rank_pocket_score', 
    'p2rank_sas_points', 
    'p2rank_surf_atoms', 
    'fpocket_closest_pose_id', 
    'fpocket_pocket_score', 
    'fpocket_number_of_alpha_spheres', 
    'fpocket_mean_alpha-sphere_radius', 
    'fpocket_hydrophobicity_score', 
    'fpocket_polarity_score', 
    'fpocket_amino_acid_based_volume_score', 
    'fpocket_pocket_volume_(convex_hull)', 
    'fpocket_charge_score', 
    'fpocket_local_hydrophobic_density_score', 
    'fpocket_number_of_apolar_alpha_sphere',
]

def load_model(
    model_filename: str = POCKET_PREDICTION_MODEL_FILENAME,
    verbose: bool = True,
    ):
    if verbose:
        print ("Loading COBDock model from", model_filename)
    return load_compressed_pickle(model_filename, verbose=verbose)

def prepare_for_model_prediction(
    dataframe,
    verbose: bool = True,
    ):

    if verbose:
        print ("Preparing data of shape", dataframe.shape, "for ML")

    return dataframe[POCKET_PREDICTION_MODEL_FEATURES]
