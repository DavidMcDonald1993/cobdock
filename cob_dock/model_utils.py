
if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import os, glob

from functools import partial

import numpy as np
import pandas as pd

from utils.io.io_utils import load_compressed_pickle

# TODO
POCKET_PREDICTION_MODEL_FILENAME = "ai_blind_docking/algorithms/cob_dock/pocket_prediction_models/model.pkl.gz"
# LATEST features
# for use with model_final.sav
# TODO, relabel with new column names 
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


def mean_distance_to_pose(
    row,
    ):

    distance_columns = ( 
        col for col in row.index
        if "distance" in col
    )

    dists = []
    for distance_column in distance_columns:
        pose_dist = row[distance_column]
        if pd.isnull(pose_dist):
            continue
        dists.append(pose_dist)

    return np.mean(dists)

def box_contains_pose(
    row,
    ):

    contains_pose_columns = ( 
        col for col
        in row.index
        if col.startswith("sampled_pose_number")
        and "natural_ligand" not in col
    )

    for contains_pose_column in contains_pose_columns:
        if row[contains_pose_column] > 0:
            return True

    return False

def box_close_to_pose(
    row,
    max_distance=15,
    distance_columns: list = None,
    ):
    if not max_distance: # return true if max_distance is None
        return True 

    if distance_columns is None:
        distance_columns = [
            col for col in row.index
            if "distance" in col 
            and "natural_ligand" not in col
        ]
    assert len(distance_columns) > 0, distance_columns

    for distance_column in distance_columns:
        if row[distance_column] <= max_distance:
            return True

    return False

def aggregate_boxes(
    pair_boxes: pd.DataFrame,
    aggregation_method: str,
    invert_columns: bool,
    location_ranking_model = None,
    verbose: bool = True,
    ):

    if aggregation_method == "mean":
        return pair_boxes.mean(axis=0)

    elif aggregation_method == "max":
        return pair_boxes.max(axis=0)

    elif aggregation_method == "min":
        return pair_boxes.min(axis=0)

    elif aggregation_method == "std":
        return pair_boxes.std(axis=0)

    elif aggregation_method == "all":
        pair_maxes = pair_boxes.max(axis=0).values
        pair_mins = pair_boxes.min(axis=0).values
        pair_means = pair_boxes.mean(axis=0).values
        pair_stds = pair_boxes.std(axis=0).values

        pair_all_summary_statistics = np.hstack([
            pair_maxes,
            pair_mins,
            pair_means,
            pair_stds,
        ])

        return pair_all_summary_statistics

    elif aggregation_method == "mean_dist": # distance is currently negative for `bigger is better`
        box_mean_dists = pair_boxes.apply(mean_distance_to_pose, axis=1)
        selected_box_idx = box_mean_dists.argmax()

        # add selected row
        return pair_boxes.iloc[selected_box_idx]

    elif "concat" in aggregation_method:
        box_mean_dists = pair_boxes.apply(mean_distance_to_pose, axis=1)
        box_mean_dists = box_mean_dists.to_dict()

        k = int(aggregation_method.split("concat")[1])

        # sort boxes by mean distance to pose, considering all programs
        top_k_boxes = sorted(
            box_mean_dists, 
            key=box_mean_dists.get, 
            reverse=invert_columns, # largest to smallest (since distance is reversed)
            )[:k] 

        n_boxes = len(top_k_boxes)
        if n_boxes < k:
            top_k_boxes.extend( [top_k_boxes[-1]] * (k-n_boxes) )
        assert len(top_k_boxes) == k

        # concatenate top_k_boxes
        row = np.hstack(
            [
                pair_boxes.loc[box].values
                for box in top_k_boxes
            ]
        )
        assert len(row.shape) == 1, row.shape
        assert row.shape[0] == len(BINDING_PREDICTION_ML_MODEL_FEATURES) * k, row.shape[0]
        return row

    # use location ranking model
    elif aggregation_method == "model":

        # further filter based on pockets only (10A)
        valid_idx = pair_boxes.apply(
            partial(
                box_close_to_pose,
                max_distance=10,
                distance_columns=["fpocket_distance", "p2rank_distance"]
            ),
            axis=1,
        )
        if valid_idx.sum() > 0: #  for safety
            pair_boxes = pair_boxes.loc[valid_idx]


        # select pocket prediction features 
        pocket_prediction_features = pair_boxes[POCKET_PREDICTION_MODEL_FEATURES]

        probabilities = location_ranking_model.predict_proba(pocket_prediction_features)[:,location_ranking_model.classes_==1]
        probabilities = probabilities.flatten()


        selected_box_idx = probabilities.argmax()

        return pair_boxes.iloc[selected_box_idx]

    else:
        raise NotImplementedError(aggregation_method)



def filter_boxes_and_compute_aggregate_scores_for_pairs(
    df_filename,
    ml_model_features,
    max_distance=15,
    aggregation_method="mean",
    invert_columns=True,
    location_ranking_model = None,
    verbose: bool = True,
    ):

    all_pair_box_df = pd.read_csv(df_filename, index_col=0)

    if max_distance == "pose":
        box_filter_function = box_contains_pose
    else:
        box_filter_function = partial(
            box_close_to_pose, 
            max_distance=max_distance
        )

    valid_box_mask = all_pair_box_df.apply(
        box_filter_function,
        axis=1)

    # remove rows containing invalid boxes
    all_pair_box_df = all_pair_box_df.loc[valid_box_mask]

    # select relevent features 
    if ml_model_features is not None:
        all_pair_box_df = all_pair_box_df[ml_model_features]

    if invert_columns:

        # flip columns in FLIP_COLUMNS to make `bigger is better`
        columns_to_invert = [
            col 
            for col in COLUMNS_TO_INVERT
            if col in BINDING_PREDICTION_ML_MODEL_FEATURES
        ]
        all_pair_box_df[columns_to_invert] = -all_pair_box_df[columns_to_invert]


    # sort lexographically
    pairs = sorted({
        "_".join(index.split("_")[:-1])
        for index in all_pair_box_df.index
    })

    pairs_aggregated = []

    for pair in pairs:

        if verbose:
            print ("processing pair", pair, 
                "aggregation method:", aggregation_method, 
                "max distance:", max_distance,
                "invert columns:", invert_columns,
                )

        pair_boxes = all_pair_box_df.loc[all_pair_box_df.index.str.match(pair)]

        # remove duplicates (accession)
        pair_boxes = pair_boxes.loc[~pair_boxes.index.duplicated(keep="first")]

        # if an aggregation method is provided, then aggregate boxes
        if aggregation_method is not None:
            pair_boxes = aggregate_boxes(
                pair_boxes=pair_boxes,
                aggregation_method=aggregation_method,
                invert_columns=invert_columns,
                location_ranking_model=location_ranking_model,
                verbose=verbose,
            )
        # if isinstance(pair_boxes, pd.DataFrame):
        #     pair_boxes = pair_boxes.values
        # raise Exception(pair_boxes.shape)
        pairs_aggregated.append(pair_boxes)        
    
    if aggregation_method is not None:
        pairs_aggregated = pd.DataFrame(pairs_aggregated, index=pairs)
    else:
        pairs_aggregated = pd.concat(pairs_aggregated) 
    return pairs_aggregated


def load_chunked_ai_docking_data(
    output_filename,
    input_filenames,
    ml_model_features=BINDING_PREDICTION_ML_MODEL_FEATURES,
    max_distance=15,
    aggregation_method="max",
    invert_columns=True,
    location_ranking_model_path = None,
    verbose: bool = True,
    ):

    if output_filename is not None and os.path.exists(output_filename):
        if verbose:
            print ("loading data from", output_filename)
        # complete_data = np.load(output_filename)
        # load using pandas
        complete_data = pd.read_csv(output_filename, index_col=0)
    else:

        location_ranking_model = None
        # load model if used for location ranking 
        if aggregation_method == "model":
            location_ranking_model = load_compressed_pickle(location_ranking_model_path, )

        complete_data = []
        for input_csv_filename in input_filenames:
            chunk_pairs_aggregated = filter_boxes_and_compute_aggregate_scores_for_pairs(
                input_csv_filename,
                ml_model_features=ml_model_features,
                max_distance=max_distance,
                aggregation_method=aggregation_method,
                invert_columns=invert_columns,
                location_ranking_model=location_ranking_model,
                verbose=verbose,
                )

            # cols_are_inf = np.isinf(chunk_pairs_aggregated).any(axis=0)
            # if cols_are_inf.any():
            #     for col, col_is_inf in cols_are_inf.items():
            #         if col_is_inf:
            #             print (col)

            complete_data.append(chunk_pairs_aggregated)

        # concatenate from list of DFs to DF
        complete_data = pd.concat(complete_data)

        # remove duplicates
        complete_data = complete_data.loc[~complete_data.index.duplicated(keep="first")]

        if output_filename is not None:

            # write in .csv format
            print ("Writing csv file to", output_filename)
            complete_data.to_csv(output_filename)

            output_stem, _ = os.path.splitext(output_filename)

            # write first 1000 rows (for testing)
            n_head_rows = 1000
            n_head_rows_filename = output_stem + f"_first_{n_head_rows}_rows.csv"
            print ("Writing first", n_head_rows, "rows to", n_head_rows_filename)
            complete_data.head(n_head_rows).to_csv(n_head_rows_filename)

            # write in .npy format
            # take values (convert to numpy array)
            complete_data_values = complete_data.values
        
            numpy_output_filename = output_stem + ".npy"
            print ("saving values to", numpy_output_filename)
            np.save(numpy_output_filename, complete_data_values)

    print ("data shape is", complete_data.shape)
    # assert not np.isinf(complete_data).any(None)

    return complete_data
