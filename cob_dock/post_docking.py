if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os, glob

import numpy as np
import pandas as pd 
from scipy.stats import rankdata

from utils.io.io_utils import load_json, write_json, load_compressed_pickle

from ai_blind_docking.algorithms.cob_dock.model_utils import load_chunked_ai_docking_data

from ai_blind_docking.data_preparation import DATA_ROOT_DIR

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.molecules.pdb_utils import identify_centre_of_mass, get_bounding_box_size
from utils.molecules.pymol_utils import create_complex_with_pymol
from ai_blind_docking.docking_utils.vina_utils import prepare_ligand_for_vina, prepare_target_for_vina, execute_vina, convert_and_separate_vina_out_file


def compute_mean_scores_for_ligand_target_pairs(
    all_ligand_target_pose_scores: dict,
    verbose: bool = True,
    ):

    if verbose:
        print ("Computing mean prediction scores for", len(all_ligand_target_pose_scores), "voxel(s)")

    mean_scores = {} 

    for ligand_target_pose, prediction in all_ligand_target_pose_scores.items():
        ligand_target = "_".join(ligand_target_pose.split("_")[:-1])
        if ligand_target not in mean_scores:
            mean_scores[ligand_target] = []
        mean_scores[ligand_target].append(prediction)
    # compute means 
    for ligand_target in mean_scores:
        mean_scores[ligand_target] = np.mean(mean_scores[ligand_target])

    return mean_scores
    

def compute_minimum_scores_for_ligand_target_pairs(
    all_ligand_target_pose_scores: dict,
    verbose: bool = True,
    ):
    if verbose:
        print ("Computing minimum prediction scores for", len(all_ligand_target_pose_scores), "voxel(s)")

    minimum_ranks = {}
    for ligand_target_pose, score in all_ligand_target_pose_scores.items():
        ligand_target = "_".join(ligand_target_pose.split("_")[:-1])
        if ligand_target not in minimum_ranks:
            minimum_ranks[ligand_target] = np.inf 
        if score < minimum_ranks[ligand_target]:
            minimum_ranks[ligand_target] = score
    return minimum_ranks

def compute_ML_based_binding_predictions(
    pair_csv_files_output_dir: str,
    output_dir: str,
    model_dir: str = os.path.join("ai_blind_docking", "binding_prediction_models", ),
    aggregation_method: str = "mean",
    max_distance: float = 10,
    invert_columns: bool = False, # not required for mean
    model_name: str = "hist_gradient_boosting_classifier",
    verbose: bool = True,
    ):
    os.makedirs(output_dir, exist_ok=True)

    model_full_path = os.path.join(model_dir, f"{model_name}_{aggregation_method}_{max_distance}_{invert_columns}.pkl.gz")
    if not os.path.exists(model_full_path):
        print(model_full_path, "does not exist")
        return None 

    if not os.path.isdir(pair_csv_files_output_dir):
        print (pair_csv_files_output_dir, "does not exist")
        return None

    # input_data_filenames = sorted(glob.iglob(os.path.join(pair_csv_files_output_dir, "*.csv")))
    input_data_filenames = sorted(glob.iglob(os.path.join(pair_csv_files_output_dir, "*.parquet")))

    # get list of pairs
    pairs = [
        os.path.splitext(os.path.basename(input_csv_filename))[0]
        for input_csv_filename in input_data_filenames
    ]

    # aggregate data
    aggregated_data = load_chunked_ai_docking_data(
        input_filenames=input_data_filenames,
        output_filename=None, # do not load/save
        aggregation_method=aggregation_method,
        max_distance=max_distance,
        invert_columns=invert_columns,
        verbose=verbose,
    )

    # load the model from disk
    loaded_model = load_compressed_pickle(model_full_path, verbose=verbose)
    # perform ML binary prediction
    y_pred = loaded_model.predict(aggregated_data)
    # probability prediction
    y_prob = loaded_model.predict_proba(aggregated_data)[:, loaded_model.classes_==1].flatten()

    # remove from memory
    del loaded_model

    ml_predictions = {
        pair: {
            "pair": pair,
            "prediction": prediction,
            "probability": probability
        }
        for pair, prediction, probability in zip(pairs, y_pred, y_prob)
    }

    ml_predictions_filename = os.path.join(output_dir, "ml_predictions.json")
    write_json(ml_predictions, ml_predictions_filename, verbose=verbose)
    
    # compute mean ML scores 
    # mean_ml_scores = compute_mean_scores_for_ligand_target_pairs(ml_predictions)

    # write_json(mean_ml_scores, os.path.join(output_dir, "mean_ml_scores.json"))

    return ml_predictions

def build_rank_aggregation_input(
    data_df: pd.DataFrame,
    weighted: bool = True,
    verbose: bool = True,
    ):
    if verbose:
        print ("Building rank aggregation input")
    if weighted:
        column_selection = \
            ["vina_distance"] * 3 + \
            ["vina_energy"] * 271 + \
            ["vina_rmsd_ub"] * 74 + \
            ["plants_distance"] * 26 + \
            ["plants_total_score"] * 195 + \
            ["plants_rmsd"] * 7 + \
            ["zdock_distance"] * 19 + \
            ["zdock_score"] * 453 + \
            ["zdock_rmsd"] * 37 + \
            ["natural_ligand_distance"] * 2 + \
            ["galaxydock_score"] * 5 + \
            ["galaxydock_rmsd"] * 35 + \
            ["galaxydock_autodock"] * 80 + \
            ["galaxydock_drug_score"] * 62 + \
            ["galaxydock_internal_energy"] * 3 + \
            ["p2rank_distance"] * 265 + \
            ["p2rank_pocket_score"] * 111 + \
            ["p2rank_sas_points"] * 4 + \
            ["p2rank_surf_atoms"] * 2 + \
            ["fpocket_distance"] * 10 + \
            ["fpocket_pocket_score"] * 353 + \
            ["fpocket_drug_score"] * 277
        
        # write_json(column_selection, "weighted_column_selection.json")
        for col in column_selection:
            assert col in data_df.columns, f"WEIGHTED INPUT FAIL {col}"
        
        data_df = data_df[column_selection]


    data_df = data_df.T # transpose to iterate over columns
    # data_df.to_csv("duplicated_data.csv")

    rank_aggregation_input = []
    for _, row in data_df.iterrows():
        rank_aggregation_input.append(row.to_dict())
    return rank_aggregation_input

def compute_minimum_ligand_target_pair_ranks_for_expert(
    ligands, # may be missing for pocket analysis programs so must be supplied outside of function 
    expert_data,
    expert_score_field, # i.e: 'energy' for Vina
    is_docking, # switch for whether or not to consider ligands
    bigger_is_better, # boolean to determine ranking
    rank_method="dense",
    ):
    # also return best scores

    # record scores
    # best depends on bigger_is_better
    ligand_target_best_expert_scores = {}

    for ligand in ligands:
            
        if not is_docking: # pocket analysis program do not require a ligand
            expert_data_ligand = expert_data
        else:
            expert_data_ligand = expert_data[ligand]

        for accession in expert_data_ligand:
            expert_data_ligand_accession = expert_data_ligand[accession]
    
            for target in expert_data_ligand_accession:
                
                expert_data_ligand_accession_target = expert_data_ligand_accession[target]

                ligand_target_best_expert_score = np.nan # initial value

                for pose in expert_data_ligand_accession_target:

                    # get score for pase based on `expert_score_field`
                    ligand_target_pose_expert_score = expert_data_ligand_accession_target[pose][expert_score_field]
                    if bigger_is_better:
                        # now smaller is better
                        ligand_target_pose_expert_score = -ligand_target_pose_expert_score

                    # currently ignoring accession
                    # pair_best_expert_scores[f"{ligand}_{target}_{pose}"] = ligand_target_pose_expert_score
                    
                    # update best score
                    if np.isnan(ligand_target_best_expert_score) or ligand_target_pose_expert_score < ligand_target_best_expert_score:
                        ligand_target_best_expert_score = ligand_target_pose_expert_score

                ligand_target_best_expert_scores[f"{ligand}_{target}"] = ligand_target_best_expert_score
                

    # begin ranking 
    # smaller is always better
    all_pairs = sorted(ligand_target_best_expert_scores)
    scores = np.array([ligand_target_best_expert_scores[pair] for pair in all_pairs])
    ranks = rankdata(scores, method=rank_method).astype(float)

    # set nan rank to nan
    ranks[np.isnan(scores)] = np.nan
    # convert to dict
    minimum_ligand_target_pair_ranks = {
        pair: rank 
        for pair, rank in zip(all_pairs, ranks)
    }

    # minimum_ligand_target_pair_ranks = pd.Series(ligand_target_best_expert_scores).rank(method=rank_method).to_dict()

    # minimum_ligand_target_pair_ranks = compute_minimum_scores_for_ligand_target_pairs(ligand_target_pose_ranks)


    return ligand_target_best_expert_scores, minimum_ligand_target_pair_ranks

def compute_minimum_ranks_for_all_experts(
    ligands_to_targets,
    fpocket_data_filename,
    p2rank_data_filename,
    vina_data_filename,
    galaxydock_data_filename,
    plants_data_filename,
    zdock_data_filename,
    verbose: bool = True,
    ):
    if verbose:
        print ("Computing minimum ranks for all experts")

    # get list of ligands 
    ligands = sorted(ligands_to_targets)


    ## build dict of expert -> ligand_target pair -> minimum rank 
    pair_minimum_scores_and_ranks = {}

    experts = {
        "vina": {
            "data_filename": vina_data_filename,
            "score_field": "energy",
            "bigger_is_better": False,
            "is_docking": True,
        },
        "galaxydock": {
            "data_filename": galaxydock_data_filename,
            "score_field": "score",
            "bigger_is_better": False,
            "is_docking": True,
        },
        "plants": {
            "data_filename": plants_data_filename,
            "score_field": "total_score",
            "bigger_is_better": False,
            "is_docking": True,
        },
        "zdock": {
            "data_filename": zdock_data_filename,
            "score_field": "score",
            "bigger_is_better": True,
            "is_docking": True,
        },
        "p2rank": {
            "data_filename": p2rank_data_filename,
            "score_field": "pocket_score",
            "bigger_is_better": True,
            "is_docking": False,
        },
        "fpocket": {
            "data_filename": fpocket_data_filename,
            "score_field": "pocket_score",
            "bigger_is_better": True,
            "is_docking": False,
        },
    }

    for expert_name, expert_data in experts.items():
        expert_data_filename = expert_data["data_filename"]
        expert_score_field = expert_data["score_field"]
        expert_bigger_is_better = expert_data["bigger_is_better"]
        expert_is_docking = expert_data["is_docking"]
        expert_key = f"{expert_name}_{expert_score_field}"

        # load the data
        expert_data = load_json(expert_data_filename)
        pair_minimum_scores, pair_minimum_ranks = compute_minimum_ligand_target_pair_ranks_for_expert(
            ligands=ligands,
            expert_data=expert_data,
            expert_score_field=expert_score_field,
            is_docking=expert_is_docking,
            bigger_is_better=expert_bigger_is_better,
        )
        # update return dict
        pair_minimum_scores_and_ranks[expert_key] = {
            "score": pair_minimum_scores,
            "rank": pair_minimum_ranks,
        }

    print ("Completed computing pair minimum ranks for each expert")

    return pair_minimum_scores_and_ranks

def execute_post_docking_old(
    ligands_to_targets: dict,
    pair_csv_files_output_dir: str,
    output_dir: str,
    fpocket_data_filename: str,
    p2rank_data_filename: str,
    vina_data_filename: str,
    galaxydock_data_filename: str,
    plants_data_filename: str,
    zdock_data_filename: str,
    verbose: bool = True,
    ):

    if verbose:
        print ("Beginning scoring of pockets with machine learning")

    # compute ML scores 
    pair_binding_predictions = compute_ML_based_binding_predictions(
        pair_csv_files_output_dir=pair_csv_files_output_dir,
        output_dir=output_dir,
        aggregation_method="mean",
        max_distance=10,
        invert_columns=False,
        model_name="hist_gradient_boosting_classifier",
    )

    # sort pairs by probability
    ligand_target_pairs_sorted_by_probability = sorted(
        pair_binding_predictions, 
        key=lambda pair: pair_binding_predictions[pair]["probability"], 
        reverse=True)

    # print ("READING CONSENSUS RANKING DATA FROM", consensus_ranking_filename)
    # consensus_ranking_input_df = pd.read_csv(
    #     consensus_ranking_filename,
    #     index_col=0)

    # weighted_rank_aggregation_input = build_rank_aggregation_input(
    #     consensus_ranking_input_df,
    #     weighted=True)
    # unweighted_rank_aggregation_input = build_rank_aggregation_input(
    #     consensus_ranking_input_df,
    #     weighted=False)

    # # write list aggregation to file for debugging
    # # write_json(weighted_rank_aggregation_input, os.path.join(output_dir, "weighted_rank_aggregation_input.json"))
    # # write_json(unweighted_rank_aggregation_input, os.path.join(output_dir, "unweighted_rank_aggregation_input.json"))

    # weighted_reversed_ranking = perform_rank_aggregation(weighted_rank_aggregation_input)
    # write_json(weighted_reversed_ranking, os.path.join(output_dir, "weighted_reversed_ranking.json"))
    
    # unweighted_reversed_ranking = perform_rank_aggregation(unweighted_rank_aggregation_input)
    # write_json(unweighted_reversed_ranking, os.path.join(output_dir, "unweighted_reversed_ranking.json"))

     # sorted poses by rank
    # sorted_poses_by_rank_weighted = sorted(
    #     weighted_reversed_ranking, 
    #     key=weighted_reversed_ranking.get, 
    #     reverse=True) # biggest first
    # weighted_true_ranking = {
    #     ligand_target_pose: i+1 
    #         for i, ligand_target_pose in enumerate(sorted_poses_by_rank_weighted)
    # }

    # sorted_poses_by_rank_unweighted = sorted(
    #     unweighted_reversed_ranking, 
    #     key=unweighted_reversed_ranking.get, 
    #     reverse=True) # biggest first
    # unweighted_true_ranking = {
    #     ligand_target_pose: i+1 
    #         for i, ligand_target_pose in enumerate(sorted_poses_by_rank_unweighted)
    # }

    # write_json(weighted_true_ranking, os.path.join(output_dir, "weighted_true_ranking.json"))
    # write_json(unweighted_true_ranking, os.path.join(output_dir,"unweighted_true_ranking.json"))

    # weighted_minimum_true_ranks = compute_minimum_scores_for_ligand_target_pairs(
    #     weighted_true_ranking)
    # write_json(weighted_minimum_true_ranks, os.path.join(output_dir,"weighted_minimum_true_ranks.json"))

    # unweighted_minimum_true_ranks = compute_minimum_scores_for_ligand_target_pairs(
    #     unweighted_true_ranking)
    # write_json(unweighted_minimum_true_ranks, os.path.join(output_dir,"unweighted_minimum_true_ranks.json"))


    # ligand_target_sorted_by_weighted_minimum_true_ranks = sorted(
    #     weighted_minimum_true_ranks, 
    #     key=weighted_minimum_true_ranks.get) # smallest to biggest

    # compute minimum ranks for each expert
    minimum_scores_and_ranks_for_each_expert = compute_minimum_ranks_for_all_experts(
        ligands_to_targets=ligands_to_targets,
        fpocket_data_filename=fpocket_data_filename,
        p2rank_data_filename=p2rank_data_filename,
        vina_data_filename=vina_data_filename,
        galaxydock_data_filename=galaxydock_data_filename,
        plants_data_filename=plants_data_filename,
        zdock_data_filename=zdock_data_filename,
    )

    write_json(minimum_scores_and_ranks_for_each_expert, 
        os.path.join(output_dir, "minimum_scores_and_ranks_for_each_expert.json"))

    # put it all together
    # smaller is better
    # ligand_target_sorted_by_consensus_ranking = sorted(consensus_ranking, key=consensus_ranking.get)

    # aggregate by accession?


    combined_output = [
        {
            "ligand_target_pair": ligand_target_pair,
            "predicted_rank": predicted_rank + 1, # start from 1
            # "weighted_minimum_rank": weighted_minimum_true_ranks[ligand_target],
            "binding_probability": pair_binding_predictions[ligand_target_pair]["probability"],
            "predicted_to_bind": pair_binding_predictions[ligand_target_pair]["prediction"],
            # "unweighted_minimum_rank": unweighted_minimum_true_ranks[ligand_target],
            # **{
            #     f"{expert}_minimum_rank": minimum_ranks_for_each_expert[expert][ligand_target]
            #     for expert in minimum_ranks_for_each_expert
            # },
            # **{
            #     f"{expert}_minimum_score": minimum_scores_for_each_expert[expert][ligand_target]
            #     for expert in minimum_scores_for_each_expert
            # },
            # unpack best expert scores and ranks
            **{
                f"best_{expert}_{score_or_rank}": minimum_scores_and_ranks_for_each_expert[expert][score_or_rank][ligand_target_pair]
                for expert in minimum_scores_and_ranks_for_each_expert
                for score_or_rank in minimum_scores_and_ranks_for_each_expert[expert]
            }
        }
        # for ligand_target in ligand_target_sorted_by_consensus_ranking
        for predicted_rank, ligand_target_pair in enumerate(ligand_target_pairs_sorted_by_probability)
    ]
    write_json(combined_output, os.path.join(output_dir, "combined_output.json"))

    # finally, convert to csv
    # combined_output_df = pd.DataFrame(combined_output, index=ligand_target_sorted_by_consensus_ranking)
    # skip csv now
    # combined_output_df = pd.DataFrame(combined_output, index=ligand_target_pairs_sorted_by_probability)
    # combined_output_df_filename = os.path.join(output_dir, "combined_output.csv")
    # combined_output_df.to_csv(combined_output_df_filename)

    return combined_output

def generate_all_local_docking_tasks(
    executor,
    ligands_to_targets: dict,
    output_directory: str,
    num_poses: int,
    verbose: bool = True,
    ):

    running_tasks = {}
        
    for ligand_id, ligand_data in ligands_to_targets.items():

        # make output for ligand
        ligand_output_dir = os.path.join(output_directory, ligand_id)
        os.makedirs(ligand_output_dir, exist_ok=True)
        
        ligand_pdb_filename = ligand_data["pdb_filename"]
        prepared_ligand_filename_vina = os.path.join(
            ligand_output_dir,
            f"{ligand_id}.pdbqt"
        )

        # prepare ligand for docking
        if not os.path.exists(prepared_ligand_filename_vina):
            prepared_ligand_filename_vina = prepare_ligand_for_vina(
                ligand_filename=ligand_pdb_filename,
                output_filename=prepared_ligand_filename_vina,
                overwrite=False,
                verbose=verbose,
            )

        if prepared_ligand_filename_vina is None or not os.path.exists(prepared_ligand_filename_vina):
            continue

        ligand_accessions = ligands_to_targets[ligand_id]["prepared_targets"]

        for accession in ligand_accessions:

            # get set of PDB targets for current accession
            pdb_ids = ligand_accessions[accession]

            # iterate over PDB targets
            for pdb_id in pdb_ids:

                ligand_target_output_dir = os.path.join(
                    ligand_output_dir,
                    accession,
                    pdb_id,
                )
                os.makedirs(ligand_target_output_dir, exist_ok=True)
                
                # handle on target data
                target_data = pdb_ids[pdb_id]

                # assume target has been prepared already
                prepared_target_filename = target_data["prepared_filename"]

                # located at root of vina_out)directory
                prepared_target_filename_vina = os.path.join(
                    output_directory, 
                    f"{pdb_id}.pdbqt")
                if not os.path.exists(prepared_target_filename_vina):
                    prepared_target_filename_vina = prepare_target_for_vina(
                        prepared_target_filename,
                        output_filename=prepared_target_filename_vina,
                        overwrite=False,
                        verbose=verbose,
                    )

                if prepared_target_filename_vina is None or not os.path.exists(prepared_target_filename_vina):
                    if verbose:
                        print ("Error in preparing target", pdb_id, "for Vina, skipping it")
                    continue

                # extract pockets 
                all_pocket_data = target_data["top_pockets"]

                for pocket_data in all_pocket_data:

                    pocket_num = pocket_data["pocket_rank"]

                    ligand_target_pocket_output_dir = os.path.join(
                        ligand_target_output_dir,
                        f"pocket_{pocket_num}",
                    )
                    os.makedirs(ligand_target_pocket_output_dir, exist_ok=True)

                    vina_output_filename = os.path.join(ligand_target_pocket_output_dir, f"{ligand_id}_{pdb_id}_{pocket_num}.pdbqt")
                    vina_log_filename = os.path.join(ligand_target_pocket_output_dir, f"{ligand_id}_{pdb_id}_{pocket_num}.log")
                    vina_log_json_filename = vina_log_filename + ".json"

                    
                    # submit task
                    task = executor.submit(
                        execute_vina,
                        ligand_filename=prepared_ligand_filename_vina,
                        target_filename=prepared_target_filename_vina,
                        output_filename=vina_output_filename,
                        log_filename=vina_log_filename,
                        center_x=pocket_data["center_x"],
                        center_y=pocket_data["center_y"],
                        center_z=pocket_data["center_z"],
                        size_x=max(pocket_data["size_x"], 23),
                        size_y=max(pocket_data["size_y"], 23),
                        size_z=max(pocket_data["size_z"], 23),
                        vina_variant="vina",
                        n_proc=1,
                        num_poses=num_poses,
                        verbose=verbose,
                    )

                    running_tasks[task] = {
                        "ligand_id": ligand_id,
                        "accession": accession,
                        "pdb_id": pdb_id,
                        "pocket_num": pocket_num,
                        "ligand_pdb_filename": ligand_pdb_filename,
                        "prepared_target_filename": prepared_target_filename,
                    }

    return running_tasks

def execute_all_final_local_docking(
    ligands_to_targets: dict,
    output_directory: str,
    num_poses: int,
    num_complexes: int,
    local_docking_program: str = "vina", # TODO
    n_proc: int = 5,
    verbose: bool = True,
    ):

    if verbose:
        print("Executing all local docking for", len(ligands_to_targets), "ligands using", n_proc, "process(es)")
        print ("Generating", num_poses, "pose(s)")
        print ("Generating", num_complexes, "complex(es)")
    
    # begin local docking execution

    with ProcessPoolExecutor(max_workers=n_proc) as p:

        # generate local docking tasks 
        running_tasks = generate_all_local_docking_tasks(
            executor=p,
            ligands_to_targets=ligands_to_targets,
            output_directory=output_directory,
            num_poses=num_poses,
            verbose=verbose,
        )

        # begin collation of docking data
        all_local_docking_collated_data = {}

        for running_task in as_completed(running_tasks):

            task_data = running_tasks[running_task]

            ligand_id = task_data["ligand_id"]
            accession = task_data["accession"]
            pdb_id = task_data["pdb_id"]
            pocket_num = task_data["pocket_num"]
            # ligand_pdb_filename = task_data["ligand_pdb_filename"]
            prepared_target_filename = task_data["prepared_target_filename"]

            # prepare all_local_docking_collated_data
            if ligand_id not in all_local_docking_collated_data:
                all_local_docking_collated_data[ligand_id] = {}
            if accession not in all_local_docking_collated_data[ligand_id]:
                all_local_docking_collated_data[ligand_id][accession] = {}
            if pdb_id not in all_local_docking_collated_data[ligand_id][accession]:
                all_local_docking_collated_data[ligand_id][accession][pdb_id] = {}
            if pocket_num not in all_local_docking_collated_data[ligand_id][accession][pdb_id]:
                all_local_docking_collated_data[ligand_id][accession][pdb_id][pocket_num] = {}

            output_filename, log_json_filename = running_task.result()

            del running_tasks[running_task]

            # begin collation of poses
            # check that Vina ran for target
            if not os.path.exists(log_json_filename) or not os.path.exists(output_filename):
                continue

             # load log file as JSON
            vina_pose_data = load_json(log_json_filename, verbose=verbose,)

            ligand_target_pocket_num_output_dir = os.path.join(
                output_directory,
                ligand_id,
                accession,
                pdb_id,
                f"pocket_{pocket_num}"
            )

            # create pose and complex dir
            pose_output_dir = os.path.join(
                ligand_target_pocket_num_output_dir,
                "poses",
            )
            os.makedirs(pose_output_dir, exist_ok=True)

            complex_output_dir = os.path.join(
                ligand_target_pocket_num_output_dir,
                "complexes",
            )
            os.makedirs(complex_output_dir, exist_ok=True)

            # split out file into pose_pdb_files
            pose_pdb_files = convert_and_separate_vina_out_file(
                vina_output_filename=output_filename,
                conversion_dir=pose_output_dir,
                ligand_id=ligand_id,
                output_format="pdb",
                verbose=verbose,
            )

            for pose_pdb_filename in pose_pdb_files:
                    
                pose_pdb_basename = os.path.basename(pose_pdb_filename)
                pose_pdb_basename = os.path.splitext(pose_pdb_basename)[0]
                pose_id = pose_pdb_basename.split("_")[1]

                pose_data = vina_pose_data[pose_id]

                if num_complexes is not None and int(pose_id) <= num_complexes:

                    # make complex
                    complex_filename = os.path.join(
                        complex_output_dir,
                        f"complex_{pose_id}.pdb")
                    create_complex_with_pymol(
                        input_pdb_files=[prepared_target_filename, pose_pdb_filename],
                        output_pdb_filename=complex_filename,
                        verbose=verbose,
                    )

                center_of_mass = identify_centre_of_mass(pose_pdb_filename, verbose=verbose)
                if center_of_mass is None:
                    continue
                center_x, center_y, center_z = center_of_mass
                size_x, size_y, size_z = get_bounding_box_size(pose_pdb_filename, verbose=verbose)

                pose_data = {
                    "center_x": center_x, 
                    "center_y": center_y, 
                    "center_z": center_z, 
                    "size_x": size_x, 
                    "size_y": size_y, 
                    "size_z": size_z, 
                    **{
                        k: float(pose_data[k])
                        for k in (
                            "energy",
                            "rmsd_lb",
                            "rmsd_ub",
                        )
                    }
                }

                all_local_docking_collated_data[ligand_id][accession][pdb_id][pocket_num][pose_id] = pose_data

    return all_local_docking_collated_data

def execute_post_docking(
    ligands_to_targets: dict,
    # all_pair_data_filenames: str,
    output_dir: str,
    # fpocket_data_filename: str,
    # p2rank_data_filename: str,
    # vina_data_filename: str,
    # galaxydock_data_filename: str,
    # plants_data_filename: str,
    # zdock_data_filename: str,
    num_top_pockets: int = 5,
    num_poses: int = 10,
    num_complexes: int = 1,
    top_pocket_distance_threshold: float = 3,
    local_docking_n_proc: int = 5, 
    verbose: bool = True,
    ):

    os.makedirs(output_dir, exist_ok=True,)

    # load model 
    from ai_blind_docking.algorithms.cob_dock.model_utils import POCKET_PREDICTION_MODEL_FEATURES, POCKET_PREDICTION_MODEL_FILENAME

    if verbose:
        print ("Beginning scoring of pockets with machine learning")
        print ("Loading prediction model from", POCKET_PREDICTION_MODEL_FILENAME)
        print ("Predicting using", len(POCKET_PREDICTION_MODEL_FEATURES), "feature(s)")
        print ("Idenifying top", num_top_pockets, "pocket(s) and writing to directory", output_dir)

    model = load_compressed_pickle(POCKET_PREDICTION_MODEL_FILENAME, default=None, verbose=verbose)
    assert model is not None 

    # which column to select probabilities for 
    label_index = model.classes_ == 1

    # for pair_id, data_filename in all_pair_data_filenames.items():
    for ligand_id, ligand_data in ligands_to_targets.items():

        ligand_accessions = ligand_data["prepared_targets"]

        for accession, accession_pdb_ids in ligand_accessions.items():
            for pdb_id, pdb_id_data in accession_pdb_ids.items():

                if "collated_data_filename" not in pdb_id_data:
                    continue
                collated_data_filename = pdb_id_data["collated_data_filename"]

                if verbose:
                    print ("Loading collated data from", collated_data_filename)

                if collated_data_filename.endswith(".parquet"):
                    collated_data = pd.read_parquet(collated_data_filename)
                elif collated_data_filename.endswith(".csv") or collated_data_filename.endswith(".csv.gz"):
                    collated_data = pd.read_csv(collated_data_filename)
                else:
                    raise NotImplementedError

                # make predictions using POCKET_PREDICTION_MODEL_FEATURES
                collated_data["voxel_pocket_probability"] = model.predict_proba(collated_data[POCKET_PREDICTION_MODEL_FEATURES])[:, label_index]
                collated_data["voxel_rank"] = collated_data["voxel_pocket_probability"].rank(ascending=False, method="dense").astype(int)

                # sort voxels by rank
                collated_data = collated_data.sort_values("voxel_rank", ascending=True)

                # write the data for all voxels back to file
                if collated_data_filename.endswith(".parquet"):
                    collated_data.to_parquet(collated_data_filename)
                elif collated_data_filename.endswith(".csv") or collated_data_filename.endswith(".csv.gz"):
                    collated_data.to_csv(collated_data_filename)
                else:
                    raise NotImplementedError

                # iterate over rows and extract top N locations 
                top_pockets = []

                for pocket_num in range(num_top_pockets):

                    if collated_data.shape[0] == 0:
                        break

                    # take first row
                    best_voxel = collated_data.iloc[0]

                    # pocket data
                    selected_pocket_program = best_voxel["selected_pocket_program"]
                    selected_pocket_id = best_voxel["selected_pocket_closest_pose_id"]

                    top_pockets.append({
                        "pocket_rank": pocket_num + 1,
                        "pocket_score": best_voxel["voxel_pocket_probability"],
                        "selected_pocket_program": selected_pocket_program,
                        "selected_pocket_id": selected_pocket_id,
                        **{
                            k: best_voxel[f"selected_pocket_{k}"]
                            for k in (
                                "center_x",
                                "center_y",
                                "center_z",
                                "size_x",
                                "size_y",
                                "size_z",
                            )
                        }
                    })

                    # remove all voxels mapped to this location (alternatively remove all voxels close to this location?)
                    # select using exact pocket id
                    # collated_data = collated_data.loc[(collated_data["selected_pocket_program"]!=selected_pocket_program) | (collated_data["selected_pocket_closest_pose_id"]!=selected_pocket_id)]
                    # select using distance 
                    selected_pocket_location = np.array([ 
                        best_voxel[f"selected_pocket_center_{k}"]
                        for k in ("x", "y", "z")
                    ])
                    distances_to_selected_pocket = np.linalg.norm(
                        selected_pocket_location - collated_data[["selected_pocket_center_x", "selected_pocket_center_y", "selected_pocket_center_z", ]],
                        axis=-1)
                    collated_data = collated_data.loc[distances_to_selected_pocket > top_pocket_distance_threshold] 

                # add "top_pockets" key to pdb_id data
                pdb_id_data["top_pockets"] = top_pockets

    # remove model from memory
    del model

    # write top_pockets to file
    all_ligand_top_pocket_data = {
        ligand_id: {
            accession: {
                pdb_id: pdb_id_data["top_pockets"]
                for pdb_id, pdb_id_data in accession_data.items()
            }
            for accession, accession_data in ligand_data["prepared_targets"].items()
        }
        for ligand_id, ligand_data in ligands_to_targets.items()
    }

    all_ligand_top_pocket_data_filename = os.path.join(output_dir, f"top_{num_top_pockets}_pockets.json")  
    write_json(all_ligand_top_pocket_data, all_ligand_top_pocket_data_filename, verbose=verbose) 

    # write_json(ligands_to_targets, "ligands_to_targets", verbose=verbose) 
    # raise Exception

    # begin execution of local docking
    local_docking_output_dir = os.path.join(output_dir, "local_docking")
    os.makedirs(local_docking_output_dir, exist_ok=True,)

    all_local_docking_collated_data = execute_all_final_local_docking(
        ligands_to_targets=ligands_to_targets,
        output_directory=local_docking_output_dir,
        num_poses=num_poses,
        num_complexes=num_complexes,
        n_proc=local_docking_n_proc,
        verbose=verbose,
    )

    # write all_local_docking_collated_data to file 
    all_local_docking_collated_data_filename = os.path.join(output_dir, "pose_data.json")
    write_json(all_local_docking_collated_data, all_local_docking_collated_data_filename, verbose=verbose)

    return all_local_docking_collated_data


if __name__== '__main__':

    pass