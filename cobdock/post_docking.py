if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os, glob

import numpy as np
import pandas as pd 

from utils.io.io_utils import load_json, write_json, load_compressed_pickle

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.molecules.pdb_utils import identify_centre_of_mass, get_bounding_box_size
from utils.molecules.pymol_utils import create_complex_with_pymol

# model prediction
from cobdock.model_utils import load_model, prepare_for_model_prediction

# local docking 
from cobdock.docking.vina.vina_utils import prepare_ligand_for_vina, prepare_target_for_vina, execute_vina, convert_and_separate_vina_out_file

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
                    # vina_log_json_filename = vina_log_filename + ".json"
                    
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

            output_filename_log_json_filename = running_task.result()

            del running_tasks[running_task]
            
            # handle missing vina / task fail
            if output_filename_log_json_filename is None:
                continue

            output_filename, log_json_filename = output_filename_log_json_filename

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
    output_dir: str,
    num_top_pockets: int = 5,
    num_poses: int = 10,
    num_complexes: int = 1,

    commercial_use_only: bool = False,
    
    top_pocket_distance_threshold: float = 3,
    local_docking_program: str = "vina",
    local_docking_n_proc: int = 5, 
    verbose: bool = True,
    ):

    os.makedirs(output_dir, exist_ok=True,)

    if verbose:
        print ("Beginning scoring of pockets with machine learning")
        print ("Idenifying top", num_top_pockets, "pocket(s) and writing to directory", output_dir)

    # load model 
    model = load_model(
        commercial_use_only=commercial_use_only,
        verbose=verbose,
    )
    assert model is not None 

    # name of positive class
    positive_class = model.positive_class

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
                ml_input_data = prepare_for_model_prediction(collated_data, commercial_use_only=commercial_use_only, verbose=verbose)

                # compute probabilities
                probs =  model.predict_proba(ml_input_data)
                
                # store predicted probability for each voxel
                collated_data["voxel_pocket_probability"] = probs[positive_class]

                # rank voxels by probability
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
                    selected_pocket_id = best_voxel["selected_pocket_min_pose_id"]

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

                    # remove all voxels within top_pocket_distance_threshold Angstrom
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

    # begin execution of local docking
    local_docking_output_dir = os.path.join(output_dir, "local_docking")
    os.makedirs(local_docking_output_dir, exist_ok=True,)

    all_local_docking_collated_data = execute_all_final_local_docking(
        ligands_to_targets=ligands_to_targets,
        output_directory=local_docking_output_dir,
        num_poses=num_poses,
        num_complexes=num_complexes,
        local_docking_program=local_docking_program,
        n_proc=local_docking_n_proc,
        verbose=verbose,
    )

    # write all_local_docking_collated_data to file 
    all_local_docking_collated_data_filename = os.path.join(output_dir, "pose_data.json")
    write_json(all_local_docking_collated_data, all_local_docking_collated_data_filename, verbose=verbose)

    return all_local_docking_collated_data

if __name__== '__main__':

    pass