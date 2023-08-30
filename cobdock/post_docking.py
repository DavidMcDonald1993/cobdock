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
from utils.molecules.openbabel_utils  import obabel_convert
from utils.io.io_utils import delete_file, delete_directory

from cobdock.docking.blind_docking_utils import (
    prepare_target_for_docking, 
    prepare_ligand_for_docking,
)

from cobdock.docking.vina.vina_utils import (
    VINA_VARIANTS,
    # prepare_ligand_for_vina, 
    # prepare_target_for_vina, 
    execute_vina, 
    convert_and_separate_vina_out_file,
)
from cobdock.docking.plants.plants_utils import (
    execute_plants,
)
from cobdock.docking.galaxydock.galaxydock_utils import (
    execute_galaxydock,
)

LOCAL_DOCKING_N_PROC = int(os.environ.get("LOCAL_DOCKING_N_PROC", default=5))

def execute_local_docking(
    ligand_id: str,
    ligand_3D_filename: str, 
    target_id: str,
    target_3D_filename: str,
    output_dir: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    docking_program: str = "vina",
    exhaustiveness: int = 8,
    num_poses: int = None,
    # num_complexes: int = None,
    # complex_prefix: str = None, # to better handle complex naming?
    n_proc: int = None,
    verbose: bool = True,
    ):

    docking_program = docking_program.lower()

    if verbose:
        print ("Performing docking using program", docking_program)
        print ("Ligand filename:", ligand_3D_filename)
        print ("Target filename:", target_3D_filename)
        print ("Outputting to:", output_dir)
        print ("Docking to location", center_x, center_y, center_z)
        print ("Docking using bounding box", size_x, size_y, size_z)

    os.makedirs(output_dir, exist_ok=True)

    pose_output_dir = os.path.join(output_dir, "poses")
    os.makedirs(pose_output_dir, exist_ok=True)

    # execute docking
    if verbose:
        print ("Outputting poses to", pose_output_dir)

    docking_program = docking_program.lower()

    if docking_program in VINA_VARIANTS:

        output_filename = os.path.join(output_dir, f"{docking_program}.pdbqt")  
        log_filename = os.path.join(output_dir, f"{docking_program}.log")  
        
        vina_out = execute_vina(
            ligand_filename=ligand_3D_filename, 
            target_filename=target_3D_filename,
            output_filename=output_filename, 
            log_filename=log_filename,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            exhaustiveness=exhaustiveness,
            num_poses=num_poses,
            vina_variant=docking_program,
            n_proc=n_proc,
            verbose=verbose,
        )

        if vina_out is not None: # run error

            vina_out_filename, all_pose_data_filename = vina_out

            # convert and separate output_file
            pose_pdb_filenames = convert_and_separate_vina_out_file(
                vina_output_filename=vina_out_filename,
                ligand_id=ligand_id,
                conversion_dir=pose_output_dir,
                verbose=verbose,
            )

            # load pose data

        else:
            pose_pdb_filenames = []

    elif docking_program == "plants":

        # run plants in this dir
        plants_output_dir = os.path.join(output_dir, "plants_output")

        # write configfile here
        configfile_location = os.path.join(output_dir, "plantsconfig")

        pose_mol2_filenames, all_pose_data_filename = execute_plants(
            ligand_title=ligand_id,
            ligand_filename=ligand_3D_filename,
            target_title=target_id,
            target_filename=target_3D_filename,
            plants_output_dir=plants_output_dir,
            configfile_location=configfile_location,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            num_poses=num_poses,
            verbose=verbose,
        )

        # convert poses to pdb
        pose_pdb_filenames = []
        if pose_mol2_filenames is not None:

            for pose_mol2_filename in pose_mol2_filenames:
                # get conf id
                stem, ext = os.path.splitext(pose_mol2_filename)
                pose_id = int(stem.split("_")[-1])
                pose_pdb_filename = obabel_convert(
                    input_format="mol2", 
                    input_filename=pose_mol2_filename, 
                    output_format="pdb",
                    output_filename=os.path.join(pose_output_dir, f"pose_{pose_id}.pdb"),
                    verbose=verbose,
                )
                # delete original mol2 file
                delete_file(pose_mol2_filename, verbose=verbose)
                # add to list of PDB pose files
                pose_pdb_filenames.append(pose_pdb_filename)
        
        # delete plants output dir
        # delete_directory(plants_output_dir, verbose=verbose)

    elif docking_program == "galaxydock":

        pose_mol2_filenames, all_pose_data_filename  = execute_galaxydock(
            ligand_filename=ligand_3D_filename,
            target_filename=target_3D_filename,
            output_dir=output_dir,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            use_multiprocessing=True,
            num_poses=num_poses,
            verbose=verbose,
        )

        if pose_mol2_filenames is not None:

            # convert and separate output_file
            # convert to pdb and split
            obabel_convert(
                input_format="mol2",
                input_filename=pose_mol2_filenames,
                output_format="pdb",
                output_filename="pose_",
                output_dir=pose_output_dir, 
                multiple=True,
                verbose=verbose,
            )

        pose_pdb_filenames = glob.glob(os.path.join(pose_output_dir, "*.pdb"))
    
    else:
        raise NotImplementedError(docking_program)
    
    # load pose data
    all_pose_data = load_json(all_pose_data_filename, key_type=int, verbose=verbose)
        
    return all_pose_data, pose_pdb_filenames

def generate_all_local_docking_tasks(
    executor,
    ligands_to_targets: dict,
    output_dir: str,
    num_poses: int,

    local_docking_program: str = "vina", # TODO

    verbose: bool = True,
    ):

    running_tasks = {}
        
    for ligand_id, ligand_data in ligands_to_targets.items():

        # make output for ligand
        ligand_output_dir = os.path.join(output_dir, ligand_id)
        os.makedirs(ligand_output_dir, exist_ok=True)
        
        ligand_pdb_filename = ligand_data["pdb_filename"]

        prepared_ligand_filename_vina = prepare_ligand_for_docking(
            ligand_filename=ligand_pdb_filename,
            docking_program=local_docking_program,
            overwrite=False,
            verbose=verbose
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

                prepared_target_filename_vina = prepare_target_for_docking(
                    target_filename=prepared_target_filename,
                    docking_program=local_docking_program,
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

                    task = executor.submit(
                        execute_local_docking,
                        ligand_id=ligand_id,
                        ligand_3D_filename=prepared_ligand_filename_vina,
                        target_id=f"{accession}_{pdb_id}_{pocket_num}",
                        target_3D_filename=prepared_target_filename_vina,
                        output_dir=ligand_target_pocket_output_dir,
                        center_x=pocket_data["center_x"],
                        center_y=pocket_data["center_y"],
                        center_z=pocket_data["center_z"],
                        size_x=max(pocket_data["size_x"], 15),
                        size_y=max(pocket_data["size_y"], 15),
                        size_z=max(pocket_data["size_z"], 15),
                        docking_program=local_docking_program,
                        num_poses=num_poses,
                        n_proc=1,
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
    output_dir: str,
    num_poses: int,
    num_complexes: int,

    local_docking_program: str = "vina", 
    
    n_proc: int = None,
    verbose: bool = True,
    ):

    if n_proc is None:
        n_proc = LOCAL_DOCKING_N_PROC

    if num_complexes is None: # convert all poses
        num_complexes = num_poses

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
            output_dir=output_dir,
            num_poses=num_poses,

            local_docking_program=local_docking_program,

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

            # get result of docking task
            task_output = running_task.result()

            del running_tasks[running_task]
            
            # handle missing vina / task fail
            if task_output is None:
                continue

            # output_filename, log_json_filename = task_output

            # # begin collation of poses
            # # check that Vina ran for target
            # if not os.path.exists(log_json_filename) or not os.path.exists(output_filename):
            #     continue

            #  # load log file as JSON
            all_pose_data, pose_pdb_filenames = task_output

            ligand_target_pocket_num_output_dir = os.path.join(
                output_dir,
                ligand_id,
                accession,
                pdb_id,
                f"pocket_{pocket_num}",
            )

            # create pose dir
            # pose_output_dir = os.path.join(
            #     ligand_target_pocket_num_output_dir,
            #     "poses",
            # )
            # os.makedirs(pose_output_dir, exist_ok=True)

            # create complexes dir
            complex_output_dir = os.path.join(
                ligand_target_pocket_num_output_dir,
                "complexes",
            )
            os.makedirs(complex_output_dir, exist_ok=True)

            # # split out file into pose_pdb_files
            # pose_pdb_files = convert_and_separate_vina_out_file(
            #     vina_output_filename=output_filename,
            #     conversion_dir=pose_output_dir,
            #     ligand_id=ligand_id,
            #     output_format="pdb",
            #     verbose=verbose,
            # )

            # iterate over pose pdb files and compute their location
            for pose_pdb_filename in pose_pdb_filenames:
                    
                pose_pdb_basename = os.path.basename(pose_pdb_filename)
                pose_pdb_basename = os.path.splitext(pose_pdb_basename)[0]
                pose_id = int(pose_pdb_basename.split("_")[1])

                # data for current pose
                pose_data = all_pose_data[pose_id]

                if num_complexes is not None and pose_id <= num_complexes:

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

                # add pose location to pose data
                pose_data = {
                    "center_x": center_x, 
                    "center_y": center_y, 
                    "center_z": center_z, 
                    "size_x": size_x, 
                    "size_y": size_y, 
                    "size_z": size_z, 
                    **pose_data
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
    local_docking_n_proc: int = None, 
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
        output_dir=local_docking_output_dir,
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