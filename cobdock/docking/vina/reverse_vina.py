import os 
import sys
import os.path

import shutil
import sys

if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.io.io_utils import load_json, write_json, delete_directory
from utils.molecules.pdb_utils import identify_centre_of_mass, get_bounding_box_size

from cobdock.docking.vina.vina_utils import (
    VINA_N_PROC,
    prepare_ligand_for_vina, 
    prepare_target_for_vina, 
    execute_vina, 
    convert_and_separate_vina_out_file,
    )
from utils.molecules.pymol_utils import create_complex_with_pymol, calculate_RMSD_pymol

MISSING_POSE_DATA = {
    "center_x": np.nan, 
    "center_y": np.nan, 
    "center_z": np.nan, 
    "size_x": np.nan, 
    "size_y": np.nan, 
    "size_z": np.nan, 
    "energy": np.nan,
    "rmsd_lb": np.nan,
    "rmsd_ub": np.nan,
}

def generate_tasks(
    executor,
    ligands_to_targets: dict,
    collated_data: dict,
    output_directory: str,
    maximum_bounding_box_volume: float,
    verbose: bool = True,
    ):

    running_tasks = {}
    
    for ligand_id in ligands_to_targets:

        # add ligand to collated data if it is not there 
        if ligand_id not in collated_data:
            collated_data[ligand_id] = {}

        # make output for ligand
        ligand_output_dir = os.path.join(output_directory, ligand_id)
        os.makedirs(ligand_output_dir, exist_ok=True)
        
        ligand_pdb_filename = ligands_to_targets[ligand_id]["pdb_filename"]
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

            # add accession to collated_data if it is not there
            if accession not in collated_data[ligand_id]:
                collated_data[ligand_id][accession] = {}

            # get set of PDB targets for current accession
            pdb_ids = ligand_accessions[accession]

            # iterate over PDB targets
            for pdb_id in pdb_ids:

                # skip target if it exists in collated_data
                # if pdb_id in collated_data[ligand_id][accession] and \
                #     len(collated_data[ligand_id][accession][pdb_id]) > 0 and \
                #     "null" not in collated_data[ligand_id][accession][pdb_id]:
                if pdb_id in collated_data[ligand_id][accession]:
                    if verbose:
                        print ("VINA: skipping target", pdb_id, "for ligand", ligand_id)
                    continue

                ligand_target_output_dir = os.path.join(
                    ligand_output_dir,
                    accession,
                    pdb_id,
                )
                os.makedirs(ligand_target_output_dir, exist_ok=True)

                # initialise dictionary for current PDB target
                collated_data[ligand_id][accession][pdb_id] = {}

                vina_output_filename = os.path.join(ligand_target_output_dir, f"{ligand_id}_{pdb_id}.pdbqt")
                vina_log_filename = os.path.join(ligand_target_output_dir, f"{ligand_id}_{pdb_id}.log")
                vina_log_json_filename = vina_log_filename + ".json"

                # handle on target data
                target_data = pdb_ids[pdb_id]
                
                # skip target based on bounding box volume
                # compute volume of docking box
                bounding_box_volume = target_data["size_x"] * target_data["size_y"] * target_data["size_z"]
                
                if maximum_bounding_box_volume is not None and bounding_box_volume > maximum_bounding_box_volume:
                    if verbose:
                        print ("Skipping target", pdb_id, "due to volume of bounding box:", bounding_box_volume)
                    continue
                
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
                
                # submit task
                task = executor.submit(
                    execute_vina,
                    ligand_filename=prepared_ligand_filename_vina,
                    target_filename=prepared_target_filename_vina,
                    output_filename=vina_output_filename,
                    log_filename=vina_log_filename,
                    center_x=target_data["center_x"],
                    center_y=target_data["center_y"],
                    center_z=target_data["center_z"],
                    size_x=target_data["size_x"],
                    size_y=target_data["size_y"],
                    size_z=target_data["size_z"],
                    vina_variant="vina",
                    n_proc=1,
                    verbose=verbose,
                )

                running_tasks[task] = {
                    "ligand_id": ligand_id,
                    "accession": accession,
                    "pdb_id": pdb_id,
                    "ligand_pdb_filename": ligand_pdb_filename,
                    "prepared_target_filename": prepared_target_filename,
                }

    return running_tasks     

def execute_reverse_docking_vina(
    ligands_to_targets: dict,
    output_dir: str,
    collated_data_filename: str,
    maximum_bounding_box_volume: float = None,
    skip_if_complete: bool = True,
    delete_output_directory: bool = True,
    compute_rmsd_with_submitted_ligand: bool = True,
    num_complexes: int = 1,
    verbose: bool = True,
    ):

    if verbose:
        print ("Executing Vina for", len(ligands_to_targets), "ligands and collating data to", collated_data_filename)

    if os.path.exists(collated_data_filename):
        if verbose:
            print (collated_data_filename, "already exists -- loading it")
        collated_data = load_json(collated_data_filename, verbose=verbose)
    else:
        collated_data = {}

    vina_output_directory = os.path.join(
        output_dir, 
        "vina",
    )
    os.makedirs(vina_output_directory, exist_ok=True)

    with ProcessPoolExecutor(max_workers=VINA_N_PROC) as p:

        running_tasks = generate_tasks(
            executor=p,
            ligands_to_targets=ligands_to_targets,
            collated_data=collated_data,
            output_directory=vina_output_directory,
            maximum_bounding_box_volume=maximum_bounding_box_volume,
            verbose=verbose,
        )

        for running_task in as_completed(running_tasks):

            task_data = running_tasks[running_task]

            ligand_id = task_data["ligand_id"]
            accession = task_data["accession"]
            pdb_id = task_data["pdb_id"]
            ligand_pdb_filename = task_data["ligand_pdb_filename"]
            prepared_target_filename = task_data["prepared_target_filename"]

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

            ligand_target_output_dir = os.path.join(
                vina_output_directory,
                ligand_id,
                accession,
                pdb_id,
            )

            # create pose and complex dir
            pose_output_dir = os.path.join(
                ligand_target_output_dir,
                "poses",
            )
            os.makedirs(pose_output_dir, exist_ok=True)

            complex_output_dir = os.path.join(
                ligand_target_output_dir,
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

                if compute_rmsd_with_submitted_ligand:

                    # calculate RMSD with submitted ligand
                    pose_data["pymol_rmsd"] = calculate_RMSD_pymol(
                        reference_filename=ligand_pdb_filename,
                        model_filename=pose_pdb_filename,
                        verbose=verbose,
                    )

                collated_data[ligand_id][accession][pdb_id][pose_id] = pose_data

   # ensure all (ligand, pdb) pairs exist in collated_data
    for ligand_id, ligand_data in ligands_to_targets.items():

        if ligand_id not in collated_data:
            collated_data[ligand_id] = {}

        for accession, pdb_ids, in ligand_data["prepared_targets"].items():
            if accession not in collated_data[ligand_id]:
                collated_data[ligand_id][accession] = {}
            for pdb_id in pdb_ids:
                if pdb_id not in collated_data[ligand_id][accession] or len(collated_data[ligand_id][accession][pdb_id]) == 0:
                    collated_data[ligand_id][accession][pdb_id] = {}
                    collated_data[ligand_id][accession][pdb_id][None] = MISSING_POSE_DATA
                    if compute_rmsd_with_submitted_ligand:
                        collated_data[ligand_id][accession][pdb_id][None]["pymol_rmsd"] = np.nan

    if delete_output_directory:
        if verbose:
            print ("Removing output directory", vina_output_directory)
        delete_directory(vina_output_directory, verbose=verbose)

    # write collated if it has changed
    write_json(collated_data, collated_data_filename, verbose=verbose)

    return collated_data

if __name__ == '__main__':
    pass