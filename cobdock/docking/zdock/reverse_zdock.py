import os
import shutil


if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.io.io_utils import load_json, write_json, copy_file, delete_directory
from utils.molecules.pdb_utils import identify_centre_of_mass, get_bounding_box_size

from cobdock.docking.zdock.zdock_utils import (
    ZDOCK_N_PROC,
    prepare_for_zdock, 
    execute_zdock, 
    run_create_pl,
)
from utils.molecules.pymol_utils import calculate_RMSD_pymol

MISSING_POSE_DATA = {
    "center_x": np.nan, 
    "center_y": np.nan, 
    "center_z": np.nan, 
    "size_x": np.nan, 
    "size_y": np.nan, 
    "size_z": np.nan, 
    "score": np.nan,
}

def generate_tasks(
    p,
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

        zdock_prepared_ligand_filename = os.path.join(
            ligand_output_dir,
            f"{ligand_id}.pdb",   
        )
        if not os.path.exists(zdock_prepared_ligand_filename):
            # execute mark_sur
            zdock_prepared_ligand_filename = prepare_for_zdock(
                input_filename=ligand_pdb_filename,
                output_filename=zdock_prepared_ligand_filename,
                overwrite=False,
                verbose=verbose,
                )

        if zdock_prepared_ligand_filename is None or not os.path.exists(zdock_prepared_ligand_filename):
            # skip_ligand = True
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
                        print ("ZDOCK: skipping target", pdb_id, "for ligand", ligand_id)
                    continue

                ligand_target_output_dir = os.path.join(
                    ligand_output_dir,
                    accession,
                    pdb_id,
                )
                os.makedirs(ligand_target_output_dir, exist_ok=True)

                # initialise dictionary for current PDB target
                collated_data[ligand_id][accession][pdb_id] = {}

                zdock_output_filename = os.path.join(
                    ligand_target_output_dir, 
                    f"{ligand_id}_{pdb_id}.out")

                # handle on target data
                target_data = pdb_ids[pdb_id]

                # skip based on bounding box volume
                bounding_box_volume = target_data["size_x"] * target_data["size_y"] * target_data["size_z"]

                if maximum_bounding_box_volume is not None and bounding_box_volume > maximum_bounding_box_volume:
                    if verbose:
                        print ("Skipping target", pdb_id, "due to volume of bounding box:", bounding_box_volume)
                    continue

                # assume target has been prepared already
                prepared_target_filename = target_data["prepared_filename"]

                # put prepared target in zdock root dir
                zdock_prepared_target_filename = os.path.join(
                    output_directory, 
                    f"{pdb_id}.pdb")

                if not os.path.exists(zdock_prepared_target_filename):
                    zdock_prepared_target_filename = prepare_for_zdock(
                        input_filename=prepared_target_filename,
                        output_filename=zdock_prepared_target_filename,
                        )
                                
                if zdock_prepared_target_filename is None or not os.path.exists(zdock_prepared_target_filename):
                    if verbose:
                        print ("Error in preparing target", pdb_id, "for ZDOCK, skipping it")
                    continue

                # copy ligand into ligand_target_output_dir
                zdock_prepared_ligand_basename = os.path.basename(zdock_prepared_ligand_filename)
                zdock_prepared_ligand_filename_in_ligand_target_output_dir = os.path.join(
                    ligand_target_output_dir,
                    zdock_prepared_ligand_basename,
                )
                if not os.path.exists(zdock_prepared_ligand_filename_in_ligand_target_output_dir):
                    copy_file(
                        zdock_prepared_ligand_filename, 
                        zdock_prepared_ligand_filename_in_ligand_target_output_dir,
                        verbose=verbose,
                    )

                # copy target into ligand_target_output_dir
                zdock_prepared_target_basename = os.path.basename(zdock_prepared_target_filename)
                zdock_prepared_target_filename_in_ligand_target_output_dir = os.path.join(
                    ligand_target_output_dir,
                    zdock_prepared_target_basename,
                )
                if not os.path.exists(zdock_prepared_target_filename_in_ligand_target_output_dir):
                    copy_file(
                        zdock_prepared_target_filename, 
                        zdock_prepared_target_filename_in_ligand_target_output_dir,
                        verbose=verbose,
                    )

                # submit tasj
                task = p.submit(
                    execute_zdock, 
                    ligand_filename=zdock_prepared_ligand_filename_in_ligand_target_output_dir,
                    target_filename=zdock_prepared_target_filename_in_ligand_target_output_dir,
                    output_filename=zdock_output_filename,
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


def execute_reverse_docking_zdock(
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
        print ("Executing ZDOCK for", len(ligands_to_targets), "ligands and collating data to", collated_data_filename)

    if os.path.exists(collated_data_filename):
        collated_data = load_json(collated_data_filename, verbose=verbose)
    else:
        collated_data = {}

    zdock_output_directory = os.path.join(
        output_dir, 
        "zdock",
    )
    os.makedirs(zdock_output_directory, exist_ok=True)

    with ProcessPoolExecutor(max_workers=ZDOCK_N_PROC) as p:

       
        running_tasks = generate_tasks(
            p=p,
            ligands_to_targets=ligands_to_targets,
            collated_data=collated_data,
            output_directory=zdock_output_directory,
            maximum_bounding_box_volume=maximum_bounding_box_volume,
            verbose=verbose,
        )   

        # await ZDOCK execution
        for running_task in as_completed(running_tasks):
            
            task_data = running_tasks[running_task]

            ligand_id = task_data["ligand_id"]
            accession = task_data["accession"]
            pdb_id = task_data["pdb_id"]
            ligand_pdb_filename = task_data["ligand_pdb_filename"]
            # prepared_target_filename = task_data["prepared_target_filename"]

            target_out_file = running_task.result()

            del running_tasks[running_task]

            if not os.path.exists(target_out_file):
                continue

            ligand_pdb_filename = ligands_to_targets[ligand_id]["pdb_filename"]
            target_pdb_filename = ligands_to_targets[ligand_id]["prepared_targets"][accession][pdb_id]["prepared_filename"]


            ligand_target_zdock_out_directory = os.path.join(
                zdock_output_directory,
                ligand_id,
                accession, 
                pdb_id,
            )
            os.makedirs(ligand_target_zdock_out_directory, exist_ok=True)

            # create pose and complex dir
            pose_output_dir = os.path.join(
                ligand_target_zdock_out_directory,
                "poses",)
            os.makedirs(pose_output_dir, exist_ok=True)

            complex_output_dir = os.path.join(
                ligand_target_zdock_out_directory,
                "complexes")
            os.makedirs(complex_output_dir, exist_ok=True)
            
            # read zdock out file and get score  
            if verbose:
                print ("Reading ZDOCK energies from", target_out_file)
            target_zdock_output_df = pd.read_csv(
                target_out_file, 
                index_col=None, 
                skiprows=4, # based on ZDock output header  
                sep="\t", 
                header=None,
                names=["rx", "ry", "rz", "tx", "ty", "tz", "energy"])

            zdock_energies = target_zdock_output_df["energy"] 

            pose_pdb_filenames = run_create_pl(
                input_filename=target_out_file, 
                pose_output_dir=pose_output_dir,
                target_filename=target_pdb_filename,
                complex_output_dir=complex_output_dir,
                num_complexes=num_complexes,
                verbose=verbose,
                )

            # for pose_num in range(1, ZDOCK_NUM_POSES+1):
            for pose_pdb_filename in pose_pdb_filenames:
                
                stem, ext = os.path.splitext(pose_pdb_filename)
                pose_id = int(stem.split("_")[-1])

                zdock_score = zdock_energies.iloc[pose_id - 1]

                centre_of_mass = identify_centre_of_mass(pose_pdb_filename, verbose=verbose)
                if centre_of_mass is None:
                    continue
                center_x, center_y, center_z = centre_of_mass
                
                size_x, size_y, size_z = get_bounding_box_size(pose_pdb_filename, verbose=verbose)

                zdock_pose_data = {
                    "center_x": center_x, 
                    "center_y": center_y, 
                    "center_z": center_z, 
                    "size_x": size_x, 
                    "size_y": size_y, 
                    "size_z": size_z, 
                    "score": zdock_score,
                }
                if compute_rmsd_with_submitted_ligand:
                    zdock_pose_data["pymol_rmsd"] = calculate_RMSD_pymol(
                        ligand_pdb_filename,
                        pose_pdb_filename,
                        verbose=verbose,
                    )
            
                collated_data[ligand_id][accession][pdb_id][pose_id] = zdock_pose_data
        
    
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
            print ("Removing output directory", zdock_output_directory)
        delete_directory(zdock_output_directory, verbose=verbose)

    # write collated data if it has changed
    write_json(collated_data, collated_data_filename, verbose=verbose)

    return collated_data


if __name__== '__main__':
    pass
    # reverse_zdock(
    #     targets_filename=args.receptor, 
    #     ligands_filename=args.ligand, 
    #     maximum_bounding_box_volume=args.searchsize ** 3,
    #     )
