import os 
import shutil
import sys
import os.path

if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import glob

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from cobdock.docking.galaxydock.galaxydock_utils import (
    GALAXYDOCK_N_PROC,
    prepare_ligand_for_galaxydock, 
    prepare_target_for_galaxydock, 
    execute_galaxydock,
    )
from utils.molecules.pymol_utils import create_complex_with_pymol, calculate_RMSD_pymol
from utils.io.io_utils import load_json, write_json, delete_directory
from utils.molecules.openbabel_utils import obabel_convert
from utils.molecules.pdb_utils import identify_centre_of_mass, get_bounding_box_size

MISSING_POSE_DATA = {
    "center_x": np.nan, 
    "center_y": np.nan, 
    "center_z": np.nan, 
    "size_x": np.nan, 
    "size_y": np.nan, 
    "size_z": np.nan, 
    "rmsd": np.nan,
    "score": np.nan,
    "autodock": np.nan,
    "drug_score": np.nan,
    "internal_energy": np.nan, 
}

def generate_tasks(
    p,
    ligands_to_targets: dict,
    collated_data: dict,
    galaxydock_output_directory: str,
    maximum_bounding_box_volume: float,
    verbose: bool = True,
    ):

    running_tasks = {}

    for ligand_id in ligands_to_targets:

        # add ligand to collated data if it is not there 
        if ligand_id not in collated_data:
            collated_data[ligand_id] = {}

        # make output directory for ligand
        ligand_output_dir = os.path.join(galaxydock_output_directory, ligand_id)
        os.makedirs(ligand_output_dir, exist_ok=True)
        
        ligand_pdb_filename = ligands_to_targets[ligand_id]["pdb_filename"]
        galaxydock_prepared_ligand_filename = os.path.join(
            ligand_output_dir,
            f"{ligand_id}.mol2" # mol2 format required for ligand 
        )

        # prepare ligand for docking
        if not os.path.exists(galaxydock_prepared_ligand_filename):
            galaxydock_prepared_ligand_filename = prepare_ligand_for_galaxydock(
                input_filename=ligand_pdb_filename,
                output_filename=galaxydock_prepared_ligand_filename,
                verbose=verbose,
                )

        if galaxydock_prepared_ligand_filename is None or not os.path.exists(galaxydock_prepared_ligand_filename):
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

                # skip target if poses exist in collated_data
                # if pdb_id in collated_data[ligand_id][accession] and \
                #     len(collated_data[ligand_id][accession][pdb_id]) > 0 and \
                #     "null" not in collated_data[ligand_id][accession][pdb_id]:
                if pdb_id in collated_data[ligand_id][accession]:
                    if verbose:
                        print ("GalaxyDock: skipping target", pdb_id, "for ligand", ligand_id)
                    continue

                # initialise dictionary for current PDB target
                collated_data[ligand_id][accession][pdb_id] = {}

                ligand_target_output_dir = os.path.join(
                    ligand_output_dir, 
                    accession,
                    pdb_id)

                os.makedirs(ligand_target_output_dir, exist_ok=True,)
                
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

                # prepare target for galaxydock (convert to PDB and some residue relabelling)
                prepared_target_filename_galaxydock = os.path.join(
                    galaxydock_output_directory, 
                    f"{pdb_id}.pdb")
                if not os.path.exists(prepared_target_filename_galaxydock):
                    prepared_target_filename_galaxydock = prepare_target_for_galaxydock(
                        input_filename=prepared_target_filename,
                        output_filename=prepared_target_filename_galaxydock,
                        verbose=verbose,
                    )

                if prepared_target_filename_galaxydock is None or not os.path.exists(prepared_target_filename_galaxydock):
                    if verbose:
                        print ("Error in preparing target", pdb_id, "for GalaxyDock, skipping it")
                    continue

                # submit task 
                task = p.submit(
                    execute_galaxydock,
                    ligand_filename=galaxydock_prepared_ligand_filename,
                    target_filename=prepared_target_filename_galaxydock,
                    output_dir=ligand_target_output_dir,
                    center_x=target_data["center_x"],
                    center_y=target_data["center_y"],
                    center_z=target_data["center_z"],
                    size_x=target_data["size_x"],
                    size_y=target_data["size_y"],
                    size_z=target_data["size_z"],
                    # use_multiprocessing=use_multiprocessing,
                    use_multiprocessing=False,
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
                        

def execute_reverse_docking_galaxydock(
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

    # precision = 3
    # use_multiprocessing = True # its not very memory intensive, but uses all cores, regardless of setting
    use_multiprocessing = False # maybe better to externally parallelise?

    if verbose:
        print ("Executing Galaxydock for", len(ligands_to_targets), "ligands and collating data to", collated_data_filename)
        print ("Using multiprocessing:", use_multiprocessing)

    if os.path.exists(collated_data_filename):
        if verbose:
            print (collated_data_filename, "already exists -- loading it")
        collated_data = load_json(collated_data_filename)
    else:
        collated_data = {}

    galaxydock_output_directory = os.path.join(
        output_dir, 
        "galaxydock",
        )
    os.makedirs(galaxydock_output_directory, exist_ok=True)

    with ProcessPoolExecutor(max_workers=GALAXYDOCK_N_PROC) as p:

        running_tasks = generate_tasks(
            p=p,
            ligands_to_targets=ligands_to_targets,
            collated_data=collated_data,
            galaxydock_output_directory=galaxydock_output_directory,
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

            galaxydock_output_info_filename, galaxydock_pose_output_filename = running_task.result()

            del running_tasks[running_task]

            # begin collation of pose data
            if not os.path.exists(galaxydock_output_info_filename) or not os.path.exists(galaxydock_pose_output_filename): 
                continue
        
            if verbose:
                print ("Reading GalaxyDock scores from", galaxydock_output_info_filename)
            galaxydock_output_df = pd.read_fwf(
                galaxydock_output_info_filename, 
                comment="!", 
                sep="\t", 
                index_col=0)
            # necessary
            galaxydock_output_df = galaxydock_output_df.fillna(0)

            ligand_target_output_dir = os.path.join(
                galaxydock_output_directory,
                ligand_id,
                accession,
                pdb_id,
            )

            # make pose and complex directory
            pose_output_dir = os.path.join(
                ligand_target_output_dir, 
                "poses")
            os.makedirs(pose_output_dir, exist_ok=True)

            complex_output_dir = os.path.join(
                ligand_target_output_dir,
                "complexes")
            os.makedirs(complex_output_dir, exist_ok=True)

            # convert to pdb and split
            obabel_convert(
                input_format="mol2",
                input_filename=galaxydock_pose_output_filename,
                output_format="pdb",
                output_filename="pose_",
                output_dir=pose_output_dir, 
                multiple=True,
                verbose=verbose,
            )

            # iterate over pose PDB files
            for pose_pdb_filename in glob.iglob(os.path.join(pose_output_dir, "pose_*.pdb")):

                # get rank
                stem, ext = os.path.splitext(pose_pdb_filename)
                rank = int(stem.split("_")[-1])

                if num_complexes is not None and rank <= num_complexes:

                    # make complex with current pose 
                    complex_filename = os.path.join(
                        complex_output_dir,
                        f"complex_{rank}.pdb")
                    create_complex_with_pymol(
                        input_pdb_files=[prepared_target_filename, pose_pdb_filename],
                        output_pdb_filename=complex_filename,
                        verbose=verbose,
                    )

                # compute bounding box for current pose
                center_of_mass = identify_centre_of_mass(pose_pdb_filename, verbose=verbose)
                if center_of_mass is None:
                    continue
                center_x, center_y, center_z = center_of_mass
                size_x, size_y, size_z = get_bounding_box_size(pose_pdb_filename, verbose=verbose)

                galaxydock_score = galaxydock_output_df["Energy"].loc[rank]
                if isinstance(galaxydock_score, str):
                    galaxydock_score = 0
                galaxydock_rmsd = galaxydock_output_df["l_RMSD"].loc[rank]
                if isinstance(galaxydock_rmsd, str):
                    galaxydock_rmsd = 0
                galaxydock_autodock = galaxydock_output_df["ATDK_E"].loc[rank]
                if isinstance(galaxydock_autodock, str):
                    galaxydock_autodock = 0
                galaxydock_internal_energy = galaxydock_output_df["INT_E"].loc[rank]
                if isinstance(galaxydock_autodock, str):
                    galaxydock_internal_energy = 0
                galaxydock_drug_score = galaxydock_output_df["DS_E"].loc[rank]
                if isinstance(galaxydock_drug_score, str):
                    galaxydock_drug_score = 0

                pose_data = {
                    "center_x": center_x, 
                    "center_y": center_y, 
                    "center_z": center_z, 
                    "size_x": size_x, 
                    "size_y": size_y, 
                    "size_z": size_z, 
                    "rmsd": galaxydock_rmsd,
                    "score": galaxydock_score,
                    "autodock": galaxydock_autodock,
                    "drug_score": galaxydock_drug_score,
                    "internal_energy": galaxydock_internal_energy, 
                }

                if compute_rmsd_with_submitted_ligand:

                    # calculate RMSD with submitted ligand
                    pose_data["pymol_rmsd"] = calculate_RMSD_pymol(
                        reference_filename=ligand_pdb_filename,
                        model_filename=pose_pdb_filename,
                        verbose=verbose,
                    )

                collated_data[ligand_id][accession][pdb_id][rank] = pose_data

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
            print ("Removing output directory", galaxydock_output_directory)
        delete_directory(galaxydock_output_directory, verbose=verbose)

    # write collated vina data if it has changed
    write_json(collated_data, collated_data_filename, verbose=verbose,)

    return collated_data

if __name__== '__main__':
    pass