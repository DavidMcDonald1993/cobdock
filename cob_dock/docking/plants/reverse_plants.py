import os 
import shutil
import sys
import os.path

if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import re, glob

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.io.io_utils import load_json, write_json, delete_file, delete_directory
from utils.molecules.pdb_utils import identify_centre_of_mass, get_bounding_box_size
from utils.molecules.openbabel_utils import obabel_convert

from ai_blind_docking.docking_utils.plants_utils import (
    PLANTS_N_PROC, 
    execute_plants, 
    prepare_for_plants, 
    # determine_binding_site_radius, 
    # get_plants_score,
)
from utils.molecules.pymol_utils import create_complex_with_pymol, calculate_RMSD_pymol

MISSING_POSE_DATA = {
    "center_x": np.nan, 
    "center_y": np.nan, 
    "center_z": np.nan, 
    "size_x": np.nan, 
    "size_y": np.nan, 
    "size_z": np.nan, 
    "total_score": np.nan,
    "score_rb_pen": np.nan,
    "score_norm_hevatoms": np.nan,
    "score_norm_crt_hevatoms": np.nan,
    "score_norm_weight": np.nan,
    "score_norm_crt_weight": np.nan,
    "score_rb_pen_norm_crt_hevatoms": np.nan,
    "score_norm_contact": np.nan,
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

        plants_prepared_ligand_filename = os.path.join(
            ligand_output_dir,
            f"{ligand_id}.mol2",
        )

        if not os.path.exists(plants_prepared_ligand_filename):
        
            # prepare ligand for PLANTS
            plants_prepared_ligand_filename = prepare_for_plants(
                title=ligand_id,
                input_filename=ligand_pdb_filename, 
                output_filename=plants_prepared_ligand_filename,
                verbose=verbose,
                )

        if plants_prepared_ligand_filename is None or not os.path.exists(plants_prepared_ligand_filename):
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
                        print ("PLANTS: skipping target", pdb_id, "for ligand", ligand_id)
                    continue

                # initialise dictionary for current PDB target
                collated_data[ligand_id][accession][pdb_id] = {}

                configfile_location = os.path.join(
                    ligand_output_dir, 
                    accession,
                    f"{pdb_id}_plantsconfig",
                    )

                ligand_target_output_dir = os.path.join(
                    ligand_output_dir, 
                    accession,
                    pdb_id,
                )
                os.makedirs(ligand_target_output_dir, exist_ok=True, )

                # plants_ranking_filename = os.path.join(ligand_target_output_dir, "ranking.csv")

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

                plants_prepared_target_filename = os.path.join(
                    output_directory, 
                    f"{pdb_id}.mol2")
                if not os.path.exists(plants_prepared_target_filename):
                    plants_prepared_target_filename = prepare_for_plants(
                        title=pdb_id,
                        input_filename=prepared_target_filename,
                        output_filename=plants_prepared_target_filename)
                        
                if plants_prepared_target_filename is None or not os.path.exists(plants_prepared_target_filename):
                    if verbose:
                        print ("Skipping target", pdb_id, "due to volume of bounding box:", bounding_box_volume)
                    continue

                # submit task
                task = p.submit(
                    execute_plants, 
                    ligand_title=ligand_id,
                    ligand_filename=plants_prepared_ligand_filename,
                    target_title=pdb_id,
                    target_filename=plants_prepared_target_filename,
                    plants_output_dir=ligand_target_output_dir,
                    configfile_location=configfile_location,
                    center_x=target_data["center_x"],
                    center_y=target_data["center_y"],
                    center_z=target_data["center_z"],
                    size_x=target_data["size_x"],
                    size_y=target_data["size_y"],
                    size_z=target_data["size_z"],
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

def execute_reverse_docking_plants(
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
        print ("Executing PLANTS for", len(ligands_to_targets), "ligands and collating data to", collated_data_filename)

    if os.path.exists(collated_data_filename):
        collated_data = load_json(collated_data_filename, verbose=verbose,)
    else:
        collated_data = {}


    plants_output_directory = os.path.join(
        output_dir, 
        "plants",
        )
    os.makedirs(plants_output_directory, exist_ok=True)

    # set of ligand, accession, target tuples to update
    # to_update = set()

    # an example of multiprocessing
    with ProcessPoolExecutor(max_workers=PLANTS_N_PROC) as p:

        
        running_tasks = generate_tasks(
            p=p,
            ligands_to_targets=ligands_to_targets,
            collated_data=collated_data,
            output_directory=plants_output_directory,
            maximum_bounding_box_volume=maximum_bounding_box_volume,
            verbose=verbose,
        )      

        # await PLANTS execution
        for running_task in as_completed(running_tasks):
            
            task_data = running_tasks[running_task]

            ligand_id = task_data["ligand_id"]
            accession = task_data["accession"]
            pdb_id = task_data["pdb_id"]
            ligand_pdb_filename = task_data["ligand_pdb_filename"]
            prepared_target_filename = task_data["prepared_target_filename"]

            pose_mol2_filenames, plants_score_filename = running_task.result()

            del running_tasks[running_task]

            ligand_target_plants_out_directory = os.path.join(
                plants_output_directory,
                ligand_id, 
                accession,
                pdb_id,
            )

            # check that PLANTS successfully ran
            if not os.path.exists(plants_score_filename):
                continue

            # load PLANTS scores
            plants_scores = load_json(plants_score_filename, key_type=int, verbose=verbose)

            # make pose and complex directory
            pose_output_dir = os.path.join(
                ligand_target_plants_out_directory,
                "poses")
            os.makedirs(pose_output_dir, exist_ok=True)

            complex_output_dir = os.path.join(
                ligand_target_plants_out_directory,
                "complexes")
            os.makedirs(complex_output_dir, exist_ok=True)

            # iterate over PLANTS poses (mol2 format) and write PDB poses
            for pose_mol2_filename in pose_mol2_filenames:

                score_index, _ = os.path.splitext(os.path.basename(pose_mol2_filename))
                pose_id = int(score_index.split("_")[-1])

                # write pose
                pose_pdb_filename = os.path.join(
                    pose_output_dir,
                    f"pose_{pose_id}.pdb"
                )
                pose_pdb_filename = obabel_convert(
                    input_format="mol2",
                    input_filename=pose_mol2_filename,
                    output_format="pdb",
                    output_filename=pose_pdb_filename,
                    verbose=verbose,
                )

                # write complex
                if num_complexes is not None and pose_id <= num_complexes:

                    complex_filename = os.path.join(
                        complex_output_dir,
                        f"complex_{pose_id}.pdb",
                    )
                    create_complex_with_pymol(
                        input_pdb_files=[prepared_target_filename, pose_pdb_filename],
                        output_pdb_filename=complex_filename,
                        verbose=verbose,
                    )
                
            # cleanup ligand_target_plants_out_directory
            for ext in (
                "pdb", 
                # "mol2", # keep mol2 for now
                ):
                for structure_filename in glob.iglob(
                    os.path.join(ligand_target_plants_out_directory, f"*conf*.{ext}")
                ):
                    # certain to exist due to usage of glob
                    delete_file(structure_filename, verbose=verbose)

            # iterate over PDB poses, compute locations and scores
            for pose_pdb_filename in glob.iglob(
                os.path.join(pose_output_dir, "pose_*.pdb")
                ):

                stem, ext = os.path.splitext(pose_pdb_filename)
                pose_id = int(stem.split("_")[-1])

                center_of_mass = identify_centre_of_mass(pose_pdb_filename, verbose=verbose)
                if center_of_mass is None: # handle center of mass fail
                    continue
                center_x, center_y, center_z = center_of_mass
                size_x, size_y, size_z = get_bounding_box_size(pose_pdb_filename, verbose=verbose)

                # extract data about current pose
                plant_score_dict = plants_scores[pose_id]

                plants_pose_data = {
                    "center_x": center_x, 
                    "center_y": center_y, 
                    "center_z": center_z, 
                    "size_x": size_x, 
                    "size_y": size_y, 
                    "size_z": size_z, 
                    **plant_score_dict,
                }
                if compute_rmsd_with_submitted_ligand:
                    plants_pose_data["pymol_rmsd"] = calculate_RMSD_pymol( 
                        os.path.abspath(ligand_pdb_filename),
                        os.path.abspath(pose_pdb_filename),
                        verbose=verbose,
                    )

                collated_data[ligand_id][accession][pdb_id][pose_id] = plants_pose_data


   # ensure all (ligand, pdb) pairs exist in collated_data
    for ligand_id, ligand_data in ligands_to_targets.items():

        if ligand_id not in collated_data:
            collated_data[ligand_id] = {}

        for accession, pdb_ids, in ligand_data["prepared_targets"].items():
            if accession not in collated_data[ligand_id]:
                collated_data[ligand_id][accession] = {}
            for pdb_id in pdb_ids:
                if pdb_id not in collated_data[ligand_id][accession] or len(collated_data[ligand_id][accession][pdb_id]) == 0:
                    collated_data[ligand_id][accession][pdb_id][None] = MISSING_POSE_DATA
                    if compute_rmsd_with_submitted_ligand:
                        collated_data[ligand_id][accession][pdb_id][None]["pymol_rmsd"] = np.nan

    if delete_output_directory:
        if verbose:
            print ("Removing output directory", plants_output_directory)
        delete_directory(plants_output_directory, verbose=verbose)

    # write collated data if it has changed
    write_json(collated_data, collated_data_filename, verbose=verbose)

    return collated_data

if __name__== '__main__':
    pass
