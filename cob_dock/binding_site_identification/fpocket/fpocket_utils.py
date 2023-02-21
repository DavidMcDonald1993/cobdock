

import os
import shutil
import re
import subprocess

from concurrent.futures import ProcessPoolExecutor as Pool

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    os.pardir,
    os.pardir,
    ))

if __name__ == "__main__":

    import sys
    import os.path

    sys.path.insert(1, PROJECT_ROOT)

from Bio.PDB import PDBParser, PDBIO

from utils.sys_utils import execute_system_command
from utils.io.io_utils import copy_file, dataframe_to_dict, delete_directory, delete_file, load_json, write_json

from utils.molecules.pdb_utils import (
    identify_centre_of_mass, 
    get_bounding_box_size, 
    download_pdb_structure_using_pdb_fetch,
    AA3_TO_AA1,
    )

# FPOCKET_EXECUTABLE = "fpocket"
FPOCKET_EXECUTABLE = "./bin/location_analysis/fpocket/fpocket"

FPOCKET_OUT = os.path.join(
    "cavities", 
    "FPOCKET_out",
    )
POCKET_ANALYSIS_N_PROC = int(os.environ.get("POCKET_ANALYSIS_N_PROC", default=1))

def call_fpocket(
    input_filename: str, 
    chains_to_keep: list = None,
    main_job_id: int = None,
    verbose: bool = False,
    ):
    if verbose:
        print ("Running fpocket on file", input_filename)
    
    cmd = f"{FPOCKET_EXECUTABLE} -f {input_filename}"
    if chains_to_keep is not None:
        if verbose:
            print ("Keeping chains", chains_to_keep)
        chains_to_keep = ",".join(chains_to_keep)
        cmd += f" -k {chains_to_keep}"

    if os.path.exists(input_filename):
        try:
            execute_system_command(cmd, main_job_id=main_job_id, verbose=verbose)
        except Exception as e:
            print ("Exception calling fpocket", e)
            return None

    return input_filename

def show_pocket_scores(
    pocket_score_filename: str,
    ):
    return dataframe_to_dict(pocket_score_filename)

def select_best_scoring_pocket(
    pocket_score_filename: str, 
    by: str = "drug_score",
    verbose: bool = False,
    ):

    pocket_scores = pd.read_csv(pocket_score_filename, sep=" ", index_col=0)

    if verbose:
        print("Using column", by, "to select best pocket")

    return pocket_scores[by].idxmax()

def return_pocket(
    best_scoring_pocket: int, 
    pocket_dir: str,
    ):
    pocket_filename = os.path.join(
        pocket_dir, 
        f"pocket{best_scoring_pocket}_atm.pdb")
    assert os.path.exists(pocket_filename)
    return pocket_filename

def read_fpocket_pdb_file(
    pdb_filename: str,
    verbose: bool = False,
    ):

    if verbose:
        print ("Reading fpocket output PDB file", pdb_filename)
    all_header_values = {}
    try:
        lines = subprocess.check_output(
            f"awk '/^HEADER [0-9]+/' {pdb_filename}", shell=True).decode("utf-8").split("\n")
        for line in lines:
            if line == "":
                continue
            line = "-".join(line.split("-")[1:])
            key, value = line.split(":")
            # leading whitespace
            key = re.sub(r"^\s+", "", key)
            # trailing whitespace
            key = re.sub(r"\s+$", "", key)
            # convert to lowercase and remove spaces
            key = key.lower().replace(" ", "_")
            # remove .
            key = key.replace(".", "")
            value = float(value)
            all_header_values[key] = value
    except:
        all_header_values = {}
    return all_header_values

def read_fpocket_info_file(
    fpocket_info_filename: str,
    verbose: bool = False,
    ):

    if verbose:
        print ("Reading FPocket info file:", fpocket_info_filename)

    if not os.path.exists(fpocket_info_filename):
        print (fpocket_info_filename, "does not exist!")
        return {}

    # read info file and skip "Pocket" and empty lines
    with open(fpocket_info_filename, "r") as f:
        lines = [
            line.strip() for line in f.readlines()
            if line != "\n" and not line.startswith("Pocket")
        ]
    num_lines = len(lines)

    # split lines into pocket chunks 
    chunk_size = 19 # 19 features per pocket 

    # number of pockets 
    num_pockets = num_lines // chunk_size

    all_pockets = {}

    for pocket_id in range(num_pockets):
        
        pocket_chunk = lines[pocket_id*chunk_size: (pocket_id+1)*chunk_size]
        
        pocket_chunk_data = {}
        
        for line in pocket_chunk:
            
            key, value = line.split("\t")
            
            # remove colon
            key = key.replace(":", "")
            # leading whitespace
            key = re.sub(r"^\s+", "", key)
            # trailing whitespace
            key = re.sub(r"\s+$", "", key)
            # remove .-
            key = re.sub("[.-]", "", key)
            # replace whitespace with _
            key = re.sub("\s+", "_", key)
            # convert to lowercase
            key = key.lower()
            
            # convert value to float 
            value = float(value)

            # rename "score" to "pocket_score" for consistency with p2rank
            if key == "score":
                key = "pocket_score"
            
            pocket_chunk_data[key] = value

        all_pockets[pocket_id+1] = pocket_chunk_data

    return all_pockets

def run_fpocket_and_collate_single_target(
    target_identifier: str, # pdb_id / target_name
    output_dir: str,
    min_pocket_score: float = 0,
    existing_pdb_filename: str = None,
    read_features_from_pocket_pdbs: bool = False,
    delete_output_dir: bool = True,
    verbose: bool = False,
    ):

    if verbose:
        print ("Running fpocket on target", target_identifier, "and collecting pocket data into a dictionary")

    output_dir = os.path.join(
        output_dir,
        FPOCKET_OUT)
    os.makedirs(output_dir, exist_ok=True)

    target_fpocket_output_dir = os.path.join(
        output_dir,
        f"{target_identifier}_out")

    if not os.path.isdir(target_fpocket_output_dir):

        pdb_filename = os.path.join(
            output_dir,
            f"{target_identifier}.pdb")

        if not os.path.exists(pdb_filename):
            if existing_pdb_filename is not None and os.path.exists(existing_pdb_filename):
                if existing_pdb_filename != pdb_filename:
                    copy_file(existing_pdb_filename, pdb_filename, verbose=verbose) # because the file is deleted after fpocket is called
            else:
                pdb_filename = download_pdb_structure_using_pdb_fetch(
                    pdb_id=target_identifier,
                    pdb_filename=pdb_filename,
                    verbose=verbose,
                )
        
        # call fpocket            
        call_fpocket(pdb_filename, verbose=verbose)

        # delete the target pdb file
        delete_file(pdb_filename, verbose=verbose)

    # read info file
    fpocket_info_filename = os.path.join(
        target_fpocket_output_dir,
        f"{target_identifier}_info.txt"
    )

    # assert os.path.exists(fpocket_info_filename)
    all_fpocket_pocket_data = read_fpocket_info_file(
        fpocket_info_filename,
        verbose=verbose
    )
      
    target_pocket_dir = os.path.join(
        target_fpocket_output_dir,
        "pockets",
        )

    fpocket_target_data = {}

    if os.path.isdir(target_pocket_dir):

        # initialise PDBParser
        parser = PDBParser()

        # build list of pocket PDB files 
        pocket_pdb_filenames = [file 
            for file in os.listdir(target_pocket_dir) 
            if file.endswith("atm.pdb") and "env" not in file]

        for pocket_pdb_filename in pocket_pdb_filenames:
            pocket_number = pocket_pdb_filename.split("_")[0]
            # remove `pocket`
            pocket_number = int(pocket_number.split("pocket")[1])

            # full filename
            pocket_pdb_filename = os.path.join(
                target_pocket_dir, 
                pocket_pdb_filename)
            
            # read pocket pdb file as text file
            if read_features_from_pocket_pdbs:
                pocket_data = read_fpocket_pdb_file(pocket_pdb_filename, verbose=verbose)

            else: # read from info file

                if pocket_number not in all_fpocket_pocket_data:
                    print (pocket_number, "is missing from info file!")
                    continue

                pocket_data = all_fpocket_pocket_data[pocket_number]

            if "pocket_score" not in pocket_data:
                print ("pocket_score is missing from pocket data" )
                continue

            pocket_score = pocket_data["pocket_score"] 

            # potentially filter pocket 
            if min_pocket_score is not None and pocket_score < min_pocket_score:
                continue
        
            center_x, center_y, center_z = identify_centre_of_mass(
                pocket_pdb_filename, 
                geometric=True,
                verbose=verbose)
            size_x, size_y, size_z = get_bounding_box_size(
                pocket_pdb_filename, 
                allowance=0,
                verbose=verbose)

            # get residue ID of pocket residues 
            # load pocket 
            pocket_structure = parser.get_structure(
                f"{target_identifier}_pocket_{pocket_number}",
                pocket_pdb_filename,
            )

            # build list of residue IDs
            # now in (resname, residue_id) pairs 
            residue_ids = []
            chain_ids = set()
            for pocket_residue in pocket_structure.get_residues():
                chain_id = pocket_residue.get_parent().id
                _, residue_id, _ = pocket_residue.id
                resname = pocket_residue.resname

                # residue_ids.append((resname, int(residue_id)))
                # residue_ids.append((chain_id, int(residue_id)))

                # (chain_id, resname, residue_id) tuple
                residue_ids.append((chain_id, resname, residue_id))

                chain_ids.add(chain_id)

            # use residue_ids to define sequence (should be ordered by residue ID)
            chain_to_residue = {}
            for chain_id, resname, residue_id in residue_ids:
                if chain_id not in chain_to_residue:
                    chain_to_residue[chain_id] = []
                chain_to_residue[chain_id].append((resname, residue_id))

            chain_to_sequence = {}
            for chain_id, chain_residues in chain_to_residue.items():
                # sort by residue id
                chain_residues = sorted(chain_residues, key=lambda residue: residue[1])
                # add to chain_to_sequence
                chain_to_sequence[chain_id] = "".join(
                    (
                        AA3_TO_AA1[resname] if resname in AA3_TO_AA1 else "X" # also taken from PyBioMed.PyGetMol.GetProtein
                        for resname, residue_id in chain_residues
                    )
                )

            # convert to list
            chain_ids = sorted(chain_ids)

            fpocket_target_data[pocket_number] = {
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "size_x": size_x,
                "size_y": size_y,
                "size_z": size_z,
                **pocket_data,
                "residue_ids": residue_ids,
                "chain_ids": chain_ids,
                "chain_to_sequence": chain_to_sequence,
            }

    if len(fpocket_target_data) == 0:
        # no pockets found 
        print ("Fpocket failed to identify targets for", target_identifier)
        fpocket_target_data[None] = {
            "center_x": None,
            "center_y":None,
            "center_z": None,
            "pocket_score": None,
            "druggability_score": None,
            "number_of_alpha_spheres": None,
            "total_sasa": None,
            "polar_sasa": None,
            "apolar_sasa": None,
            "volume": None,
            "mean_local_hydrophobic_density": None,
            "mean_alpha_sphere_radius": None,
            "mean_alp_sph_solvent_access": None,
            "apolar_alpha_sphere_proportion": None,
            "hydrophobicity_score": None,
            "volume_score": None,
            "polarity_score": None,
            "charge_score": None,
            "proportion_of_polar_atoms": None,
            "alpha_sphere_density": None,
            "cent_of_mass_alpha_sphere_max_dist": None,
            "flexibility": None,
            "residue_ids": None,
            "chain_ids": None,
            "chain_to_sequence": None,
        }

   

    if delete_output_dir:
        delete_directory(target_fpocket_output_dir, verbose=verbose)

    return fpocket_target_data

# def collate_fpocket_data(
#     targets,
#     output_filename,
#     output_dir,
#     min_pocket_score=0,
#     ):

#     if os.path.exists(output_filename):
#         print (output_filename, "ALREADY EXISTS -- LOADING IT")
#         fpocket_data = load_json(output_filename)
#     else:

#         print ("RUNNNING FPOCKET FOR TARGETS", targets, "USING MIN_SCORE", min_pocket_score)

#         # os.makedirs(FPOCKET_OUT, exist_ok=True)

#         fpocket_data = {}

#         with Pool(processes=POCKET_ANALYSIS_N_PROC) as p:
            
#             fpocket_data_all_targets = p.map(
#                 partial(
#                     run_fpocket_and_collate_single_target,
#                     output_dir=output_dir,
#                     min_pocket_score=min_pocket_score,
#                     existing_pdb_filename=None,   
#                 ),
#                 targets,
#             )

#             for fpocket_data_target in fpocket_data_all_targets:
#                 fpocket_data.update(fpocket_data_target)
            
#         print ("WRITING FPOCKET DATA TO", output_filename)
#         write_json(fpocket_data, output_filename)
    
#     return fpocket_data

if __name__ == "__main__":

    fpocket_output = run_fpocket_and_collate_single_target(
        target_identifier="3ETO",
        output_dir="fpocket_out",
        min_pocket_score=None,
        existing_pdb_filename="3ETO.pdb",
        read_features_from_pocket_pdbs=True,
        delete_output_dir=False,
        verbose=True,
    )

    write_json(fpocket_output, "FPOCKET_OUT.json")

    # pocket_data = read_fpocket_info_file("fpocket_out_test/cavities/FPOCKET_out/3PP1_out/3PP1_info.txt")[10]

    # for key in pocket_data:
    #     print (f"\"{key}\": None,")