
if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import os
import copy

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

from itertools import product, combinations

# from utils.molecules.openbabel_utils import obabel_convert
from utils.molecules.pdb_utils import (
    get_number_of_models_in_pdb_file,
    # identify_centre_of_mass, 
    # get_bounding_box_size, 
    get_completeness_from_pdb_file, 
    get_resolution_from_pdb_file, 
    get_r_value_free_from_pdb_file,
    get_r_value_observed_from_pdb_file,
    get_is_mutant_from_pdb_file,
    get_is_engineered_from_pdb_file,
    remove_all_hetero_residues_using_biopython,
    get_all_natural_ligands_from_pdb_file,
    select_chains_from_pdb_file,
    get_number_of_atoms_in_pdb_file,
    download_pdb_structure_using_pdb_fetch,
    write_single_model,
    get_all_chain_ids_in_a_PDB_file,
    get_cofactors_for_accessions,
    define_target_bounding_box_using_biopython,
)
from utils.molecules.openbabel_utils import obabel_convert, smiles_to_3D
from utils.molecules.pymol_utils import convert_to_ligand, create_complex_with_pymol, cleanup_with_pymol
from utils.io.io_utils import (
    copy_file, 
    delete_directory, 
    delete_file, 
    gunzip_file, 
    gzip_file, 
    load_compressed_pickle, 
    load_json, 
    write_compressed_pickle, 
    write_json,
    get_token,
    sanitise_filename,
    read_smiles,
)
from utils.rank_aggregation import perform_rank_aggregation
from utils.molecules.protonation_utils import protonate_pdb
from utils.alphafold_utils import download_alphafold_structure

from cobdock.binding_site_identification.fpocket.fpocket_utils import run_fpocket_and_collate_single_target
from cobdock.binding_site_identification.p2rank.p2rank_utils import run_p2rank_and_collate_single_target

from cobdock.docking.vina.vina_utils import VINA_N_PROC, prepare_ligand_for_vina, prepare_target_for_vina, VINA_VARIANTS
from cobdock.docking.plants.plants_utils import PLANTS_N_PROC, prepare_for_plants
from cobdock.docking.galaxydock.galaxydock_utils import GALAXYDOCK_N_PROC, prepare_ligand_for_galaxydock, prepare_target_for_galaxydock

PREPARE_TARGETS_ROOT_DIR = os.path.join(
    "data",
    "prepared_targets",
)
os.makedirs(PREPARE_TARGETS_ROOT_DIR, exist_ok=True)

PREPARE_TARGETS_N_PROC = int(os.environ.get("PREPARE_TARGETS_N_PROC", default=1))


def predocking_ligand_preparation(
    ligands_to_targets: dict,
    submitted_ligand_structure_files: dict,
    ligand_structure_output_dir: str,
    ligand_pH: float = 7.4,
    molecule_identifier_key: str = "molecule_id",
    smiles_key: str = "smiles",
    verbose: bool = True,
    ):

    # clean up supplied ligand names
    ligands_to_targets_clean = {}
    for ligand_id, ligand_data in ligands_to_targets.items():

        ligand_id_clean = sanitise_filename(ligand_id, max_length=100)
        if ligand_id_clean in ligands_to_targets_clean:

            # add a wart
            ligand_id_clean += get_token(5)

        ligands_to_targets_clean[ligand_id_clean] = ligand_data

    ligands_to_targets = ligands_to_targets_clean

    ligand_structure_filename_keys = ("structure_filename", "pdb_filename", )

    # add/update location of structure files (only required when files are received from client)
    if isinstance(submitted_ligand_structure_files, dict) and len(submitted_ligand_structure_files) > 0:

        submitted_ligand_structure_files = {
            sanitise_filename(ligand_id): ligand_filename
            for ligand_id, ligand_filename in submitted_ligand_structure_files.items()
        }

        for ligand_id, ligand_data in ligands_to_targets.items():
            if verbose:
                print ("Processing filename for ligand", ligand_id)
            # ligand_id_sanitised = sanitise_filename(ligand_id)
            if ligand_id in submitted_ligand_structure_files:
                # update structure file location 
                ligand_data[ligand_structure_filename_keys[0]] = submitted_ligand_structure_files[ligand_id]

    # process any ligands that have SMILES strings under the "smiles" key in ligands_to_targets
    # by generating a separate PDB file for each one
    all_ligand_smiles = []
    for ligand_id, ligand_data in ligands_to_targets.items():
        
        # skip ligand if no "smiles" key exists
        if "smiles" not in ligand_data:
            continue

        # skip ligand if it already has a structure
        has_ligand_structure_filename_key = False
        for ligand_structure_filename_key in ligand_structure_filename_keys:
            if ligand_structure_filename_key in ligand_data:
                has_ligand_structure_filename_key = True
                break
        if has_ligand_structure_filename_key:
            continue

        # add SMILES to all_ligand_smiles 
        ligand_smiles = ligand_data["smiles"]
        all_ligand_smiles.append({
            molecule_identifier_key: ligand_id, 
            smiles_key: ligand_smiles,
        })

    if len(all_ligand_smiles) > 0:
        # map smiles to PDB files
        ligand_pdb_filenames = smiles_to_3D(
            supplied_molecules=all_ligand_smiles,
            output_dir=ligand_structure_output_dir,
            output_format="pdb",
            smiles_key=smiles_key,
            molecule_identifier_key=molecule_identifier_key,
            pH=ligand_pH,
            verbose=verbose,
        )
        # update ligands_to_targets with pdb_filenames
        for ligand_id, ligand_pdb_filename in ligand_pdb_filenames.items():
            if ligand_id in ligands_to_targets:
                ligands_to_targets[ligand_id]["pdb_filename"] = ligand_pdb_filename


    # ensure all ligands have a PDB file and convert other filetypes if necessary
    if verbose:
        print ("Ensuring all ligands are described in the PDB format and removing any ligands unsuitable for run")

    # maintain a list of ligands that have issues, to remove before the run begins
    ligands_to_drop = set()

    # prepare ligand structures
    for ligand_id, ligand_data in ligands_to_targets.items():
        
        has_ligand_structure_filename_key = False
        for filename_key in ligand_structure_filename_keys:
            if filename_key in ligand_data:
                ligand_structure_filename = ligand_data[filename_key] 
                if ligand_structure_filename is not None:
                    has_ligand_structure_filename_key = True
                    break
        if not has_ligand_structure_filename_key:
            if verbose:
                print ("Structure filename missing for ligand", ligand_id)
            ligands_to_drop.add(ligand_id)
            continue

        # some simple attempt at file format conversion to PDB
        stem, ext = os.path.splitext(ligand_structure_filename)
        # drop .
        ext = ext.replace(".", "")

        # handle conversion to SMILES
        if "smiles" not in ligand_data:
            ligand_smiles_filename = os.path.join(ligand_structure_output_dir, f"{ligand_id}.smi")
            if ext not in {"smi", "smiles", "txt"}:
                # convert from 3D to 2D
                ligand_smiles_filename = obabel_convert(
                    input_format=ext,
                    input_filename=ligand_structure_filename,
                    output_format="smi",
                    output_filename=ligand_smiles_filename,
                    title=ligand_id,
                    add_hydrogen=True,
                    verbose=verbose,
                )
            else:
                # already in smiles format, copy it
                ligand_smiles_filename = copy_file(
                    ligand_structure_filename,
                    ligand_smiles_filename,
                    verbose=verbose,
                )
            ligand_smiles = read_smiles(
                ligand_smiles_filename,
                remove_invalid_molecules=True,
                return_list=True, 
                molecule_identifier_key=molecule_identifier_key,
                smiles_key=smiles_key,
                verbose=verbose)
            if len(ligand_smiles) > 0:
                ligand_data["smiles"] = ligand_smiles[0][smiles_key]

        # handle conversion to PDB
        ligand_pdb_filename = os.path.join(ligand_structure_output_dir, f"{ligand_id}.pdb")
        if not os.path.exists(ligand_pdb_filename):
            if ext in {"mol", "mol2", "sdf", "pdb", }:
                ligand_pdb_filename = obabel_convert(
                    input_format=ext,
                    input_filename=ligand_structure_filename,
                    output_format="pdb",
                    output_filename=ligand_pdb_filename,
                    pH=ligand_pH,
                    gen_3d=ext == "sdf", # ensure 3D
                    title=ligand_id,
                    verbose=verbose,
                )
            elif ext in {"txt", "smi", "smiles"}:
                ligand_pdb_filename_map = smiles_to_3D(
                    supplied_molecules=ligand_structure_filename,
                    output_dir=ligand_structure_output_dir,
                    desired_output_filename=ligand_pdb_filename,
                    output_format="pdb",
                    add_hydrogen=True,
                    pH=ligand_pH,
                    overwrite=True,
                    verbose=verbose,
                )
                # index into dict using key
                key = list(ligand_pdb_filename_map)[0]
                ligand_pdb_filename = ligand_pdb_filename_map[key]
            else:
                ligand_pdb_filename = None
        
        if ligand_pdb_filename is None: # no value for file
            if verbose:
                print ("ligand_pdb_filename is None for ligand", ligand_id)
            ligands_to_drop.add(ligand_id)
        elif not os.path.exists(ligand_pdb_filename): # file does not exist
            if verbose:
                print (ligand_pdb_filename, "does not exist")
            ligands_to_drop.add(ligand_id)
        elif os.stat(ligand_pdb_filename).st_size == 0: # empty file
            if verbose:
                print (ligand_pdb_filename, "is empty")
            ligands_to_drop.add(ligand_id)
        else:
            # update pdb filename
            
            # ensure all residues are labelled as UNL
            ligand_pdb_filename = convert_to_ligand(
                input_filename=ligand_pdb_filename,
                output_filename=ligand_pdb_filename,
                verbose=verbose,
            )

            ligand_data["pdb_filename"] = ligand_pdb_filename

    # delete ligands with no structures or invalid for other reasons
    for ligand_to_drop in ligands_to_drop:
        if verbose:
            print ("Dropping ligand", ligand_to_drop)
        del ligands_to_targets[ligand_to_drop]

    return ligands_to_targets

def predocking_target_preparation_uniprot_target_input(
    ligands_to_targets: dict,
    output_dir: str,
    number_of_pdb: int = 1,
    use_alphafold: bool = True,
    allow_mutant: bool = True, 
    allow_engineered: bool = True,   
    prefix_size: int = 3,
    consider_prefix_in_reranking: bool = True,
    verbose: bool = True,
    ):
    
    # all targets in ligands to targets
    all_uniprot_targets_in_ligands_to_targets = {
        accession
        for ligand_id, ligand_data in ligands_to_targets.items()
        for accession in ligand_data["targets"]
    }

    # map from accession to pdb id using file (also includes chains)
    # format is accession -> pdb_id -> list of chains
    uniprot_to_pdb_id_data = map_uniprot_accession_to_pdb_id(
        accessions=all_uniprot_targets_in_ligands_to_targets,
        return_single_chain_only=True, # take first chain only?
        # keep_maximum_sequence_length_only=True, # only largest chains possible?
        keep_maximum_sequence_length_only=False, # all structures considered?
        verbose=verbose,
    )

    # can handle multiple chains from same PDB if needed
    # convert to upper case because there is duplication (P25025/6KVA_B, P25025/6KVA_b)
    uniprot_to_pdb_id_chains = {
        accession: {
            f"{pdb_id}_{chain}".upper(): [chain.upper()]
            for pdb_id, pdb_chains in accession_data.items()
            for chain in pdb_chains # all chains 
        }
        for accession, accession_data in uniprot_to_pdb_id_data.items()
    }

    # write ranks to file
    pdb_id_ranks_for_all_uniprot_accessions_filename = os.path.join(
        output_dir, 
        "pdb_id_ranks_for_all_uniprot_accessions.pkl.gz")
    # load ranks file if it exists
    if os.path.exists(pdb_id_ranks_for_all_uniprot_accessions_filename):
        pdb_id_ranks_for_all_uniprot_accessions = load_compressed_pickle(
            pdb_id_ranks_for_all_uniprot_accessions_filename, 
            verbose=verbose)
    else:
        pdb_id_ranks_for_all_uniprot_accessions = {}

    # list of new accessions to identify PDB structures for
    accessions_to_rank_structures_for = {
        accession
        for accession in uniprot_to_pdb_id_chains
        if accession not in pdb_id_ranks_for_all_uniprot_accessions
    }

    if len(accessions_to_rank_structures_for) > 0:

        uniprot_to_pdb_id_chains_to_rank = {
            accession: pdb_id_chain 
            for accession, pdb_id_chain in uniprot_to_pdb_id_chains.items()
            if accession in accessions_to_rank_structures_for
        }

        # filter and rank PDB IDS for each unique Uniprot accession
        pdb_id_ranks_for_all_uniprot_accessions_new = rank_all_pdb_structures_for_uniprot_accessions(
            uniprot_to_pdb_id=uniprot_to_pdb_id_chains_to_rank,
            allow_multi_model=True, # ?
            allow_mutant=allow_mutant,
            allow_engineered=allow_engineered,
            prefix_size=prefix_size,
            consider_prefix_in_reranking=consider_prefix_in_reranking,
            verbose=verbose,
        )

        # update pdb_id_ranks_for_all_uniprot_accessions
        pdb_id_ranks_for_all_uniprot_accessions.update(pdb_id_ranks_for_all_uniprot_accessions_new)

        # write pdb_id_ranks_for_all_uniprot_accessions to file 
        write_compressed_pickle(
            pdb_id_ranks_for_all_uniprot_accessions, 
            pdb_id_ranks_for_all_uniprot_accessions_filename, 
            verbose=verbose)
        
    
    # iterate over ligands and update "targets"
    for ligand_id, ligand_data in ligands_to_targets.items():

        # original target list
        ligand_targets = ligand_data["targets"]

        if verbose:
            print ("Selecting", number_of_pdb, "best PDB structures for each target for ligand", ligand_id)

        # select top ranked pdb ids
        # update ligand target list
        ligand_uniprot_targets_with_pdb = dict()

        # iterate over uniprot targets and select pdb ids
        for accession in ligand_targets:

            if verbose:
                print ("Processing UniProt target", accession, "for ligand", ligand_id)

            selected_pdb_ids_for_uniprot_id = dict()
            
            # check that PDB structures exist in PDB database for current accession
            if accession in pdb_id_ranks_for_all_uniprot_accessions: 

                # sort PDB structures for current accession using `reranked_rank`
                sorted_pdb_ids_for_current_uniprot_id = sorted(
                    pdb_id_ranks_for_all_uniprot_accessions[accession],
                    key=lambda pdb_id: pdb_id_ranks_for_all_uniprot_accessions[accession][pdb_id]["reranked_rank"]
                ) 
                for pdb_id in sorted_pdb_ids_for_current_uniprot_id:
                    
                    # break out of loop if enough pdb ids have been found
                    if number_of_pdb is not None and len(selected_pdb_ids_for_uniprot_id) >= number_of_pdb:
                        break 
                    selected_pdb_ids_for_uniprot_id[pdb_id] = pdb_id_ranks_for_all_uniprot_accessions[accession][pdb_id]

            # update ligand_uniprot_targets_with_pdb with PDB IDs selected for current accession
            ligand_uniprot_targets_with_pdb[accession] = selected_pdb_ids_for_uniprot_id

        # add alphafold-predicted crystal structures
        for accession in ligand_targets:
            if use_alphafold == 0:
                continue
            # only use alphafold if no structures are found?
            if accession not in ligand_uniprot_targets_with_pdb: # should not be missing
                ligand_uniprot_targets_with_pdb[accession] = {}
            # only add alphafold if no PDB structures are found
            # if use_alphafold == 1 or (use_alphafold == 2 and len(ligand_uniprot_targets_with_pdb[accession]) == 0):
            # only add alphafold if not enough PDB structures are found
            if use_alphafold == 1 or (use_alphafold == 2 and len(ligand_uniprot_targets_with_pdb[accession]) < number_of_pdb): 
                ligand_uniprot_targets_with_pdb[accession][f"alphafold-{accession}_A"] = None # value currently required for consistency  

        # replace ligand_targets variable
        ligand_targets = ligand_uniprot_targets_with_pdb

        # overwrite targets with pdb targets
        ligand_data["targets"] = ligand_targets

    return ligands_to_targets

def predocking_target_preparation_pdb_target_input(
    ligands_to_targets: dict,
    max_num_chains: int = 2, # max number of chains to keep in a single crystal structure
    verbose: bool = True,
    ):

    # all targets in ligands to targets
    all_targets = {
        accession
        for ligand_id, ligand_data in ligands_to_targets.items()
        for accession in ligand_data["targets"]
    }

    # get accessions for all requested chains of given PDB structures
    # can accept PDB ID and PDB ID_chain forms
    pdb_id_to_uniprot_accession = map_pdb_id_to_uniprot_accession(
        pdb_ids=all_targets, # keyed by PDB ID only
        verbose=verbose,
    )

    # extract accessions from pdb_id_to_uniprot_accession and consider chain combinations if necessary
    for pdb_id, all_chains_for_pdb_id in pdb_id_to_uniprot_accession.items():

        # include all single chains and combinations (upto max_num_chains)
        all_chain_combinations_to_keep = {
            "".join(sorted(c))
            for i in range(max_num_chains)
            for c in combinations(all_chains_for_pdb_id, i+1)
        }

        # otherwise, remove unnecessary chains
        accessions_to_keep = {}

        # iterate over chains to keep and select uniprot accessions
        for chain_combination_to_keep in all_chain_combinations_to_keep:

            # get accessions for each chain in chain_combination_to_keep
            chain_to_keep_accessions = {
                accession
                for c in chain_combination_to_keep
                for accession in all_chains_for_pdb_id[c]
            }

            accession_key = "-".join(sorted(chain_to_keep_accessions))
            if accession_key not in accessions_to_keep:
                accessions_to_keep[accession_key] = set()

            accessions_to_keep[accession_key].add(chain_combination_to_keep)

        # update pdb_id_to_uniprot_accession
        pdb_id_to_uniprot_accession[pdb_id] = accessions_to_keep

    # iterate ovver ligands and  update "targets"
    for ligand_id, ligand_data in ligands_to_targets.items():
        
        # original target list
        ligand_targets = ligand_data["targets"]
        
        # convert original list to accession -> pdb id
        ligand_uniprot_to_pdb_map = {}

        # ligand targets is a list of PDB IDs
        for pdb_id in ligand_targets:
            
            pdb_id = pdb_id.upper()
            
            if pdb_id in pdb_id_to_uniprot_accession:

                
                # get accessions for current PDB ID
                accessions_for_pdb_id = pdb_id_to_uniprot_accession[pdb_id]

                # remove _chain_id
                if "_" in pdb_id:
                    pdb_id = pdb_id.split("_")[0]

                # iterate over accessions and corresponding chains 
                for accession, pdb_chains in accessions_for_pdb_id.items():
                    # initialise accession for current ligand
                    if accession not in ligand_uniprot_to_pdb_map:
                        ligand_uniprot_to_pdb_map[accession] = set()
                    # get chain(s) of current pdb corresponding to current accession(s)
                    for chain_id in pdb_chains:

                        ligand_uniprot_to_pdb_map[accession].add(f"{pdb_id}_{chain_id}")

            else:
                # pdb ID cannot be mapped to accession number 
                accession = pdb_id # use submitted name
                # add all chains 
                # initialise accession for current ligand
                if accession not in ligand_uniprot_to_pdb_map:
                    ligand_uniprot_to_pdb_map[accession] = set()
                ligand_uniprot_to_pdb_map[accession].add(pdb_id)
                
        ligands_to_targets[ligand_id]["targets"] = ligand_uniprot_to_pdb_map

        # raise Exception(ligand_uniprot_to_pdb_map)

    return ligands_to_targets

def predocking_target_preparation_preprocessed_target_input(
    ligands_to_targets: dict,
    verbose: bool = True,
    ):

    for ligand_id, ligand_data in ligands_to_targets.items():

        ligand_data["targets"] = {
            f"preprocessed-{target}": target 
            for target in ligand_data["targets"]
        }

    return ligands_to_targets 

def predocking_preparation(
    ligands_to_targets: dict,
    output_dir: str,
    submitted_ligand_structure_files: dict = {}, # mapping from ligand name to .pdb file
    submitted_target_pdb_files: dict = {}, # mapping from target name to .pdb file
    map_uniprot_to_pdb: bool = False, 
    number_of_pdb: int = 1,
    allow_mutant: bool = True, # mutant pdb file targets 
    allow_engineered: bool = True, # engineered pdb file targets  
    bounding_box_scale: float = 1., 
    max_bounding_box_size: float = None,
    prefix_size: int = 3,
    consider_prefix_in_reranking: bool = True,
    compute_voxel_locations: bool = False,
    fill_pdb_list_with_similar_targets: bool = False,
    determine_natural_ligands: bool = True,
    target_pH: float = 7.4,
    ligand_pH: float = 7.4,
    run_fpocket: bool = True,
    run_fpocket_old: bool = False,
    run_p2rank: bool = True,
    use_alphafold: bool = False,
    max_pockets_to_keep: int = 1,
    keep_cofactors: bool = False,
    molecule_identifier_key: str = "molecule_id",
    smiles_key: str = "smiles",
    verbose: bool = True,
    ):

    if not isinstance(ligands_to_targets, dict):
        print ("ligands_to_targets is not a dictionary, returning", {})
        return {}

    if submitted_ligand_structure_files is None:
        submitted_ligand_structure_files = {}

    if verbose:
        print ("Beginning all pre-docking preparation")
        print ("Making output directory", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ligand_structure_output_dir = os.path.join(output_dir, "ligand_structures")
    os.makedirs(ligand_structure_output_dir, exist_ok=True)

    # cleanup and prepare ligands 
    ligands_to_targets = predocking_ligand_preparation(
        ligands_to_targets=ligands_to_targets,
        submitted_ligand_structure_files=submitted_ligand_structure_files,
        ligand_structure_output_dir=ligand_structure_output_dir,
        ligand_pH=ligand_pH,
        molecule_identifier_key=molecule_identifier_key,
        smiles_key=smiles_key,
        verbose=verbose,
    )

    # handle target preparation
    if map_uniprot_to_pdb == 1:
        # uniprot target input
        ligands_to_targets = predocking_target_preparation_uniprot_target_input(
            ligands_to_targets=ligands_to_targets,
            output_dir=output_dir,
            number_of_pdb=number_of_pdb,
            use_alphafold=use_alphafold,
            allow_mutant=allow_mutant,
            allow_engineered=allow_engineered,
            prefix_size=prefix_size,
            consider_prefix_in_reranking=consider_prefix_in_reranking,
            verbose=verbose,
        )
    elif map_uniprot_to_pdb == 0:
        # PDB target input
        ligands_to_targets = predocking_target_preparation_pdb_target_input(
            ligands_to_targets=ligands_to_targets,
            max_num_chains=2,
            verbose=verbose,
        )
    else:
        # prepared targets 
        ligands_to_targets = predocking_target_preparation_preprocessed_target_input(
            ligands_to_targets=ligands_to_targets,
            verbose=verbose,
        )

    # extract targets to prepare from ligands_to_targets
    all_targets_to_prepare = {}

    # iterate over all ligands and add targets to set of all_targets_to_prepare
    for ligand_id, ligand_data in ligands_to_targets.items():
        # update all_targets_to_prepare with targets of current ligand
        for accession, accession_targets in ligand_data["targets"].items():
            if accession not in all_targets_to_prepare:
                all_targets_to_prepare[accession] = set()
            for pdb_id in accession_targets:
                all_targets_to_prepare[accession].add(pdb_id)

    # base directory to write decompressed prepared targets to
    prepared_targets_dir = os.path.join(output_dir, "prepared_targets")

    # prepare all targets in all_pdb_targets set (loads data if exists)
    prepared_targets = prepare_all_targets(
        accession_to_pdb=all_targets_to_prepare,
        decompress_to_directory=prepared_targets_dir,
        existing_target_filenames=submitted_target_pdb_files, # pass in any submitted targets
        bounding_box_scale=bounding_box_scale,
        max_bounding_box_size=max_bounding_box_size,
        allow_multi_model=True, # ?
        allow_mutant=allow_mutant,
        allow_engineered=allow_engineered,
        protonate_to_pH=target_pH,
        run_fpocket=run_fpocket,
        run_fpocket_old=run_fpocket_old,
        run_p2rank=run_p2rank,
        compute_voxel_locations=compute_voxel_locations,
        max_pockets_to_keep=max_pockets_to_keep,
        keep_cofactors=keep_cofactors,
        verbose=verbose,
    )

    # for accession, accession_data in prepared_targets.items():
    #     for pdb_id, pdb_id_data in accession_data.items():
    #         if "natural_ligands" in pdb_id_data:
    #             del pdb_id_data["natural_ligands"]

    # write_json(prepared_targets, "prepared_targets.json")
    # raise Exception

    # add pdb id ranks if map_uniprot_to_pdb
    # if map_uniprot_to_pdb == 1:
    #     for accession in prepared_targets:
    #         if accession not in pdb_id_ranks_for_all_uniprot_accessions:
    #             continue
    #         for pdb_id, pdb_id_data in prepared_targets[accession].items():
    #             if pdb_id not in pdb_id_ranks_for_all_uniprot_accessions[accession]:
    #                 continue
    #             # add rank of PDB ID for current accession
    #             pdb_id_data["pdb_id_rank_for_accession"] = pdb_id_ranks_for_all_uniprot_accessions[accession][pdb_id]["reranked_rank"]
            
    # update ligand_to_targets with target data
    for ligand_id, ligand_data in ligands_to_targets.items():
        if verbose:
            print ("Adding prepared_targets for ligand", ligand_id)

        # initialise prepared_targets dict for current ligand
        ligand_prepared_targets = {}
        for accession, ligand_accession_pdb_ids in ligand_data["targets"].items():
            # skip accession if it could not be prepared
            if accession not in prepared_targets:
                continue
            prepared_targets_for_accession = prepared_targets[accession]
            ligand_accession_pdb_ids = sorted(ligand_accession_pdb_ids)
            
            # check for accession=pdb_id
            if accession == ligand_accession_pdb_ids[0]:
                # add all chains
                ligand_prepared_targets[accession] = prepared_targets_for_accession
            else:
                ligand_prepared_targets_for_accession = {}

                for ligand_accession_pdb_id in ligand_accession_pdb_ids:
                    # if ligand_accession_pdb_id.startswith("alphafold-"):
                    #     # append chain
                    #     ligand_accession_pdb_id += "_A"
                    ligand_accession_pdb_id = ligand_accession_pdb_id.upper()
                    # if ligand_accession_pdb_id in prepared_targets_for_accession:
                    # account for potential appending of chain(s)

                    for prepared_target_for_accession in prepared_targets_for_accession:
                        if prepared_target_for_accession.startswith(ligand_accession_pdb_id):
                            ligand_prepared_targets_for_accession[prepared_target_for_accession] = prepared_targets_for_accession[prepared_target_for_accession]
                # add all prepared pdb IDs for accession to ligand_prepared_targets
                ligand_prepared_targets[accession] = ligand_prepared_targets_for_accession

        # add "prepared_targets" key to ligand_data
        # ensure copy
        ligand_data["prepared_targets"] = copy.deepcopy(ligand_prepared_targets)

        # delete "targets" key to save memory
        del ligand_data["targets"]

    # write_json(ligands_to_targets, "ligands_to_targets.json")
    # raise Exception

    return ligands_to_targets, prepared_targets_dir, ligand_structure_output_dir


# def predocking_preparation(
#     ligands_to_targets: dict,
#     output_dir: str,
#     submitted_ligand_structure_files: dict ={}, # mapping from ligand name to .pdb file
#     submitted_target_pdb_files: dict ={}, # mapping from target name to .pdb file
#     map_uniprot_to_pdb: bool = False, 
#     number_of_pdb: int = 1,
#     allow_mutant: bool = True, # mutant pdb file targets 
#     allow_engineered: bool = True, # engineered pdb file targets  
#     bounding_box_scale: float = 1., 
#     max_bounding_box_size: float = None,
#     prefix_size: int = 3,
#     consider_prefix_in_reranking: bool = True,
#     compute_voxel_locations: bool = False,
#     fill_pdb_list_with_similar_targets: bool = False,
#     determine_natural_ligands: bool = True,
#     target_pH: float = 7.4,
#     ligand_pH: float = 7.4,
#     run_fpocket: bool = True,
#     run_fpocket_old: bool = False,
#     run_p2rank: bool = True,
#     use_alphafold: bool = False,
#     max_pockets_to_keep: int = 1,
#     keep_cofactors: bool = False,
#     molecule_identifier_key: str = "molecule_id",
#     smiles_key: str = "smiles",
#     verbose: bool = True,
#     ):

#     if not isinstance(ligands_to_targets, dict):
#         print ("ligands_to_targets is not a dictionary, returning", {})
#         return {}

#     if submitted_ligand_structure_files is None:
#         submitted_ligand_structure_files = {}

#     if verbose:
#         print ("Beginning all pre-docking preparation")
#         print ("Making output directory", output_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     ligand_structure_output_dir = os.path.join(output_dir, "ligand_structures")
#     os.makedirs(ligand_structure_output_dir, exist_ok=True)

#     # clean up supplied ligand names
#     ligands_to_targets = {
#         sanitise_filename(ligand_id): ligand_data
#         for ligand_id, ligand_data in ligands_to_targets.items()
#     }

#     ligand_structure_filename_keys = ("structure_filename", "pdb_filename", )

#     # add/update location of structure files (only required when files are received from client)
#     if isinstance(submitted_ligand_structure_files, dict) and len(submitted_ligand_structure_files) > 0:

#         submitted_ligand_structure_files = {
#             sanitise_filename(ligand_id): ligand_filename
#             for ligand_id, ligand_filename in submitted_ligand_structure_files.items()
#         }

#         for ligand_id, ligand_data in ligands_to_targets.items():
#             if verbose:
#                 print ("Processing filename for ligand", ligand_id)
#             # ligand_id_sanitised = sanitise_filename(ligand_id)
#             if ligand_id in submitted_ligand_structure_files:
#                 # update structure file location 
#                 ligand_data[ligand_structure_filename_keys[0]] = submitted_ligand_structure_files[ligand_id]

#     # process any ligands that have SMILES strings under the "smiles" key in ligands_to_targets
#     # by generating a separate PDB file for each one
#     all_ligand_smiles = []
#     for ligand_id, ligand_data in ligands_to_targets.items():
        
#         # skip ligand if no "smiles" key exists
#         if "smiles" not in ligand_data:
#             continue

#         # skip ligand if it already has a structure
#         has_ligand_structure_filename_key = False
#         for ligand_structure_filename_key in ligand_structure_filename_keys:
#             if ligand_structure_filename_key in ligand_data:
#                 has_ligand_structure_filename_key = True
#                 break
#         if has_ligand_structure_filename_key:
#             continue

#         # add SMILES to all_ligand_smiles 
#         ligand_smiles = ligand_data["smiles"]
#         all_ligand_smiles.append({
#             molecule_identifier_key: ligand_id, 
#             smiles_key: ligand_smiles,
#         })

#     if len(all_ligand_smiles) > 0:
#         # map smiles to PDB files
#         ligand_pdb_filenames = smiles_to_3D(
#             supplied_molecules=all_ligand_smiles,
#             output_dir=ligand_structure_output_dir,
#             output_format="pdb",
#             smiles_key=smiles_key,
#             molecule_identifier_key=molecule_identifier_key,
#             pH=ligand_pH,
#             verbose=verbose,
#         )
#         # update ligands_to_targets with pdb_filenames
#         for ligand_id, ligand_pdb_filename in ligand_pdb_filenames.items():
#             if ligand_id in ligands_to_targets:
#                 ligands_to_targets[ligand_id]["pdb_filename"] = ligand_pdb_filename


#     # ensure all ligands have a PDB file and convert other filetypes if necessary
#     if verbose:
#         print ("Ensuring all ligands are described in the PDB format and removing any ligands unsuitable for run")

#     # maintain a list of ligands that have issues, to remove before the run begins
#     ligands_to_drop = set()

#     # prepare ligand structures
#     for ligand_id, ligand_data in ligands_to_targets.items():
        
#         has_ligand_structure_filename_key = False
#         for filename_key in ligand_structure_filename_keys:
#             if filename_key in ligand_data:
#                 ligand_structure_filename = ligand_data[filename_key] 
#                 if ligand_structure_filename is not None:
#                     has_ligand_structure_filename_key = True
#                     break
#         if not has_ligand_structure_filename_key:
#             if verbose:
#                 print ("Structure filename missing for ligand", ligand_id)
#             ligands_to_drop.add(ligand_id)
#             continue

#         # some simple attempt at file format conversion to PDB
#         stem, ext = os.path.splitext(ligand_structure_filename)
#         # drop .
#         ext = ext.replace(".", "")

#         # handle conversion to SMILES
#         if "smiles" not in ligand_data:
#             ligand_smiles_filename = os.path.join(ligand_structure_output_dir, f"{ligand_id}.smi")
#             if ext not in {"smi", "smiles", "txt"}:
#                 # convert from 3D to 2D
#                 ligand_smiles_filename = obabel_convert(
#                     input_format=ext,
#                     input_filename=ligand_structure_filename,
#                     output_format="smi",
#                     output_filename=ligand_smiles_filename,
#                     title=ligand_id,
#                     add_hydrogen=True,
#                     verbose=verbose,
#                 )
#             else:
#                 # already in smiles format, copy it
#                 ligand_smiles_filename = copy_file(
#                     ligand_structure_filename,
#                     ligand_smiles_filename,
#                     verbose=verbose,
#                 )
#             ligand_smiles = read_smiles(
#                 ligand_smiles_filename,
#                 remove_invalid_molecules=True,
#                 return_list=True, 
#                 molecule_identifier_key=molecule_identifier_key,
#                 smiles_key=smiles_key,
#                 verbose=verbose)
#             if len(ligand_smiles) > 0:
#                 ligand_data["smiles"] = ligand_smiles[0][smiles_key]

#         # handle conversion to PDB
#         ligand_pdb_filename = os.path.join(ligand_structure_output_dir, f"{ligand_id}.pdb")
#         if not os.path.exists(ligand_pdb_filename):
#             if ext in {"mol", "mol2", "sdf", "pdb", }:
#                 ligand_pdb_filename = obabel_convert(
#                     input_format=ext,
#                     input_filename=ligand_structure_filename,
#                     output_format="pdb",
#                     output_filename=ligand_pdb_filename,
#                     pH=ligand_pH,
#                     gen_3d=ext == "sdf", # ensure 3D
#                     title=ligand_id,
#                     verbose=verbose,
#                 )
#             elif ext in {"txt", "smi", "smiles"}:
#                 ligand_pdb_filename_map = smiles_to_3D(
#                     supplied_molecules=ligand_structure_filename,
#                     output_dir=ligand_structure_output_dir,
#                     desired_output_filename=ligand_pdb_filename,
#                     output_format="pdb",
#                     add_hydrogen=True,
#                     pH=ligand_pH,
#                     overwrite=True,
#                     verbose=verbose,
#                 )
#                 # index into dict using key
#                 key = list(ligand_pdb_filename_map)[0]
#                 ligand_pdb_filename = ligand_pdb_filename_map[key]
#             else:
#                 ligand_pdb_filename = None
        
#         if ligand_pdb_filename is None: # no value for file
#             if verbose:
#                 print ("ligand_pdb_filename is None for ligand", ligand_id)
#             ligands_to_drop.add(ligand_id)
#         elif not os.path.exists(ligand_pdb_filename): # file does not exist
#             if verbose:
#                 print (ligand_pdb_filename, "does not exist")
#             ligands_to_drop.add(ligand_id)
#         elif os.stat(ligand_pdb_filename).st_size == 0: # empty file
#             if verbose:
#                 print (ligand_pdb_filename, "is empty")
#             ligands_to_drop.add(ligand_id)
#         else:
#             # update pdb filename
            
#             # ensure all residues are labelled as UNL
#             ligand_pdb_filename = convert_to_ligand(
#                 input_filename=ligand_pdb_filename,
#                 output_filename=ligand_pdb_filename,
#                 verbose=verbose,
#             )

#             ligand_data["pdb_filename"] = ligand_pdb_filename

#     # delete ligands with no structures or invalid for other reasons
#     for ligand_to_drop in ligands_to_drop:
#         if verbose:
#             print ("Dropping ligand", ligand_to_drop)
#         del ligands_to_targets[ligand_to_drop]

#     # build set of unique targets 
#     # may be either PDB IDs or Uniprot accessions
#     all_targets = set()

#     for ligand_id in ligands_to_targets:
#         # uniprot targets for the ligand
#         ligand_targets = ligands_to_targets[ligand_id]["targets"]
#         all_targets.update(ligand_targets)


#     if map_uniprot_to_pdb == 1:

#         # map from accession to pdb id using file (also includes chains)
#         # format is accession -> pdb_id -> list of chains
#         uniprot_to_pdb_id_data = map_uniprot_accession_to_pdb_id(
#             accessions=all_targets,
#             # return_single_chain_only=True, # take first chain only?
#             return_single_chain_only=False, 
#             verbose=verbose,
#         )

#         # can handle multiple chains from same PDB if needed
#         # convert to upper case because there is duplication (P25025/6KVA_B, P25025/6KVA_b)
#         uniprot_to_pdb_id_chains = {
#             accession: {
#                 f"{pdb_id}_{chain['chain_id']}".upper(): [chain["chain_id"].upper()]
#                 for pdb_id, pdb_chains in accession_data.items()
#                 for chain in pdb_chains # all chains 
#                 # for chain in sorted(pdb_chains, key=lambda chain: (chain["sequence_length"], chain["chain_id"]))[:1] # first chain
#             }
#             for accession, accession_data in uniprot_to_pdb_id_data.items()
#         }
#         # sequence lengths 
#         uniprot_to_pdb_sequence_lengths = {
#             accession: {
#                f"{pdb_id}_{chain['chain_id']}".upper() : chain["sequence_length"]
#                 for pdb_id, pdb_chains in accession_data.items()
#                 for chain in pdb_chains # all chains 
#                 # for chain in sorted(pdb_chains, key=lambda chain: (chain["sequence_length"], chain["chain_id"]))[:1] # first chain
#             }
#             for accession, accession_data in uniprot_to_pdb_id_data.items()
#         }

#         # write ranks to file
#         pdb_id_ranks_for_all_uniprot_accessions_filename = os.path.join(
#             output_dir, 
#             "pdb_id_ranks_for_all_uniprot_accessions.pkl.gz")
#         if os.path.exists(pdb_id_ranks_for_all_uniprot_accessions_filename):
#             pdb_id_ranks_for_all_uniprot_accessions = load_compressed_pickle(pdb_id_ranks_for_all_uniprot_accessions_filename, verbose=verbose)
#         else:

#             pdb_id_ranks_for_all_uniprot_accessions = {}

#         accessions_to_rank_structures_for = {
#             accession
#             for accession in uniprot_to_pdb_id_chains
#             if accession not in pdb_id_ranks_for_all_uniprot_accessions
#         }

#         if len(accessions_to_rank_structures_for) > 0:

#             uniprot_to_pdb_id_chains_to_rank = {
#                 accession: pdb_id_chain 
#                 for accession, pdb_id_chain in uniprot_to_pdb_id_chains.items()
#                 if accession in accessions_to_rank_structures_for
#             }

#             # filter and rank PDB IDS for each unique Uniprot accession
#             pdb_id_ranks_for_all_uniprot_accessions_new = rank_all_pdb_structures_for_uniprot_accessions(
#                 uniprot_to_pdb_id=uniprot_to_pdb_id_chains_to_rank,
#                 sequence_lengths=uniprot_to_pdb_sequence_lengths,
#                 allow_multi_model=True, # ?
#                 allow_mutant=allow_mutant,
#                 allow_engineered=allow_engineered,
#                 prefix_size=prefix_size,
#                 consider_prefix_in_reranking=consider_prefix_in_reranking,
#                 verbose=verbose,
#             )

#             # add chains and update pdb_id_ranks_for_all_uniprot_accessions
#             for accession in pdb_id_ranks_for_all_uniprot_accessions_new:
#                 if accession not in uniprot_to_pdb_id_chains:
#                     continue

#                 # add access to pdb_id_ranks_for_all_uniprot_accessions
#                 if accession not in pdb_id_ranks_for_all_uniprot_accessions:
#                     pdb_id_ranks_for_all_uniprot_accessions[accession] = {}

#                 # iterate over PDB IDs and separate chains
#                 for pdb_id, pdb_id_data in pdb_id_ranks_for_all_uniprot_accessions_new[accession].items():

#                     if pdb_id not in uniprot_to_pdb_id_chains[accession]:
#                         continue
#                     pdb_id_data["chains"] = uniprot_to_pdb_id_chains[accession][pdb_id]

#             # update pdb_id_ranks_for_all_uniprot_accessions
#             pdb_id_ranks_for_all_uniprot_accessions.update(pdb_id_ranks_for_all_uniprot_accessions_new)

#             # write pdb_id_ranks_for_all_uniprot_accessions to file 
#             write_compressed_pickle(pdb_id_ranks_for_all_uniprot_accessions, pdb_id_ranks_for_all_uniprot_accessions_filename, verbose=verbose)

#     # map PDB -> uniprot for screening down the line
#     elif map_uniprot_to_pdb == 0:

#         # no mapping from accession to pdb
#         pdb_id_ranks_for_all_uniprot_accessions = {}

#         # all PDB IDs in all_targets
#         # map PDB ID to chain set to keep
#         pdb_id_to_chain_id = {}

#         # delete other chains if chains have been specified
#         for pdb_id in all_targets:

#             chain_id = None # keep all

#             if pdb_id.count("_") == 1:
#                 pdb_id, chain_id = pdb_id.split("_")
#             if pdb_id not in pdb_id_to_chain_id:
#                 pdb_id_to_chain_id[pdb_id] = set()
#             if chain_id is not None:
#                 pdb_id_to_chain_id[pdb_id].add(chain_id)
        
#         pdb_id_to_uniprot_accession = map_pdb_id_to_uniprot_accession(
#             pdb_ids=pdb_id_to_chain_id, # keyed by PDB ID only
#             verbose=verbose,
#         )

#         for pdb_id, existing_accessions_for_pdb_id in pdb_id_to_uniprot_accession.items():

#             chains_to_keep = pdb_id_to_chain_id[pdb_id]
#             if len(chains_to_keep) == 0: 
#                 # keep all, no need to delete chains 
#                 continue

#             # otherwise, remove unnecessary chains
#             accessions_to_keep = {}

#             for accession, accession_chains in existing_accessions_for_pdb_id.items():
#                 accession_chains = [ 
#                     accession_chain 
#                     for accession_chain in accession_chains
#                     if accession_chain["chain_id"] in chains_to_keep
#                 ]
#                 if len(accession_chains) > 0:
#                     accessions_to_keep[accession] = accession_chains

#             # update pdb_id_to_uniprot_accession
#             pdb_id_to_uniprot_accession[pdb_id] = accessions_to_keep

#     # else:
#     #     # no mapping 
#     #     target_mapping = {
#     #         target: target for target in all_targets
#     #     }

#     # identify all PDB targets
#     # change to uniprot -> pdb
#     all_targets_to_prepare = {}

#     # construct uniprot -> pdb map for preparation
#     for ligand_id in ligands_to_targets:

#         ligand_targets = ligands_to_targets[ligand_id]["targets"]

#         if map_uniprot_to_pdb == 1:

#             if verbose:
#                 print ("Selecting", number_of_pdb, "best PDB structures for each target for ligand", ligand_id)

#             # select top ranked pdb ids
#             # update ligand target list
#             ligand_uniprot_targets_with_pdb = dict()

#             # iterate over uniprot targets and select pdb ids
#             for accession in ligand_targets:

#                 if verbose:
#                     print ("Processing UniProt target", accession, "for ligand", ligand_id)

#                 selected_pdb_ids_for_uniprot_id = dict()
                
#                 # check that PDB structures exist in PDB database for current accession
#                 if accession in pdb_id_ranks_for_all_uniprot_accessions: 

#                     # sort PDB structures for current accession using `reranked_rank`
#                     sorted_pdb_ids_for_current_uniprot_id = sorted(
#                         pdb_id_ranks_for_all_uniprot_accessions[accession],
#                         key=lambda pdb_id: pdb_id_ranks_for_all_uniprot_accessions[accession][pdb_id]["reranked_rank"]
#                     ) 
#                     for pdb_id in sorted_pdb_ids_for_current_uniprot_id:
                        
#                         # break out of loop if enough pdb ids have been found
#                         if number_of_pdb is not None and len(selected_pdb_ids_for_uniprot_id) >= number_of_pdb:
#                             break 
#                         selected_pdb_ids_for_uniprot_id[pdb_id] = pdb_id_ranks_for_all_uniprot_accessions[accession][pdb_id]

#                 # update ligand_uniprot_targets_with_pdb with PDB IDs selected for current accession
#                 ligand_uniprot_targets_with_pdb[accession] = selected_pdb_ids_for_uniprot_id

#             # add alphafold-predicted crystal structures
#             # if use_alphafold:
                
#             for accession in ligand_targets:
#                 if use_alphafold == 0:
#                     continue
#                 # only use alphafold if no structures are found?
#                 if accession not in ligand_uniprot_targets_with_pdb: # should not be missing
#                     ligand_uniprot_targets_with_pdb[accession] = {}
#                 if use_alphafold == 1 or (use_alphafold == 2 and len(ligand_uniprot_targets_with_pdb[accession]) == 0): # only add alphafold if no PDB structures are found
#                 # if use_alphafold == 1 or (use_alphafold == 2 and len(ligand_uniprot_targets_with_pdb[accession]) < number_of_pdb): # only add alphafold if not enough PDB structures are found
#                     ligand_uniprot_targets_with_pdb[accession][f"alphafold-{accession}_A"] = None # value currently required for consistency  

#             # replace ligand_targets variable
#             ligand_targets = ligand_uniprot_targets_with_pdb

#             # overwrite targets with pdb targets
#             ligands_to_targets[ligand_id]["targets"] = ligand_targets

#         elif map_uniprot_to_pdb == 0:
#             # change to accession -> PDB ID format
#             ligand_uniprot_to_pdb = {}
#             # ligand targets is a list of PDB IDs
#             for pdb_id in ligand_targets:
                
#                 pdb_id = pdb_id.upper()

#                 # select only pdb_id (remove chain)
#                 pdb_id = pdb_id[:4]
                
#                 if pdb_id in pdb_id_to_uniprot_accession:
#                     accessions = pdb_id_to_uniprot_accession[pdb_id]
#                     for accession, pdb_chains in accessions.items():
#                         # initialise accession for current ligand
#                         if accession not in ligand_uniprot_to_pdb:
#                             ligand_uniprot_to_pdb[accession] = []
#                         # get chain(s) of current pdb corresponding to current accession
#                         for pdb_chain in pdb_chains:
#                             chain_id = pdb_chain["chain_id"]
#                             ligand_uniprot_to_pdb[accession].append(f"{pdb_id}_{chain_id}")

#                 else:
#                     # print ("not in mapping")
#                     # pdb ID cannot be mapped to accession number 
#                     accession = pdb_id # use submitted name
#                     # add all chains 
#                     # initialise accession for current ligand
#                     if accession not in ligand_uniprot_to_pdb:
#                         ligand_uniprot_to_pdb[accession] = []
#                     ligand_uniprot_to_pdb[accession].append(pdb_id)
                    
#             ligands_to_targets[ligand_id]["targets"] = ligand_uniprot_to_pdb

#         else: # np mapping
            
#             ligands_to_targets[ligand_id]["targets"] = {
#                 f"preprocessed-{target}" : {target} for target in ligand_targets
#             }


#         # update all_targets_to_prepare with targets of current ligand
#         for accession, accession_targets in ligands_to_targets[ligand_id]["targets"].items():
#             if accession not in all_targets_to_prepare:
#                 all_targets_to_prepare[accession] = set()
#             for pdb_id in accession_targets:
#                 all_targets_to_prepare[accession].add(pdb_id)

#     # base directory to write decompressed prepared targets to
#     prepared_targets_dir = os.path.join(output_dir, "prepared_targets")

#     # prepare all targets in all_pdb_targets set (loads data if exists)
#     prepared_targets = prepare_all_targets(
#         accession_to_pdb=all_targets_to_prepare,
#         decompress_to_directory=prepared_targets_dir,
#         existing_target_filenames=submitted_target_pdb_files, # pass in any submitted targets
#         bounding_box_scale=bounding_box_scale,
#         max_bounding_box_size=max_bounding_box_size,
#         allow_multi_model=True, # ?
#         allow_mutant=allow_mutant,
#         allow_engineered=allow_engineered,
#         protonate_to_pH=target_pH,
#         run_fpocket=run_fpocket,
#         run_fpocket_old=run_fpocket_old,
#         run_p2rank=run_p2rank,
#         compute_voxel_locations=compute_voxel_locations,
#         max_pockets_to_keep=max_pockets_to_keep,
#         keep_cofactors=keep_cofactors,
#         verbose=verbose,
#     )

#     # add pdb id ranks if map_uniprot_to_pdb
#     if map_uniprot_to_pdb == 1:
#         for accession in prepared_targets:
#             if accession not in pdb_id_ranks_for_all_uniprot_accessions:
#                 continue
#             for pdb_id, pdb_id_data in prepared_targets[accession].items():
#                 if pdb_id not in pdb_id_ranks_for_all_uniprot_accessions[accession]:
#                     continue
#                 # add rank of PDB ID for current accession
#                 pdb_id_data["pdb_id_rank_for_accession"] = pdb_id_ranks_for_all_uniprot_accessions[accession][pdb_id]["reranked_rank"]
            
#     # update ligand_to_targets with target data
#     for ligand_id, ligand_data in ligands_to_targets.items():
#         if verbose:
#             print ("Adding prepared_targets for ligand", ligand_id)

#         # initialise prepared_targets dict for current ligand
#         ligand_prepared_targets = {}
#         for accession, ligand_accession_pdb_ids in ligand_data["targets"].items():
#             # skip accession if it could not be prepared
#             if accession not in prepared_targets:
#                 continue
#             prepared_targets_for_accession = prepared_targets[accession]
#             ligand_accession_pdb_ids = sorted(ligand_accession_pdb_ids)
            
#             # check for accession=pdb_id
#             if accession == ligand_accession_pdb_ids[0]:
#                 # add all chains
#                 ligand_prepared_targets[accession] = prepared_targets_for_accession
#             else:
#                 ligand_prepared_targets_for_accession = {}

#                 for ligand_accession_pdb_id in ligand_accession_pdb_ids:
#                     # if ligand_accession_pdb_id.startswith("alphafold-"):
#                     #     # append chain
#                     #     ligand_accession_pdb_id += "_A"
#                     ligand_accession_pdb_id = ligand_accession_pdb_id.upper()
#                     # if ligand_accession_pdb_id in prepared_targets_for_accession:
#                     # account for potential appending of chain(s)

#                     for prepared_target_for_accession in prepared_targets_for_accession:
#                         if prepared_target_for_accession.startswith(ligand_accession_pdb_id):
#                             ligand_prepared_targets_for_accession[prepared_target_for_accession] = prepared_targets_for_accession[prepared_target_for_accession]
#                 # add all prepared pdb IDs for accession to ligand_prepared_targets
#                 ligand_prepared_targets[accession] = ligand_prepared_targets_for_accession

#         # add "prepared_targets" key to ligand_data
#         # ensure copy
#         ligand_data["prepared_targets"] = copy.deepcopy(ligand_prepared_targets)

#         # delete "targets" key to save memory
#         del ligand_data["targets"]


#     return ligands_to_targets, prepared_targets_dir, ligand_structure_output_dir

def map_pdb_id_to_uniprot_accession(
    pdb_ids,
    pdb_to_uniprot_filename: str = "data/databases/pdb/pdb_to_uniprot.pkl.gz",
    verbose: bool = False,
    ):
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
    
    if verbose:
        print ("Mapping", len(pdb_ids), "PDB IDs to Uniprot accession ID")

    pdb_ids = map(str.upper, pdb_ids)
    # length filter no longer required
    # pdb_ids = filter(lambda s: len(s) == 4, pdb_ids)

    # pdb_to_uniprot = load_json(mapping_json_filename, verbose=verbose)
    pdb_to_uniprot = load_compressed_pickle(pdb_to_uniprot_filename, verbose=verbose)

    mapping = { 
        pdb_id: pdb_to_uniprot[pdb_id] 
        for pdb_id in pdb_ids
        if pdb_id in pdb_to_uniprot 
    }

    return mapping


def map_uniprot_accession_to_pdb_id(
    accessions: list,
    minimum_sequence_length: int = 150,
    return_single_chain_only: bool = True,
    keep_maximum_sequence_length_only: bool = False,
    uniprot_to_pdb_filename: str = "data/databases/pdb/uniprot_to_pdb.pkl.gz",
    verbose: bool = False,
    ):
    if isinstance(accessions, str):
        accessions = [accessions]
    
    if verbose:
        print ("Mapping", len(accessions), "Uniprot accession IDs to PDB ID")

    # uniprot_to_pdb = load_json(uniprot_to_pdb_filename, verbose=verbose)
    uniprot_to_pdb = load_compressed_pickle(uniprot_to_pdb_filename, verbose=verbose)

    # ensure all accessions are in upper case
    accessions = map(str.upper, accessions)

    # initialise mapping
    accession_to_pdb_mapping = {}
    for accession in accessions:

        if accession in uniprot_to_pdb:
            accession_pdb_ids = uniprot_to_pdb[accession]
        else:
            accession_pdb_ids = {}

        # new format: pdb_id -> chain -> {"sequence_length"}

        # filter by minimum sequence length
        if minimum_sequence_length is not None:
            accession_pdb_ids = {
                pdb_id: {
                    chain: chain_data 
                    for chain, chain_data in pdb_id_chain_data.items()
                    if chain_data["sequence_length"] >= minimum_sequence_length
                }
                for pdb_id, pdb_id_chain_data in accession_pdb_ids.items()
            }
            # remove PDB IDs with no chains
            accession_pdb_ids = {
                pdb_id: pdb_id_chain_data
                for pdb_id, pdb_id_chain_data in accession_pdb_ids.items()
                if len(pdb_id_chain_data) > 0
            }
        

        # filter for maximum sequence length only
        sequence_lengths = [
            chain_data["sequence_length"]
            for pdb_id, pdb_id_chain_data in accession_pdb_ids.items()
            for chain, chain_data in pdb_id_chain_data.items()  
        ]
        if keep_maximum_sequence_length_only and len(sequence_lengths) > 0:
            # store maximum lengths
            maximum_sequence_length = max(sequence_lengths)

            # keep only max sequence length
            accession_pdb_ids = {
                pdb_id: {
                    chain: chain_data 
                    for chain, chain_data in pdb_id_chain_data.items()
                    if chain_data["sequence_length"] == maximum_sequence_length
                }
                for pdb_id, pdb_id_chain_data in accession_pdb_ids.items()
            }
            # remove PDB IDs with no chains
            accession_pdb_ids = {
                pdb_id: pdb_id_chain_data
                for pdb_id, pdb_id_chain_data in accession_pdb_ids.items()
                if len(pdb_id_chain_data) > 0
            }

        # select only single chain
        if return_single_chain_only:
            
            accession_pdb_ids = {
                pdb_id: {
                    chain: chain_data 
                    for i, (chain, chain_data) in enumerate(pdb_id_chain_data.items())
                    if i < 1
                }
                for pdb_id, pdb_id_chain_data in accession_pdb_ids.items()
            }
        
        accession_to_pdb_mapping[accession] = accession_pdb_ids

    return accession_to_pdb_mapping


def prepare_ligand_for_docking(
    ligand_filename: str,
    docking_program: str,
    output_filename: str = None,
    overwrite: bool = False,
    verbose: bool = False,
    ):

    docking_program = docking_program.lower()

    if docking_program in VINA_VARIANTS:
        return prepare_ligand_for_vina(
            ligand_filename=ligand_filename,
            output_filename=output_filename,
            overwrite=overwrite,
            verbose=verbose,
        )
    elif docking_program == "plants":
        title = os.path.basename(ligand_filename)
        title = os.path.splitext(title)[0]
        return prepare_for_plants(
            title=title,
            input_filename=ligand_filename,
            output_filename=output_filename,
            overwrite=overwrite,
            verbose=verbose,
        )
    elif docking_program == "galaxydock":
        return prepare_ligand_for_galaxydock(
            input_filename=ligand_filename,
            output_filename=output_filename,
            overwrite=overwrite,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(docking_program)

def prepare_target_for_docking(
    target_filename: str,
    docking_program: str,
    output_filename: str = None,
    overwrite: bool = False,
    verbose: bool = False,
    ):
    docking_program = docking_program.lower()

    if docking_program in VINA_VARIANTS:
        return prepare_target_for_vina(
            target_filename=target_filename,
            output_filename=output_filename,
            overwrite=overwrite,
            verbose=verbose,
        )
    elif docking_program == "plants":
        title = os.path.basename(target_filename)
        title = os.path.splitext(title)[0]
        return prepare_for_plants(
            title=title,
            input_filename=target_filename,
            output_filename=output_filename,
            overwrite=overwrite,
            verbose=verbose,
        )
    elif docking_program == "galaxydock":
        return prepare_target_for_galaxydock(
            input_filename=target_filename,
            output_filename=output_filename,
            overwrite=overwrite,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(docking_program)

def get_default_n_proc(
    docking_program: str,
    ):
    docking_program = docking_program.lower()
    if docking_program in VINA_VARIANTS:
        return VINA_N_PROC
    elif docking_program == "plants":
        return PLANTS_N_PROC
    elif docking_program == "galaxydock":
        return GALAXYDOCK_N_PROC
    else:
        raise NotImplementedError(docking_program)

def get_docking_output_filenames(
    docking_program: str,
    ):
    docking_program = docking_program.lower()
    if docking_program in VINA_VARIANTS:
        pose_data_basename = f"{docking_program}.log.json"
        pose_score_field_name = "energy"
    elif docking_program == "plants":
        pose_data_basename = "plants_scores.json"
        pose_score_field_name = "total_score"
    else: 
        raise NotImplementedError(docking_program)
    return pose_data_basename, pose_score_field_name

def compute_voxel_locations_function(
    center_of_mass: np.ndarray,
    bounding_box_size: np.ndarray,
    voxel_size: float,
    verbose: bool = False,
    ):
    """compute the centres of grid boxes of size `voxel_size` that cover the entire area of 
    `bounding_box_size`.

    Parameters
    ----------
    center_of_mass : np.ndarray
        The x, y, z co-ordinates of the center of the bounding box
    bounding_box_size : np.ndarray
        The x, y, z sizes of the bounding box
    voxel_size : float
        The size of the voxels
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    list
        List of dictionaries containing the location and sizes of each grid box. 
    """

    if verbose:
        print ("Computing the co-ordinates of the grid boxes")

    # smallest x, y, z
    start_point = center_of_mass - (bounding_box_size + voxel_size) / 2
    end_point = center_of_mass + bounding_box_size / 2

    box_x_coords = np.arange(start_point[0], end_point[0], voxel_size)
    box_y_coords = np.arange(start_point[1], end_point[1], voxel_size)
    box_z_coords = np.arange(start_point[2], end_point[2], voxel_size)

    all_coords = []
    for x, y, z in product(box_x_coords, box_y_coords, box_z_coords):
        all_coords.append({
            "center_x": x,
            "center_y": y,
            "center_z": z,
            "size_x": voxel_size,
            "size_y": voxel_size,
            "size_z": voxel_size,
        })
    return all_coords

def cleanup_target_for_docking(
    target_identifier: str,
    target_filename: str,
    output_dir: str,
    prepared_target_filename: str = None,
    cleanup_hydrogens_with_pymol: bool = False,
    verbose: bool = False,
    ):
    """Remove all hetero residues from a target using Biopython.
    Optionally, use PyMol to remove hydrogens and foldx to try to improve the quality of the structure.

    Parameters
    ----------
    target_identifier : str
        Identifier of the target
    target_filename : str
        Path of the PDB file to clean
    output_dir : str
        Directory to otput the cleaned structure to
    prepared_target_filename : str, optional
        Filename to write the cleaned structure to, by default None
    use_foldx_to_improve_protein_structure : bool, optional
        Flag to run foldx, by default False
    cleanup_hydrogens_with_pymol : bool, optional
        Flag to run Pymol, by default False

    Returns
    -------
    str
        File path of the cleaned target structure
    """
  
    if cleanup_hydrogens_with_pymol:
        target_filename = cleanup_with_pymol(
            target_filename, 
            output_target_pdb_filename=target_filename,
            verbose=verbose)

    if prepared_target_filename is None:
        prepared_target_filename = os.path.join(
            output_dir, 
            f"{target_identifier}_prepared.pdb")

    if verbose:
        print ("Cleaning up target structure file", target_filename, "for docking")
        print ("Writing to", prepared_target_filename)

    # remove all hetero residues
    prepared_target_filename = remove_all_hetero_residues_using_biopython(
        pdb_id=target_identifier,
        pdb_filename=target_filename,
        output_filename=prepared_target_filename,
    )

    return prepared_target_filename

def prepare_single_target(
    target_identifier: str,
    existing_target_filename: str,
    output_dir: str, 
    decompress_to_directory: str = None,
    allow_multi_model: bool = True,
    allow_mutant: bool = True,
    allow_engineered: bool = True,
    min_atoms_in_natural_ligand: int = 8,
    run_fpocket: bool = True,
    run_fpocket_old: bool = False,
    min_fpocket_pocket_score: float = None,
    run_p2rank: bool = True,
    min_p2rank_pocket_score: float = None,
    compute_bounding_box: bool = False,
    bounding_box_scale: float = 1,    
    min_bounding_box_size: float = None,
    max_bounding_box_size: float = None,
    cofactors_to_keep: list = None,
    compute_voxel_locations: bool = False,
    voxel_size: float = 10,
    max_num_atoms: int = None,
    max_pockets_to_keep: int = 1,
    protonate_to_pH: float = 7.4,
    verbose: bool = False,
    ):

    # existing_target_filename, # leave as None to download from pdbid
    # desired_chain, # leave as None for all chains

    if target_identifier is None:
        return {}

    existing_target_file_provided = existing_target_filename is not  None 
    
    target_identifier = target_identifier.strip()
    target_identifier = target_identifier.upper()

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print ("Preparing target", target_identifier, "and outputting data and structure to directory", output_dir)

    # identify chains of interest
    # use target identifier to define chains to keep
    if target_identifier.count("_") == 1:
        target_identifier, chains = target_identifier.split("_")
        # split characters (AB -> ["A", "B"]) ?
        # all_chains_to_process = sorted(chains)
        # keep together?
        all_chains_to_process = [chains]

    # identify all chains in PDB file 
    # if all_chains_to_process is None:
    else:
        # provided PDB file
        if existing_target_filename is not None and os.path.exists(existing_target_filename):
            # read all chains from existing pdb filename 
            all_chains_to_process = get_all_chain_ids_in_a_PDB_file(
                pdb_id=target_identifier,
                pdb_filename=existing_target_filename,
            )

            # keep all chains in user supplied file?
            all_chains_to_process = {"".join(sorted(all_chains_to_process))}


        elif target_identifier.startswith("ALPHAFOLD-"): # alphafold target
            all_chains_to_process = ["A"] # alphafold
        else: 
            # download pdb file and identify all chains 
            temp_pdb_filename = os.path.join(output_dir, f"{get_token()}.pdb")
            original_target_filename = download_pdb_structure_using_pdb_fetch(
                pdb_id=target_identifier,
                pdb_filename=temp_pdb_filename,
                verbose=verbose,
            )

            # check for empty file / error in download
            if original_target_filename is None or not os.path.exists(original_target_filename) \
                or os.stat(original_target_filename).st_size == 0:
                print (original_target_filename, "is missing, returning {}")
                return {}

            all_chains_to_process = get_all_chain_ids_in_a_PDB_file(
                pdb_id=target_identifier,
                pdb_filename=original_target_filename,
            )
            # delete original file 
            delete_file(original_target_filename, verbose=verbose)

    # ensure unique chains
    all_chains_to_process = set(all_chains_to_process)

    # iterate over desired chains and return data for all 
    data_all_chains = {}

    # iterate over all chain_to_process string in all_chains_to_process set
    for chain_to_process in all_chains_to_process:

        # sort desired chain for consistency
        num_chains = len(chain_to_process)
        if num_chains > 1:
            chain_to_process = "".join(sorted(chain_to_process))

        target_identifier_desired_chain = f"{target_identifier}_{chain_to_process}"

        desired_chain_output_dir = os.path.join(
            output_dir,
            target_identifier_desired_chain,
        )
        os.makedirs(desired_chain_output_dir, exist_ok=True,)

        if decompress_to_directory is not None:
            desired_chain_decompress_to_directory = os.path.join(
                decompress_to_directory,
                target_identifier_desired_chain,
            )
            os.makedirs(desired_chain_decompress_to_directory, exist_ok=True,)
        else:
            desired_chain_decompress_to_directory = None 

        # load target data if it exists
        target_data_filename = os.path.join(
            desired_chain_output_dir, 
            f"{target_identifier_desired_chain}_data.pkl.gz")
        if os.path.exists(target_data_filename):
            if verbose:
                print (target_data_filename, "already exists, loading it")
            target_data = load_compressed_pickle(target_data_filename, verbose=verbose)
        else:
            target_data = {}

        write_target_data_to_file = False

        # download original file 
        download_pdb_structure = "completeness" not in target_data \
            or "resolution" not in target_data \
            or "r_value_observed" not in target_data \
            or "r_value_free" not in target_data \
            or "is_mutant" not in target_data \
            or "is_engineered" not in target_data \
            or "num_models" not in target_data \
            or "num_atoms" not in target_data \
            or "natural_ligands" not in target_data \
            or "num_natural_ligands" not in target_data \
            or "compressed_prepared_filename" not in target_data \
            or "prepared_filename" not in target_data

        if download_pdb_structure:

            # update to JSON
            write_target_data_to_file = True

            # original filename
            original_target_filename = os.path.join(
                desired_chain_output_dir, 
                f"{target_identifier_desired_chain}.pdb")

            if not os.path.exists(original_target_filename) or os.path.getsize(original_target_filename) == 0:
                if existing_target_filename is not None and os.path.exists(existing_target_filename):
                    # copy from exisiting
                    copy_file(existing_target_filename, original_target_filename, verbose=verbose)
                elif target_identifier.startswith("ALPHAFOLD"): 
                    # download from Alphafold DB
                    if verbose:
                        print ("Downloading structure from AlphaFold database")
                    accession = target_identifier.split("-")[1] # ALPHAFOLD-accession
                    original_target_filename = download_alphafold_structure(
                        uniprot_accession=accession,
                        output_filename=original_target_filename,
                        verbose=verbose,
                    )
                    if original_target_filename is None:
                        print ("AlphaFold download error for target", target_identifier)
                        return {}
                else:
                    # download with pdb_fetch
                    if verbose:
                        print ("Downloading target structure using pdb_fetch")
                    
                    assert "_" not in target_identifier
                    assert len(target_identifier) == 4, target_identifier

                    original_target_filename = download_pdb_structure_using_pdb_fetch(
                        pdb_id=target_identifier, # use target_identifier to download structure
                        pdb_filename=original_target_filename,
                        verbose=verbose,
                    )
                    if original_target_filename is None:
                        print ("Download structure with pdb_fetch error", target_identifier)
                        return {}

            # check for empty file (normally caused by invalid PDB ID)
            if os.path.getsize(original_target_filename) == 0:
                if verbose:
                    print (original_target_filename, "is empty!")
                # delete it 
                delete_file(original_target_filename, verbose=verbose)
                return {} # returning {} makes sense since all chains will fail 

            # obabel covert from pdb to pdb to fix any formatting issues
            # original_target_filename = obabel_convert(
            #     input_format="pdb",
            #     input_filename=original_target_filename,
            #     output_format="pdb",
            #     output_filename=original_target_filename,
            #     overwrite=True,
            #     verbose=verbose,
            # )

            if original_target_filename is None or os.path.getsize(original_target_filename) == 0:
                return {} # returning {} makes sense since all chains will fail 

            # read values from raw PDB file
            for key, read_pdb_file_function in (
                ("completeness", get_completeness_from_pdb_file),
                ("resolution", get_resolution_from_pdb_file),
                ("r_value_observed", get_r_value_observed_from_pdb_file),
                ("r_value_free", get_r_value_free_from_pdb_file),
                ("is_mutant", get_is_mutant_from_pdb_file),
                ("is_engineered", get_is_engineered_from_pdb_file),
            ):
                if key not in target_data:
                    target_data[key] = read_pdb_file_function(
                        original_target_filename,
                        verbose=verbose,
                        )

            # count number of models in original file
            num_models = get_number_of_models_in_pdb_file(
                pdb_id=target_identifier_desired_chain,
                pdb_filename=original_target_filename,
                output_dir=desired_chain_output_dir,
                verbose=verbose,
            )
            target_data["num_models"] = num_models
            
            # select first model only
            first_model_filename = write_single_model(
                pdb_id=target_identifier_desired_chain,
                pdb_filename=original_target_filename,
                output_dir=desired_chain_output_dir,
                verbose=verbose,
            )
            # delete original file
            delete_file(original_target_filename, verbose=verbose)

            # select chain
            assert chain_to_process is not None

            target_filename = select_chains_from_pdb_file(
                pdb_id=target_identifier_desired_chain,
                pdb_filename=first_model_filename,
                chain_ids=list(chain_to_process), # convert to list
                verbose=verbose,
            ) 
            # delete first_model_filename
            delete_file(first_model_filename, verbose=verbose)
           
            # natural ligands 
            try:
                target_natural_ligands = get_all_natural_ligands_from_pdb_file(
                    pdb_id=target_identifier_desired_chain,
                    pdb_filename=target_filename,
                    min_heavy_atoms=min_atoms_in_natural_ligand,
                    output_dir=desired_chain_output_dir,
                    delete_output_dir=False, # keep natural ligands (compressed)
                    verbose=verbose,
                    )
            except Exception:
                target_natural_ligands = [] 

            # convert to dict to make consistent with other attributes
            total_num_natural_ligands = 0
            target_natural_ligand_data = {}
            # iterate over all natural ligands in remaining chains
            for natural_ligand in target_natural_ligands:
                base_ligand_id = natural_ligand["ligand_id"]

                if base_ligand_id is None or (isinstance(base_ligand_id, float) and np.isnan(base_ligand_id)):
                    ligand_id = None
                else:
                    # increment count of total number of natural ligands
                    total_num_natural_ligands += 1
                    counter = 1
                    ligand_id = f"{base_ligand_id}_{counter}"
                    while ligand_id in target_natural_ligand_data:
                        counter += 1
                        ligand_id = f"{base_ligand_id}_{counter}"

                target_natural_ligand_data[ligand_id] = natural_ligand


            target_data["natural_ligands"] = target_natural_ligand_data
            # add natural ligand count
            target_data["num_natural_ligands"] = total_num_natural_ligands

            # prepared for docking filename
            stem, ext = os.path.splitext(target_filename)
            prepared_target_filename = stem + "_prepared.pdb"

            # skip this for provided files?
            if existing_target_file_provided:
                copy_file(target_filename, prepared_target_filename, verbose=verbose)
            else:

                prepared_target_filename = cleanup_target_for_docking(
                    target_identifier=target_identifier_desired_chain,
                    target_filename=target_filename,
                    prepared_target_filename=prepared_target_filename,
                    output_dir=desired_chain_output_dir,
                    verbose=verbose,
                )

            # delete target filename now that it has been prepared
            delete_file(target_filename, verbose=verbose)

            # number of atoms (in cleaned structure)
            target_data["num_atoms"] = get_number_of_atoms_in_pdb_file(
                pdb_id=target_identifier_desired_chain,
                pdb_filename=prepared_target_filename,
                verbose=verbose,
            )

            # save path of target with no hetero residues 
            target_data["prepared_filename"] = prepared_target_filename
            compressed_prepared_target_filename = gzip_file(
                input_filename=prepared_target_filename,
                delete_original_file=False,
                verbose=verbose,
            )
            # save path to compressed prepared target filename
            target_data["compressed_prepared_filename"] = compressed_prepared_target_filename

        # ensure all natural ligands have "num_atoms"
        if target_data["num_natural_ligands"] > 0:
            for natural_ligand_id, natural_ligand_data in target_data["natural_ligands"].items():
                if "num_atoms" in natural_ligand_data:
                    continue
                if "atom_centers" not in natural_ligand_data:
                    continue

                natural_ligand_data["num_atoms"] = len(natural_ligand_data["atom_centers"])
                
                # update file on disk
                write_target_data_to_file = True

        # all subsequent functions should be performed on cleaned structure
        prepared_target_filename = target_data["prepared_filename"] # raw .pdb file
        compressed_prepared_target_filename = target_data["compressed_prepared_filename"]

        if run_fpocket and "fpocket" not in target_data:
            
            # gunzip compressed prepared structure 
            if not os.path.exists(prepared_target_filename):
                gunzip_file(
                    gzip_filename=compressed_prepared_target_filename,
                    output_filename=prepared_target_filename,
                    delete_gzip_file=False,
                    verbose=verbose,
                )

            target_data["fpocket"] = run_fpocket_and_collate_single_target(
                target_identifier=target_identifier_desired_chain,
                output_dir=desired_chain_output_dir,
                min_pocket_score=min_fpocket_pocket_score,
                existing_pdb_filename=prepared_target_filename,
                verbose=verbose,
            )

            # update to JSON
            write_target_data_to_file = True

        # old style for COBDock
        if run_fpocket_old and "fpocket_old" not in target_data:
            
            # gunzip compressed prepared structure 
            if not os.path.exists(prepared_target_filename):
                gunzip_file(
                    gzip_filename=compressed_prepared_target_filename,
                    output_filename=prepared_target_filename,
                    delete_gzip_file=False,
                    verbose=verbose,
                )

            target_data["fpocket_old"] = run_fpocket_and_collate_single_target(
                target_identifier=target_identifier_desired_chain,
                output_dir=desired_chain_output_dir,
                min_pocket_score=min_fpocket_pocket_score,
                existing_pdb_filename=prepared_target_filename,
                read_features_from_pocket_pdbs=True,
                verbose=verbose,
            )

            # update to JSON
            write_target_data_to_file = True
        
        if run_p2rank and "p2rank" not in target_data:

            # gunzip compressed prepared structure 
            if not os.path.exists(prepared_target_filename):
                gunzip_file(
                    gzip_filename=compressed_prepared_target_filename,
                    output_filename=prepared_target_filename,
                    delete_gzip_file=False,
                    verbose=verbose,
                )
    
            target_data["p2rank"] = run_p2rank_and_collate_single_target(
                target_identifier=target_identifier_desired_chain,
                output_dir=desired_chain_output_dir,
                min_pocket_score=min_p2rank_pocket_score,
                existing_pdb_filename=prepared_target_filename,
                verbose=verbose,
            )

            # update to JSON
            write_target_data_to_file = True
        
        # remove "cavities" directory
        cavities_directory = os.path.join(desired_chain_output_dir, "cavities")
        delete_directory(cavities_directory, verbose=verbose)
        
        # compute mass center and size of target (EXPENSIVE FOR LARGE STRUCTURES) (required for COB dock)
        if (compute_bounding_box or compute_voxel_locations) and ("center_x" not in target_data or "size_x" not in target_data):

            # gunzip compressed prepared structure 
            if not os.path.exists(prepared_target_filename):
                gunzip_file(
                    gzip_filename=compressed_prepared_target_filename,
                    output_filename=prepared_target_filename,
                    delete_gzip_file=False,
                    verbose=verbose,
                )

            # dict containing center and size properties
            target_center_of_mass_and_bounding_box = define_target_bounding_box_using_biopython(
                pdb_filename=prepared_target_filename,
                scale=1,
                precision=3,
                verbose=verbose,
            )

            target_data.update(target_center_of_mass_and_bounding_box)

            # update to JSON
            write_target_data_to_file = True
        
        # grid box locations (required for COB dock)
        if compute_voxel_locations and "voxel_locations" not in target_data and "center_x" in target_data and "size_x" in target_data:

            # update to JSON
            write_target_data_to_file = True
    
            center_of_mass = np.array([
                target_data[f"center_{k}"]
                    for k in ("x", "y", "z")
            ])
            bounding_box_size = np.array([
                target_data[f"size_{k}"]
                    for k in ("x", "y", "z")
            ])
            voxel_locations = compute_voxel_locations_function(
                center_of_mass=center_of_mass,
                bounding_box_size=bounding_box_size,
                voxel_size=voxel_size,
                verbose=verbose,
            )
            target_data.update({
                "voxel_locations": voxel_locations,
                "number_of_voxel_locations": len(voxel_locations),
            })

        # write target data to file if any changes were made 
        if write_target_data_to_file:
            # write_json(target_data, target_data_filename, verbose=verbose)
            write_compressed_pickle(target_data, target_data_filename, verbose=verbose)

        # delete prepared PDB file (before returns to ensure they are deleted)
        if os.path.exists(prepared_target_filename):
            delete_file(prepared_target_filename, verbose=verbose)

        # check for num models
        if not allow_multi_model and "num_models" in target_data:
            num_models = target_data["num_models"]
            if num_models > 1:
                if verbose:
                    print (target_identifier_desired_chain, "contains too many models")
                # return {}
                continue # skip chain

        # check for atom number (after selecting first model only and optionally selecting chain)
        if max_num_atoms is not None and "num_atoms" in target_data:
            num_atoms = target_data["num_atoms"]
            if num_atoms > max_num_atoms:
                if verbose:
                    print (target_identifier_desired_chain, "has too many atoms:", num_atoms)
                continue # skip chain

        # other filter criteria 
        if not allow_mutant and target_data["is_mutant"]:
            if verbose:
                print (target_identifier_desired_chain, "is a mutant, skipping")
            continue # skip chain

        if not allow_engineered and target_data["is_engineered"]:
            if verbose:
                print (target_identifier_desired_chain, "is engineered, skipping")
            continue # skip chain 

        # clip bounding box if necessary (do not affect JSON file)
        if min_bounding_box_size is not None:
            for key in ("size_x", "size_y", "size_z"):
                target_data[key] = np.maximum(target_data[key], min_bounding_box_size)
        if max_bounding_box_size is not None:
            for key in ("size_x", "size_y", "size_z"):
                target_data[key] = np.minimum(target_data[key], max_bounding_box_size)

        # remove data about pockets not required for docking
        if max_pockets_to_keep is not None:
            for pocket_location_key in (
                "fpocket",
                "p2rank",
            ):
                if pocket_location_key not in target_data:
                    continue
                pocket_ids_to_delete = []
                for pocket_id in target_data[pocket_location_key]:
                    if pocket_id is None or pocket_id == "null":
                        continue
                    pocket_id_int = int(pocket_id)
                    if pocket_id_int > max_pockets_to_keep:
                        pocket_ids_to_delete.append(pocket_id)
                for pocket_id_to_delete in pocket_ids_to_delete:
                    del target_data[pocket_location_key][pocket_id_to_delete]

        # handle decompression and update prepared_filename attribute 
        # also create complexes with co-factors if cofactors_to_keep is not empty
        if desired_chain_decompress_to_directory is not None:
            os.makedirs(desired_chain_decompress_to_directory, exist_ok=True,)

            target_decompressed_filename = os.path.join(
                desired_chain_decompress_to_directory,
                f"{target_identifier_desired_chain}_prepared.pdb",
            )
            target_decompressed_filename = gunzip_file(
                gzip_filename=compressed_prepared_target_filename,
                output_filename=target_decompressed_filename,
                delete_gzip_file=False,
                verbose=verbose,
            )

            # protonate if required 
            if protonate_to_pH is not None:

                stem, _ = os.path.splitext(target_decompressed_filename)

                protonated_pqr_filename = stem + f"_protonated_{protonate_to_pH}.pqr"
                protonated_pdb_filename = stem + f"_protonated_{protonate_to_pH}.pdb"

                if not os.path.exists(protonated_pdb_filename):

                    protonated_pdb_filename = protonate_pdb(
                        input_filename=target_decompressed_filename,
                        pqr_output_filename=protonated_pqr_filename,
                        pdb_output_filename=protonated_pdb_filename,
                        pH=protonate_to_pH,
                        return_as_pdb=True,
                        verbose=verbose,
                    )

                if protonated_pdb_filename is not None \
                    and os.path.exists(protonated_pdb_filename):
                    # remove old decompressed filename
                    delete_file(target_decompressed_filename, verbose=verbose)
                    target_decompressed_filename = protonated_pdb_filename

            # handle co-factors
            if cofactors_to_keep is not None:
                cofactor_ligand_filenames = []
                for cofactor_to_keep in cofactors_to_keep:

                    for natural_ligand_id, natural_ligand_data in target_data["natural_ligands"].items():
                        if not natural_ligand_id.startswith(cofactor_to_keep):
                            continue # skip natural ligands that are not the cofactor of interest
                        natural_ligand_compressed_filename = natural_ligand_data["compressed_pdb_filename"]
                        natural_ligand_decompressed_filename = os.path.join(
                            desired_chain_decompress_to_directory,
                            f"{natural_ligand_id}.pdb"
                        )
                        natural_ligand_decompressed_filename = gunzip_file(
                            natural_ligand_compressed_filename,
                            natural_ligand_decompressed_filename,
                            verbose=verbose,
                        )
                        # add to list of all cofactor PDB filenames
                        cofactor_ligand_filenames.append(natural_ligand_decompressed_filename)
                
                # create complex if cofactors were found
                if len(cofactor_ligand_filenames) > 0:
                    # cant do this outside of main thread
                    # lets return a list
                    target_decompressed_filename = [target_decompressed_filename] + cofactor_ligand_filenames

            # update "prepared_filename"
            target_data["prepared_filename"] = target_decompressed_filename

        # update  data_all_chains
        data_all_chains[target_identifier_desired_chain] = target_data

        if verbose:
            print ("Completed preparing target", target_identifier_desired_chain)

    return data_all_chains

def prepare_all_targets(
    accession_to_pdb: list,
    output_dir: str = None,
    existing_target_filenames: dict = {},
    allow_multi_model: bool = False,
    allow_mutant: bool = True,
    allow_engineered: bool = True,
    bounding_box_scale: float = 1,
    run_fpocket: bool = True,
    run_fpocket_old: bool = False, # TODO: remove after update
    run_p2rank: bool = True,
    min_bounding_box_size: float = None,
    max_bounding_box_size: float = None,
    compute_voxel_locations: bool = False,
    voxel_size: float = 10,
    max_num_atoms: int = None,
    max_pockets_to_keep: int = 1,
    decompress_to_directory: str = None,
    protonate_to_pH: float = 7.4,
    keep_cofactors: bool = True,
    n_proc: int = None,
    verbose: bool = True,
    ):

    if output_dir is None:
        output_dir = PREPARE_TARGETS_ROOT_DIR

    if n_proc is None:
        n_proc = PREPARE_TARGETS_N_PROC

    if verbose:
        print ("Preparing targets", list(accession_to_pdb), "using", n_proc, "process(es)")
        print ("Outputting to directory", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # handle user-provided target names
    if existing_target_filenames is None:
        existing_target_filenames = {}
    if isinstance(existing_target_filenames, dict):
        # map keys to upper case
        existing_target_filenames = {
            k.upper(): v 
            for k, v in existing_target_filenames.items()
        }

    # add cofactors
    if keep_cofactors:
        accession_to_cofactors = get_cofactors_for_accessions(
            accessions=accession_to_pdb,
            verbose=verbose
        )
    else:
        accession_to_cofactors = {
            accession: None # None value will skip cofactor step 
            for accession in accession_to_pdb
        }

    # identify mass centers, bounding box, and prepare for docking 
    all_prepared_target_data = {}

    with ProcessPoolExecutor(max_workers=n_proc) as p:

        running_tasks = {}

        for accession in accession_to_pdb:
            accession_upper = accession.upper()

            # cofactors 
            accession_cofactors = accession_to_cofactors[accession]

            # enforce unique PDB IDs
            for pdb_id in set(accession_to_pdb[accession]):

                # existing pdb file
                if pdb_id in existing_target_filenames:
                    existing_target_filename = existing_target_filenames[pdb_id]
                else:
                    existing_target_filename = None

                if not pdb_id.startswith("alphafold-"):
                    pdb_id = pdb_id.upper() 

                accession_pdb_id_output_dir = os.path.join(
                    output_dir, 
                    accession_upper,
                    # pdb_id, #  do this internally
                )
                if decompress_to_directory is not None:
                    accession_pdb_id_decompress_to_directory = os.path.join(
                        decompress_to_directory,
                        accession_upper,
                        # pdb_id,
                    )
                else:
                    accession_pdb_id_decompress_to_directory = None


                kwargs = {
                    "target_identifier": pdb_id,
                    "existing_target_filename": existing_target_filename,
                    "output_dir": accession_pdb_id_output_dir, 
                    "decompress_to_directory": accession_pdb_id_decompress_to_directory, # decompress prepared structure here 
                    "allow_multi_model": allow_multi_model,
                    "allow_mutant": allow_mutant,
                    "allow_engineered": allow_engineered,
                    "bounding_box_scale": bounding_box_scale,
                    "run_fpocket": run_fpocket,
                    "run_fpocket_old": run_fpocket_old,
                    "run_p2rank": run_p2rank,
                    "compute_voxel_locations": compute_voxel_locations,
                    "min_bounding_box_size": min_bounding_box_size,
                    "max_bounding_box_size": max_bounding_box_size,
                    "voxel_size": voxel_size,
                    "max_num_atoms": max_num_atoms,
                    "max_pockets_to_keep": max_pockets_to_keep,
                    "protonate_to_pH": protonate_to_pH,
                    "cofactors_to_keep": accession_cofactors,
                    "verbose": verbose,
                }

                task = p.submit(
                    prepare_single_target,
                    **kwargs,
                )

                running_tasks[task] = {
                    "accession": accession,
                    # "pdb_id": pdb_id,
                    "kwargs": kwargs,
                }

        # iterate over tasks
        for task in as_completed(running_tasks):
            # await task completion
            task_result = task.result()
            # task_result == {} -> target did not meet criteria / some other failure
            if task_result is None or len(task_result) == 0:
                continue
            
            task_data = running_tasks[task]
            accession = task_data["accession"]
            # pdb_id = task_data["pdb_id"]

            if accession not in all_prepared_target_data:
                all_prepared_target_data[accession] = {}
            # all_prepared_target_data[accession][pdb_id] = task_result
            # task result in the form:
            # {pdb_id_chain_id: data, ....} (may be multiple pdb_id_chain_ids)

            # create complexes of cofactors in main thread
            for pdb_id, pdb_id_data in task_result.items():
                pdb_id_prepared_filename = pdb_id_data["prepared_filename"]
                if isinstance(pdb_id_prepared_filename, list):
                    pdb_id_prepared_filename = create_complex_with_pymol(
                        input_pdb_files=pdb_id_prepared_filename,
                        output_pdb_filename=pdb_id_prepared_filename[0], # always first
                        verbose=verbose,
                    )
                    # update pdb_id_data["prepared_filename"]
                    pdb_id_data["prepared_filename"] = pdb_id_prepared_filename

            all_prepared_target_data[accession].update(task_result)

    return all_prepared_target_data

def rank_all_pdb_structures_for_uniprot_accessions(
    uniprot_to_pdb_id: dict,
    sequence_lengths: dict = {},
    allow_multi_model: bool = False,
    allow_mutant: bool = True,
    allow_engineered: bool = True,
    allow_unliganded: bool = False,
    consider_prefix_in_reranking: bool = True,
    allow_duplicate_prefixes: bool = False,
    prefix_size: int = 3,
    min_sequence_length: int = 100,
    n_proc: int = None,
    verbose: bool = False,
    ):

    # NOTE that considering prefix means that multiple chains for each crystal structure are not considered

    if verbose:
        print ("Begin ranking of PDB structures for uniprot targets:", list(uniprot_to_pdb_id))
        print ("Using", n_proc, "process(es)")

    uniprot_to_pdb_data = prepare_all_targets( # now handles accession -> PDB internally
        accession_to_pdb=uniprot_to_pdb_id,
        output_dir=None, # use default: data/ai_blind_docking/prepared_targets
        allow_multi_model=allow_multi_model,
        allow_mutant=allow_mutant,
        allow_engineered=allow_engineered,
        run_fpocket=False,
        run_p2rank=False,
        compute_voxel_locations=False,
        n_proc=n_proc,
        verbose=verbose,
    )

    # add sequence lengths
    for accession in uniprot_to_pdb_data:
        if accession not in sequence_lengths:
            continue
        for pdb_id in uniprot_to_pdb_data[accession]:
            if pdb_id not in sequence_lengths[accession]:
                continue
            
            # add sequence length
            uniprot_to_pdb_data[accession][pdb_id]["sequence_length"] = sequence_lengths[accession][pdb_id]

    # perform rank aggregation
    # package expects bigger is better 
    rank_aggregation_keys = {
        "completeness": 1, # bigger is better 
        "resolution": -1, # smaller is better
        "r_value_free": -1, # smaller is better
        "r_value_observed": -1, # smaller is better
    }
    all_ranks = {}
    for accession, pdb_data in uniprot_to_pdb_data.items():

        if len(pdb_data) == 0: # no valid PDB IDs for current uniprot ID
            all_ranks[accession] = {}
            continue

        # build experts
        experts = []
        for expert, scalar in rank_aggregation_keys.items():
            experts.append(
                {
                    pdb_id: scalar * expert_scores[expert]
                    for pdb_id, expert_scores in pdb_data.items()
                }
            )
        # expects bigger is better
        pdb_id_ranks_aggregated = perform_rank_aggregation(experts, are_scores=True)
        # pdb_id: rank dictionary
        pdb_rank_data_for_current_uniprot_accession = {
            pdb_id: {
                "accession": accession,
                "pdb_id": pdb_id,
                "rank": rank,
                **uniprot_to_pdb_data[accession][pdb_id],
            }
            for pdb_id, rank in pdb_id_ranks_aggregated.items()
        }

        # prioritise non-mutant and co-crystalised natural ligands?

        # assign new ranks based on prefix (if `consider_prefix_in_reranking`)
        seen_pdb_prefixes = set()

        # base ranked list of PDB IDs
        pdb_ids_sorted_by_rank = sorted(
            pdb_rank_data_for_current_uniprot_accession, 
            key=lambda pdb_id: (pdb_rank_data_for_current_uniprot_accession[pdb_id]["rank"], pdb_id), # rank first by rank, then pdb_id_chain
            )

        current_rank = 1

        if verbose:
            print ("Reranking PDB structures based on is_mutant, num_natural_ligands, sequence_length and prefix")

        # mutant, ligands and prefix
        for pdb_id in pdb_ids_sorted_by_rank:

            pdb_id_data = pdb_rank_data_for_current_uniprot_accession[pdb_id]

            # check if already processed
            if "reranked_rank" in pdb_id_data:
                continue

            # check sequence length
            if "sequence_length" not in pdb_id_data:
                continue
            sequence_length = pdb_id_data["sequence_length"]
            if min_sequence_length is not None and sequence_length < min_sequence_length:
                if verbose:
                    print ("PDB structure", pdb_id, "has sequence length:", sequence_length, 
                        "which is less than", min_sequence_length, ", deprioritising it")
                continue

            # deprioritise mutants
            is_mutant = pdb_id_data["is_mutant"]
            if is_mutant:
                if verbose:
                    print ("PDB Structure", pdb_id, "is a mutant, deprioritising it")
                continue

            # number of native ligands in file 
            number_of_natural_ligands = pdb_id_data["num_natural_ligands"]
            if number_of_natural_ligands == 0:
                if verbose:
                    print ("PDB Structure", pdb_id, "contains no natural ligands, deprioritising it")
                continue

            # skip current PDB ID based on prefix (if prefixes are considered)
            pdb_id_prefix = pdb_id[:prefix_size]
            if pdb_id_prefix in seen_pdb_prefixes:
                if verbose:
                    print ("PDB structure", pdb_id, "has prefix", pdb_id_prefix, "which has been seen before, deprioritising it for diversity")
                continue
            # add rank based on insertion order
            pdb_id_data["reranked_rank"] = current_rank
            current_rank += 1

            # update list of seen prefixes if considering prefix in reranking
            if consider_prefix_in_reranking: 
                seen_pdb_prefixes.add(pdb_id_prefix)
                
        # add ranks to everything left 
        # (ignore small sequences if structures have already been found)
        allow_small_sequences = current_rank == 1
        for pdb_id in pdb_ids_sorted_by_rank:
            pdb_id_data = pdb_rank_data_for_current_uniprot_accession[pdb_id]
            
            # skip structure if it already has a rank
            if "reranked_rank" in pdb_id_data:
                continue

            # if current rank != 1, then other structeures have been selected and we can ignore small sequences
            if not allow_small_sequences:

                # check sequence length
                if "sequence_length" not in pdb_id_data:
                    continue
                sequence_length = pdb_id_data["sequence_length"]
                if min_sequence_length is not None and sequence_length < min_sequence_length:
                    if verbose:
                        print ("Skipping PDB structure", pdb_id, "due to sequence length:", sequence_length)
                    continue

            # handle mutants
            is_mutant = pdb_id_data["is_mutant"]
            if is_mutant and not allow_mutant:
                if verbose:
                    print ("PDB Structure", pdb_id, "is a mutant, skipping it")
                continue

            # handle unliganded
            number_of_natural_ligands = pdb_id_data["num_natural_ligands"]
            if number_of_natural_ligands == 0 and not allow_unliganded:
                if verbose:
                    print ("PDB Structure", pdb_id, "contains no natural ligands, skipping it")
                continue

            # handle duplicate prefixes
            pdb_id_prefix = pdb_id[:prefix_size]
            if pdb_id_prefix in seen_pdb_prefixes and not allow_duplicate_prefixes:
                if verbose:
                    print ("PDB structure", pdb_id, "has prefix", pdb_id_prefix, "which has been seen before, skipping it")
                continue

            # assign rank
            pdb_id_data["reranked_rank"] = current_rank
            current_rank += 1

        # delete all data about PDB IDs with no reranked_rank field
        pdb_ids_to_delete = {
            pdb_id 
            for pdb_id, pdb_id_data in pdb_rank_data_for_current_uniprot_accession.items()
            if "reranked_rank" not in pdb_id_data
        }
        for pdb_id_to_delete in pdb_ids_to_delete:
            if verbose:
                print ("Deleting", accession, pdb_id_to_delete, "from ranking")
            del pdb_rank_data_for_current_uniprot_accession[pdb_id_to_delete]

        all_ranks[accession] = pdb_rank_data_for_current_uniprot_accession

    return all_ranks

if __name__ == "__main__":

    accession_to_pdb = {
        "P09874": {
            "1UK0_AB",
            "1UK0_A",
        },
    }

    prepare_all_targets(
        accession_to_pdb=accession_to_pdb,
        output_dir="checkme",
        decompress_to_directory="checkmedecompress",
        allow_multi_model=True,
        n_proc=1,
        verbose=True,
    )

    # target_data = prepare_single_target(
    #     target_identifier="6KVA",
    #     existing_target_filename=None,
    #     output_dir="my_prepared_target_output_dir",
    #     decompress_to_directory="my_decompress_to_directory",
    #     allow_multi_model=True,
    #     cofactors_to_keep=[],
    #     verbose=True,
    # )

    # for key, v in target_data.items():
    #     del v["natural_ligands"]

    # write_json(target_data, "checkme.json")