
if __name__ == "__main__":

    # load num processes etc.
    from dotenv import load_dotenv
    load_dotenv()

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        )))


import os, shutil

from cobdock.docking.blind_docking_utils import predocking_preparation
from cobdock.data_collation import collate_all_data
from cobdock.post_docking import execute_post_docking

from cobdock.docking.plants.plants_utils import PLANTS_LOCATION
from cobdock.docking.galaxydock.galaxydock_utils import RUN_GALAXYDOCK_LOCATION

from utils.io.io_utils import delete_directory

def execute_cobdock(
    ligands_to_targets: dict,
    output_dir: str,
    submitted_ligand_pdb_files: dict = {}, # mapping from ligand name to .pdb file
    submitted_target_pdb_files: dict = {}, # mapping from target name to .pdb file
    map_uniprot_to_pdb: bool = False, # uniprot input
    number_of_pdb: int = 1,
    allow_mutant: bool = True, # mutant pdb file targets 
    allow_engineered: bool = True, # engineered pdb file targets  
    bounding_box_scale: float = 1., 
    max_box_size: float = None, # max dimension of docking box 
    compute_rmsd_with_submitted_ligand: bool = False,
    rerank_with_prefix: bool = True,
    prefix_size: int = 3,
    fill_pdb_list_with_similar_targets: bool = False,
    use_alphafold: bool = False,

    delete_output_directory: bool = False,
    delete_prepared_targets_directory: bool = False,

    run_post_docking_analysis: bool = True, 

    commercial_use_only: bool = False,
    
    # final output
    local_docking_program: str = "plants",
    num_poses: int = 10, # 10 poses per pocket
    num_complexes: int = None, # convert all poses into complexes
    num_top_pockets: int = 3, # number of top-ranked pockets to execute local docking into
    top_pocket_distance_threshold: float = 10, # at least 10A between selected pockets 

    verbose: bool = True,
    ):

    docking_root_output_dir = os.path.join(output_dir, "docking")

    all_natural_ligands_filename = os.path.join(docking_root_output_dir, "all_natural_ligands.json")
    fpocket_data_filename = os.path.join(docking_root_output_dir, "fpocket_data.json")
    p2rank_data_filename = os.path.join(docking_root_output_dir, "p2rank_data.json")

    vina_data_filename = os.path.join(docking_root_output_dir, "vina_data.json")
    galaxydock_data_filename = os.path.join(docking_root_output_dir, "galaxydock_data.json")
    plants_data_filename = os.path.join(docking_root_output_dir, "plants_data.json")
    zdock_data_filename = os.path.join(docking_root_output_dir, "zdock_data.json")

    ligands_to_targets, prepared_targets_dir, ligand_structure_output_dir = predocking_preparation(
        ligands_to_targets=ligands_to_targets,
        output_dir=output_dir,
        submitted_ligand_structure_files=submitted_ligand_pdb_files,
        submitted_target_pdb_files=submitted_target_pdb_files,
        map_uniprot_to_pdb=map_uniprot_to_pdb,
        number_of_pdb=number_of_pdb,
        allow_mutant=allow_mutant,
        allow_engineered=allow_engineered,
        bounding_box_scale=bounding_box_scale,
        max_bounding_box_size=max_box_size,
        consider_prefix_in_reranking=rerank_with_prefix,
        prefix_size=prefix_size,
        fill_pdb_list_with_similar_targets=fill_pdb_list_with_similar_targets,
        compute_voxel_locations=True,
        determine_natural_ligands=True,
        run_fpocket=False,
        run_fpocket_old=True,
        run_p2rank=True,
        use_alphafold=use_alphafold, # TODO
        max_pockets_to_keep=None, # retain data of all pockets 
        keep_cofactors=False, # TODO: GalaxyDock requires separate files for these 
        verbose=verbose,
    )    

    ####################################################################################################################                             
    ##########                               Docking and Location Analysis                                  ############
    ####################################################################################################################

    ligands_to_targets = collate_all_data(
        ligands_to_targets=ligands_to_targets,
        output_dir=docking_root_output_dir,

        commercial_use_only=commercial_use_only,
        
        all_target_natural_ligands_filename=all_natural_ligands_filename,
        fpocket_data_filename=fpocket_data_filename,
        p2rank_data_filename=p2rank_data_filename,
        vina_data_filename=vina_data_filename,
        galaxydock_data_filename=galaxydock_data_filename,
        plants_data_filename=plants_data_filename,
        zdock_data_filename=zdock_data_filename,

        delete_output_directory=delete_output_directory,
        compute_rmsd_with_submitted_ligand=compute_rmsd_with_submitted_ligand,
        num_complexes=0, # complexes are generated in post_docking 
        verbose=verbose,
    )

    if verbose:
        print ("All docking and data collation has been completed",)

   
    # ####################################################################################################################
    # ##########                                 Post-Docking                                     ########################
    # ####################################################################################################################

    # select local docking program
    # vina is default
    if commercial_use_only \
        or (local_docking_program == "plants" and not os.path.exists(PLANTS_LOCATION))\
        or (local_docking_program == "galaxydock" and not os.path.exists(RUN_GALAXYDOCK_LOCATION)):
        if verbose:
            print ("Setting local docking program to Vina")
        local_docking_program = "vina"

    if run_post_docking_analysis:

        # top five pocket locations per pair
        post_docking_output_dir = os.path.join(output_dir, "pocket_locations")
        
        all_local_docking_collated_data = execute_post_docking(
            ligands_to_targets=ligands_to_targets,
            output_dir=post_docking_output_dir,
            num_top_pockets=num_top_pockets,
            num_poses=num_poses,
            num_complexes=num_complexes,
            commercial_use_only=commercial_use_only,
            local_docking_program=local_docking_program,
            top_pocket_distance_threshold=top_pocket_distance_threshold,
            verbose=verbose,
        )

    else:
        all_local_docking_collated_data = {}
    
    if delete_prepared_targets_directory:
        if verbose: 
            print ("Removing prepared targets directory", prepared_targets_dir)
            print ("Removing prepared ligands directory", ligand_structure_output_dir)
        delete_directory(prepared_targets_dir, verbose=verbose)
        delete_directory(ligand_structure_output_dir, verbose=verbose)

    if verbose:
        print ("COBDock successfully completed")

    return all_local_docking_collated_data


if __name__ == "__main__":


    ligands_to_targets = {
        "aspirin": {
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "targets": ["1OXR_A", ],
        },
        "ibuprofen": {
            "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
            "targets": ["3P6H_A", ],
        },
        "Olaparib": {
            "smiles": "O=C1C2=CC=CC=C2C(CC3=CC(C(N4CCN(CC4)C(C5CC5)=O)=O)=C(C=C3)F)=NN1",
            "targets": ["4TVJ_A"],
        }
    }

    output_dir = "output"

    commercial_use_only = False
    # commercial_use_only = True

    execute_cobdock(
        ligands_to_targets=ligands_to_targets,
        output_dir=output_dir,

        map_uniprot_to_pdb=False,

        # map_uniprot_to_pdb=True,
        # number_of_pdb=1,

        commercial_use_only=commercial_use_only,

        local_docking_program="plants",
        num_top_pockets=3,
        num_poses=10,
        num_complexes=None,

        verbose=True,
    )
