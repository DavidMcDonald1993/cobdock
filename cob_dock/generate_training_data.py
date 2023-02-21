
if __name__ == "__main__":

    # load num processes etc.
    from dotenv import load_dotenv
    load_dotenv()


    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        
        )))

import os, shutil

import numpy as np

from utils.io.io_utils import load_json

from ai_blind_docking.algorithms.cob_dock.cob_dock import execute_blind_docking_cob_dock

N_CHUNKS = 107

if __name__ == "__main__":

    root_output_dir = "/media/david/Elements/data/ai_blind_docking/pdbbind-v2020-refined-set-output"

    for chunk_num in range(35, N_CHUNKS ):

        chunk_ligands_to_targets_filename = f"/media/david/Elements/data/ai_blind_docking/pdbbind-v2020-refined-set/chunk_{chunk_num}.json"
        chunk_target_pdb_filenames_filename = f"/media/david/Elements/data/ai_blind_docking/pdbbind-v2020-refined-set/chunk_{chunk_num}_target_pdb_filenames.json"

        chunk_ligands_to_targets = load_json(chunk_ligands_to_targets_filename, verbose=True,)
        chunk_target_pdb_filenames = load_json(chunk_target_pdb_filenames_filename, verbose=True,)

        chunk_output_dir = os.path.join(root_output_dir, f"chunk_{chunk_num}")
        os.makedirs(chunk_output_dir, exist_ok=True,)

        execute_blind_docking_cob_dock(
            ligands_to_targets=chunk_ligands_to_targets,
            output_dir=chunk_output_dir,
            submitted_target_pdb_files=chunk_target_pdb_filenames,
            map_uniprot_to_pdb=-1, # no mapping

            delete_output_directory=True, 
            delete_prepared_targets_directory=False, # TODO set to True
            run_post_docking_analysis=False,
            verbose=True,
        )