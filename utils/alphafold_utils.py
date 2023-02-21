

if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir, 
        )))

import os, glob

from concurrent.futures import ProcessPoolExecutor as Pool

from utils.io.io_utils import delete_file, gunzip_file
from utils.request_utils import make_http_request

# def prepare_alphafold_structures(
#     compressed_alphafold_structures: list,
#     temporary_structure_dir: str = "alphafold_structures",
#     n_proc: int = 24,
#     verbose: bool = True,
#     ):

#     from ai_blind_docking.docking_utils.blind_docking_utils import (
#         PREPARE_TARGETS_ROOT_DIR,
#         prepare_single_target,
#         )

#     if verbose:
#         print ("Preprocessing AlphaFold structures for", len(compressed_alphafold_structures), "files")
#         print ("Outputting to temporary structures to", temporary_structure_dir)
#         print ("Using", n_proc, "process(es)")

#     os.makedirs(temporary_structure_dir, exist_ok=True)

#     with Pool(max_workers=n_proc) as p:

#         running_tasks = []

#         for alphafold_filename in compressed_alphafold_structures:

#             if not os.path.exists(alphafold_filename):
#                 continue

#             basename = os.path.basename(alphafold_filename)
#             accession = basename.split("-")[1]

#             target_identifier = f"alphafold-{accession}"
#             existing_target_filename = os.path.join(temporary_structure_dir, f"AF-{accession}-F1-model_v3.pdb")

#             # decompress
#             existing_target_filename = gunzip_file(
#                 gzip_filename=alphafold_filename,
#                 output_filename=existing_target_filename,
#                 delete_gzip_file=False,
#                 verbose=verbose,
#             )

#             output_dir = os.path.join(PREPARE_TARGETS_ROOT_DIR, accession, target_identifier)

#             task = p.submit(
#                 prepare_single_target,
#                 target_identifier=target_identifier,
#                 existing_target_filename=existing_target_filename,
#                 desired_chain=None,
#                 run_p2rank=False,
#                 output_dir=output_dir,
#                 verbose=verbose,
#             )

#             running_tasks.append(
#                (existing_target_filename, task)
#             )

#         for existing_target_filename, running_task in running_tasks:
#             task_result = running_task.result()
#             delete_file(existing_target_filename, verbose=verbose)

#     return 

def download_alphafold_structure(
    uniprot_accession: str,
    output_filename: str,
    verbose: bool = True,
    ):

    if not output_filename.endswith(".pdb"):
        output_filename += ".pdb"

    if verbose:
        print ("Attempting to download AlphaFold structure for Uniprot accession", uniprot_accession)
        print ("Writing response to file", output_filename)

    output_dir = os.path.dirname(output_filename)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_accession}-F1-model_v3.pdb"

    try:
        response = make_http_request(
            url=url,
            params={},
            stream=False,
            sleep_duration=10,
            max_retries=1,
            verbose=verbose,
        )

        if response is not None:
            with open(output_filename, "w") as f:
                f.write(response.text)
            return output_filename
            
    except Exception as e:
        print ("Download AlphaFold exception", e)
        # raise e
    return None

if __name__ == "__main__":

    # compressed_alphafold_structures = glob.glob("/home/david/Downloads/alphafold/pdb/*.pdb.gz")

    # prepare_alphafold_structures(
    #     compressed_alphafold_structures,
    #     n_proc=10,
    #     verbose=True,
    # )


    # tuberculosis_accession = "P96368"

    # output_filename = f"testdir/alphafold-{tuberculosis_accession}"

    accession = "P49327"

    download_alphafold_structure(
        accession,
        f"AF-{accession}-F1-model_v3.pdb",
        verbose=True,
    )