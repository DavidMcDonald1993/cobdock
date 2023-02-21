if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        )))


import os
from utils.io.io_utils import delete_directory, delete_file
from utils.molecules.openbabel_utils import obabel_convert 

from utils.sys_utils import execute_system_command
# from utils.molecules.openbabel_utils import pqr_to_pdb

def protonate_pdb(
    input_filename: str,
    pqr_output_filename: str = None,
    pdb_output_filename: str = None,
    pH: float = 7.4,
    ff: str = "PARSE",
    drop_water: bool = True,
    titration_state_method: str = "propka",
    # titration_state_method: str = None,
    return_as_pdb: bool = True,
    overwrite: bool = False,
    main_job_id: int = None,
    verbose: bool = False,
    ):
    """Use pdb2pqr to protonate protein.

    Parameters
    ----------
    input_filename : str
        Input PDB filename
    output_filename : str, optional
        Desired output filename, will be generated based on input_filename if not given, by default None
    pH : float, optional
        Desired pH level, by default 7.4
    return_as_pdb : bool, optional
        Flag to return in PDB format, by default True
    main_job_id : int, optional
        Job ID to associate with, by default None
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    str
        Output filename
    """

    base, ext = os.path.splitext(input_filename)
    
    if pqr_output_filename is None:
        pqr_output_filename = base + f"_protonated_pH={pH}.pqr"
    
    if not pqr_output_filename.endswith(".pqr"):
        pqr_output_filename += ".pqr"

    if pdb_output_filename is None:
        pdb_output_filename = pqr_output_filename.replace(".pqr", ".pdb")


    if verbose:
        print("Protonating input file", input_filename, "to pH", pH)
        print ("Outputting PQR to", pqr_output_filename)
        print ("Outputting PDB to", pdb_output_filename)

    if return_as_pdb:
        output_filename = pdb_output_filename
    else:
        output_filename = pqr_output_filename

    # check for file already existing
    if not overwrite and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        if verbose:
            print (output_filename, "already exists, skipping protonation")
        return output_filename


    # convert PDB to PDB to fix formatting issues
    input_filename = obabel_convert(
        input_format="pdb",
        input_filename=input_filename,
        output_format="pdb",
        output_filename=input_filename,
        overwrite=True,
        verbose=verbose,
    )


    apbs_input_filename = f"{base}.in"

    cmd = f"pdb2pqr30 {input_filename} {pqr_output_filename} --with-ph={pH} --ff={ff} --apbs-input={apbs_input_filename} --pdb-output {pdb_output_filename}"
    if drop_water:
        cmd += " --drop-water"
    if titration_state_method is not None:
        cmd += f" --titration-state-method={titration_state_method}"

    try:
        execute_system_command(cmd, main_job_id=main_job_id, verbose=verbose)
    except Exception as e:
        print ("Protonation exception", e)
        # raise e
        return None 
    # delete apbs input
    delete_file(apbs_input_filename, verbose=verbose)
    # delete log filename
    log_filename = pqr_output_filename.replace(".pqr", ".log")
    delete_file(log_filename, verbose=verbose)

    if not os.path.exists(pqr_output_filename) or not os.path.exists(pdb_output_filename):
        print ("Missing protonated output filename")
        return None 

    if return_as_pdb:
        delete_file(pqr_output_filename, verbose=verbose)
    else:
        delete_file(pdb_output_filename, verbose=verbose)

    return output_filename

if __name__ == "__main__":

    input_filename = "checkmeobabel.pdb"
    # input_filename = "test_compounds/aspirin.pdb"
    pdb_output_filename = protonate_pdb(
        input_filename,
        pH=7.4,
        verbose=True)

    print (pdb_output_filename)
