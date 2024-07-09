if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        )))

import os
import pdb2pqr # confirm package

from utils.io.io_utils import delete_directory, delete_file
from utils.molecules.openbabel_utils import obabel_convert 

from utils.sys_utils import execute_system_command

'''
usage: pdb2pqr [-h] [--ff {AMBER,CHARMM,PARSE,TYL06,PEOEPB,SWANSON}]
               [--userff USERFF] [--clean] [--nodebump] [--noopt]
               [--keep-chain] [--assign-only]
               [--ffout {AMBER,CHARMM,PARSE,TYL06,PEOEPB,SWANSON}]
               [--usernames USERNAMES] [--apbs-input APBS_INPUT]
               [--pdb-output PDB_OUTPUT] [--ligand LIGAND] [--whitespace]
               [--neutraln] [--neutralc] [--drop-water] [--include-header]
               [--titration-state-method {propka}] [--with-ph PH]
               [-f FILENAMES] [-r REFERENCE] [-c CHAINS] [-i TITRATE_ONLY]
               [-t THERMOPHILES] [-a ALIGNMENT] [-m MUTATIONS] [-p PARAMETERS]
               [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-o PH]
               [-w WINDOW WINDOW WINDOW] [-g GRID GRID GRID]
               [--mutator MUTATOR] [--mutator-option MUTATOR_OPTIONS] [-d]
               [-l] [-k] [-q] [--protonate-all] [--version]
               input_path output_pqr

PDB2PQR v3.4.1: biomolecular structure conversion software.

positional arguments:
  input_path            Input PDB path or ID (to be retrieved from RCSB
                        database
  output_pqr            Output PQR path

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Mandatory options:
  One of the following options must be used

  --ff {AMBER,CHARMM,PARSE,TYL06,PEOEPB,SWANSON}
                        The forcefield to use. (default: PARSE)
  --userff USERFF       The user-created forcefield file to use. Requires
                        --usernames and overrides --ff (default: None)
  --clean               Do no optimization, atom addition, or parameter
                        assignment, just return the original PDB file in
                        aligned format. Overrides --ff and --userff (default:
                        False)

General options:
  --nodebump            Do not perform the debumping operation (default: True)
  --noopt               Do not perform hydrogen optimization (default: True)
  --keep-chain          Keep the chain ID in the output PQR file (default:
                        False)
  --assign-only         Only assign charges and radii - do not add atoms,
                        debump, or optimize. (default: False)
  --ffout {AMBER,CHARMM,PARSE,TYL06,PEOEPB,SWANSON}
                        Instead of using the standard canonical naming scheme
                        for residue and atom names, use the names from the
                        given forcefield (default: None)
  --usernames USERNAMES
                        The user-created names file to use. Required if using
                        --userff (default: None)
  --apbs-input APBS_INPUT
                        Create a template APBS input file based on the
                        generated PQR file at the specified location.
                        (default: None)
  --pdb-output PDB_OUTPUT
                        Create a PDB file based on input. This will be missing
                        charges and radii (default: None)
  --ligand LIGAND       Calculate the parameters for the specified MOL2-format
                        ligand at the path specified by this option. (default:
                        None)
  --whitespace          Insert whitespaces between atom name and residue name,
                        between x and y, and between y and z. (default: False)
  --neutraln            Make the N-terminus of a protein neutral (default is
                        charged). Requires PARSE force field. (default: False)
  --neutralc            Make the C-terminus of a protein neutral (default is
                        charged). Requires PARSE force field. (default: False)
  --drop-water          Drop waters before processing biomolecule. (default:
                        False)
  --include-header      Include pdb header in pqr file. WARNING: The resulting
                        PQR file will not work with APBS versions prior to 1.5
                        (default: False)

pKa options:
  Options for titration calculations

  --titration-state-method {propka}
                        Method used to calculate titration states. If a
                        titration state method is selected, titratable residue
                        charge states will be set by the pH value supplied by
                        --with_ph (default: None)
  --with-ph PH          pH values to use when applying the results of the
                        selected pH calculation method. (default: 7.0)

PROPKA invocation options:
  -f FILENAMES, --file FILENAMES
                        read data from <filename>, i.e. <filename> is added to
                        arguments (default: [])
  -r REFERENCE, --reference REFERENCE
                        setting which reference to use for stability
                        calculations [neutral/low-pH] (default: neutral)
  -c CHAINS, --chain CHAINS
                        creating the protein with only a specified chain.
                        Specify " " for chains without ID [all] (default:
                        None)
  -i TITRATE_ONLY, --titrate_only TITRATE_ONLY
                        Treat only the specified residues as titratable. Value
                        should be a comma-separated list of "chain:resnum"
                        values; for example: -i "A:10,A:11" (default: None)
  -t THERMOPHILES, --thermophile THERMOPHILES
                        defining a thermophile filename; usually used in
                        'alignment-mutations' (default: None)
  -a ALIGNMENT, --alignment ALIGNMENT
                        alignment file connecting <filename> and <thermophile>
                        [<thermophile>.pir] (default: None)
  -m MUTATIONS, --mutation MUTATIONS
                        specifying mutation labels which is used to modify
                        <filename> according to, e.g. N25R/N181D (default:
                        None)
  -p PARAMETERS, --parameters PARAMETERS
                        set the parameter file [{default:s}] (default: /home/d
                        avid/miniconda3/envs/aienv_clean/lib/python3.7/site-
                        packages/propka/propka.cfg)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        logging level verbosity (default: INFO)
  -o PH, --pH PH        setting pH-value used in e.g. stability calculations
                        [7.0] (default: 7.0)
  -w WINDOW WINDOW WINDOW, --window WINDOW WINDOW WINDOW
                        setting the pH-window to show e.g. stability profiles
                        [0.0, 14.0, 1.0] (default: (0.0, 14.0, 1.0))
  -g GRID GRID GRID, --grid GRID GRID GRID
                        setting the pH-grid to calculate e.g. stability
                        related properties [0.0, 14.0, 0.1] (default: (0.0,
                        14.0, 0.1))
  --mutator MUTATOR     setting approach for mutating <filename>
                        [alignment/scwrl/jackal] (default: None)
  --mutator-option MUTATOR_OPTIONS
                        setting property for mutator [e.g. type="side-chain"]
                        (default: None)
  -d, --display-coupled-residues
                        Displays alternative pKa values due to coupling of
                        titratable groups (default: False)
  -l, --reuse-ligand-mol2-files
                        Reuses the ligand mol2 files allowing the user to
                        alter ligand bond orders (default: False)
  -k, --keep-protons    Keep protons in input file (default: False)
  -q, --quiet           suppress non-warning messages (default: None)
  --protonate-all       Protonate all atoms (will not influence pKa
                        calculation) (default: False)
'''

def protonate_pdb_using_pdb2pqr(
    input_filename: str,
    pqr_output_filename: str = None,
    pdb_output_filename: str = None,
    pH: float = 7.4,
    # ff: str = "PARSE",
    ff: str = "AMBER", # ?
    drop_water: bool = True,
    titration_state_method: str = "propka",
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
        center=True,
        overwrite=True,
        verbose=verbose,
    )

    apbs_input_filename = f"{base}.in"

    args = [
        input_filename,
        pqr_output_filename,
        f"--pdb-output {pdb_output_filename}",

        f"--ff={ff}",
        f"--pH={pH}",
        f"--apbs-input={apbs_input_filename}",
        "--keep-chain",
        # "--whitespace",
    ]
    if drop_water:
        args.append("--drop-water")
    if titration_state_method is not None:
        args.append(f"--titration-state-method={titration_state_method}")
        args.append(f"--with-ph={pH}")

    # join args 
    args = " ".join(args)

    cmd = f"pdb2pqr30 {args}" 

    if not verbose:
        cmd += " > /dev/null 2>&1"

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

    input_filename = "6nna_clean.pdb"
    # input_filename = "test_compounds/aspirin.pdb"
    pdb_output_filename = protonate_pdb_using_pdb2pqr(
        input_filename,
        return_as_pdb=False,
        pH=7.4,
        verbose=True)

    print (pdb_output_filename)
