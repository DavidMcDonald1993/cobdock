import os 
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    os.pardir,
    os.pardir,
    os.pardir,
    ))

if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, PROJECT_ROOT)

import numpy as np

from utils.io.io_utils import delete_directory, delete_file, write_json, write_compressed_pickle
from utils.molecules.openbabel_utils import obabel_convert
from utils.sys_utils import execute_system_command

MGLTOOLS_LOCATION = os.path.join(PROJECT_ROOT, "MGLTools-1.5.6")

PYTHONSH_LOCATION = os.path.join(MGLTOOLS_LOCATION, "bin", "pythonsh")

AUTODOCK_TOOLS_DIR = os.path.join(MGLTOOLS_LOCATION, "MGLToolsPckgs", "AutoDockTools", "Utilities24")
PREPARE_LIGAND_LOCATION = os.path.join(AUTODOCK_TOOLS_DIR, "prepare_ligand4.py")
BACKUP_PREPARE_LIGAND_LOCATION = os.path.join(AUTODOCK_TOOLS_DIR, "prepare_ligand.py")
PREPARE_TARGET_LOCATION = os.path.join(AUTODOCK_TOOLS_DIR, "prepare_receptor4.py")
BACKUP_PREPARE_TARGET_LOCATION = os.path.join(AUTODOCK_TOOLS_DIR, "prepare_receptor.py")

VINA_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "vina", "vina")
QVINA_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "vina", "qvina02")
PSOVINA_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "vina", "psovina")
SMINA_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "vina", "smina")
SMINA_FEATURE_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "vina", "smina_feature")

VINA_MAX_MODES = 100

VINA_N_PROC = int(os.environ.get("VINA_N_PROC", default=1))
VINA_TIMEOUT = os.environ.get("VINA_TIMEOUT", default="1h")

CUSTOM_SCORE_FUNCTION_ROOT_DIR = os.path.join(
    PROJECT_ROOT, 
    "data", 
    "vina_score_functions")

VINA_48_LOCATION = os.path.join(CUSTOM_SCORE_FUNCTION_ROOT_DIR, "vina_48.txt")

# check here for more info: https://github.com/mwojcikowski/smina/blob/master/README
VINA_VARIANTS = {"vina", "qvina", "psovina", "smina", "vinardo", "lin_f9", "ad4_scoring", "dkoes_fast", "dkoes_scoring", }

NUM_VINA_TERMS = {
    "vina": 6,
    "vinardo": 5,
    "ad4_scoring": 5,
    "dkoes_fast": 4,
    "dkoes_scoring": 4,
    "Lin_F9": 9,
    "vina_48": 48, 
}

VALID_AUTODOCK_ATOMS = {
    "H", # hydrogen
    "C", # carbon
    "N", # nitrogen
    "O", # oxygen
    "F", # Fluorine
    "Mg", # magnesium
    "P", # phosphorus
    "S", # sulphur
    "Cl", # chlorine
    "Ca", # calcium
    "Mn", # manganese
    "Fe", # iron
    "Zn", # zinc
    "Br", # bromine
    "I", # iodine
}

# gas constant
k = 0.001987
# temperature in Kelvin
T = 298


# https://www.novoprolabs.com/tools/deltag2kd
def convert_vina_energy_to_Kd(
    vina_energy: float,
    ):

    return np.exp(vina_energy / (k * T))

def convert_vina_energy_to_pKd(
    vina_energy: float,
    ):

    return -np.log10(np.exp(vina_energy / (k * T)))

def contains_only_valid_autodock_atoms(
    mol,
    ):
    if mol is None:
        return False

    elements = {
        atom.GetSymbol()
        for atom in mol.GetAtoms()
    }

    for element in elements:
        if element not in VALID_AUTODOCK_ATOMS:
            return False

    return True

def prepare_ligand_for_vina(
    ligand_filename,
    output_filename=None,
    overwrite=False,
    verbose: bool = True,
    ):

    if verbose:
        print ("Preparing ligand", ligand_filename, "for Vina")

    stem, ext = os.path.splitext(ligand_filename)
    # assume ext is input format
    ext = ext.replace(".", "")

    if output_filename is None:
        output_filename = stem + ".pdbqt"
   
    cmd = f'''\
    {PYTHONSH_LOCATION} {PREPARE_LIGAND_LOCATION}\
        -l {ligand_filename}\
        -o {output_filename}
    
    '''

    if not os.path.exists(output_filename) or ligand_filename == output_filename or overwrite:
        try:
            execute_system_command(cmd, verbose=verbose)
            return output_filename
        except Exception as e:
            print ("VINA PREPARE LIGAND exception", e)

            # try openbabel 
            output_filename = obabel_convert(
                input_format=ext,
                input_filename=ligand_filename,
                output_format="pdbqt",
                output_filename=output_filename,
                verbose=verbose,
            )
            # output filename will be None if obabel fails

    return output_filename

def prepare_target_for_vina(
    target_filename,
    output_filename=None,
    overwrite=False,
    verbose: bool = True,
    ):
    
    if verbose:
        print ("Preparing target", target_filename, "for Vina")

    if output_filename is None:
        stem, _ = os.path.splitext(target_filename)
        output_filename = stem + ".pdbqt"

    if os.path.exists(output_filename) and not overwrite:
        return output_filename

    for prepare_target_location in (
        PREPARE_TARGET_LOCATION, 
        ):

        cmd = f'''\
        {PYTHONSH_LOCATION} {prepare_target_location}\
            -r {target_filename}\
            -o {output_filename}
        '''

        if not os.path.exists(output_filename) or target_filename == output_filename or overwrite:

            try:
                execute_system_command(cmd, verbose=verbose)
                return output_filename
            
            except Exception as e:
                print ("VINA PREPARE TARGET exception", e)
                # try to target convert to mol2 format
                if verbose:
                    print ("Coverting receptor to mol2 format")
                stem, ext = os.path.splitext(target_filename)
                ext = ext.replace(".", "")
                target_filename = obabel_convert(
                    input_filename=target_filename,
                    input_format=ext,
                    output_format="mol2",
                    verbose=verbose
                )

                cmd = f'''\
                {PYTHONSH_LOCATION} {prepare_target_location}\
                    -r {target_filename}\
                    -o {output_filename}
                '''
                try:
                    execute_system_command(cmd, verbose=verbose)
                    return output_filename
                
                except Exception as e:
                    pass

    return None # fail

def execute_vina(
    ligand_filename: str, 
    target_filename: str,
    output_filename: str, 
    log_filename: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    exhaustiveness: int = 8,
    energy_range: int = 3,
    seed: int = None,
    n_proc: int = None,
    num_poses: int = None,
    prepare_ligand: bool = False,
    prepare_target: bool = False,
    vina_variant: str = "vina",
    verbose: bool = True,
    ):

    if n_proc is None:
        n_proc = VINA_N_PROC
    if num_poses is None:
        num_poses = VINA_MAX_MODES

    log_json_filename = log_filename + ".json"

    if verbose:
        print ("Executing Vina for ligand", ligand_filename, "and target", target_filename)
        print ("Using", n_proc, "process(es)")
        print ("Generating", num_poses, "binding pose(s)")
        print ("Writing to output file", output_filename)
        print ("Logging to", log_json_filename)

    if os.path.exists(output_filename) and os.path.exists(log_json_filename):
        if verbose: 
            print (output_filename, "and", log_json_filename, "already exist, skipping docking")
        return output_filename, log_json_filename

    if prepare_ligand or not ligand_filename.endswith(".pdbqt"):
        ligand_filename = prepare_ligand_for_vina(ligand_filename)
        if ligand_filename is None:
            return None

    if prepare_target or not target_filename.endswith(".pdbqt"):
        target_filename = prepare_target_for_vina(target_filename)
        if target_filename is None:
            return None # error

    args = f'''\
    --receptor {target_filename} \
    --ligand {ligand_filename} \
    --out {output_filename} \
    --log {log_filename} \
    --exhaustiveness {exhaustiveness} \
    --center_x {center_x} \
    --center_y {center_y} \
    --center_z {center_z} \
    --size_x {size_x} \
    --size_y {size_y} \
    --size_z {size_z} \
    --energy_range {energy_range} \
    --num_modes {num_poses}\
    '''
    if VINA_N_PROC is not None:  
        args += f" --cpu {n_proc}"
    if seed is not None: # reproducability
        args += f" --seed {seed}"

    if vina_variant.lower() == "qvina" and os.path.exists(QVINA_LOCATION):
        if verbose:
            print ("Using QVina")
        vina_location = QVINA_LOCATION
    elif vina_variant.lower() == "psovina" and os.path.exists(PSOVINA_LOCATION):
        if verbose:
            print ("Using PSOVina")
        vina_location = PSOVINA_LOCATION
    elif vina_variant.lower() in {"smina", "vinardo", "lin_f9", } and os.path.exists(SMINA_LOCATION):
        if verbose:
            print ("Using SMINA")
        vina_location = SMINA_LOCATION

        # select scoring function
        smina_scoring = "vinardo"
        if vina_variant.lower() == "lin_f9":
            smina_scoring = "Lin_F9"
        elif vina_variant.lower() == "vina":
            smina_scoring = "vina"
        
        args += f" --scoring {smina_scoring}"
    else:
        if verbose:
            print ("Using Vina")
        vina_location = VINA_LOCATION

    cmd = f"{vina_location} {args}"

    try: # catch invalid atoms
        execute_system_command(cmd, timeout=VINA_TIMEOUT, verbose=verbose)
    except Exception as e:
        print ("VINA exception", e)
        return None

  
    if os.path.exists(log_filename):
        log_as_json = read_vina_log_file(log_filename, verbose=verbose)
        write_json(log_as_json, log_json_filename, verbose=verbose)
        # delete log filename 
        delete_file(log_filename, verbose=verbose)

    return output_filename, log_json_filename

def score_with_vina(
    ligand_filename: str, 
    target_filename: str,
    log_filename: str,
    score_function: str = "vinardo",
    log_pickle_filename: str = None,
    verbose: bool = True,
    ):

    # ligand and target can be pdb format

    if verbose:
        print ("Using Vina to score ligand", ligand_filename, "and target", target_filename)

    # if prepare_ligand or not ligand_filename.endswith(".pdbqt"):
    #     ligand_filename = prepare_ligand_for_vina(ligand_filename)
    #     if ligand_filename is None:
    #         return None

    # if prepare_target or not target_filename.endswith(".pdbqt"):
    #     target_filename = prepare_target_for_vina(target_filename)
    #     if target_filename is None:
    #         return None # error

    args = f'''\
    --score_only \
    --receptor {target_filename} \
    --ligand {ligand_filename} \
    --log {log_filename} \
    '''

    vina_location = SMINA_FEATURE_LOCATION

    if score_function == "vina_48":
        args += f" --custom_scoring {VINA_48_LOCATION}"
    elif score_function in {"vinardo", "vina", "ad4_scoring", "Lin_F9", "dkoes_fast", "dkoes_scoring"}:
        args += f" --scoring {score_function}"
    else:
        raise NotImplementedError(score_function)

    cmd = f"{vina_location} {args}"

    try: # catch invalid atoms
        execute_system_command(cmd, verbose=verbose)
    except Exception as e:
        print ("VINA exception", e)
        return None
    
    log_as_list = None

    if os.path.exists(log_filename):

        try:
            # read log file 
            with open(log_filename, "r") as f:
                line = f.readline().strip()
                # line starts with ## 
                line = line[3:]
                log_as_list = list(map(float, line.split()))
                # for i, val in enumerate(line.split()):
                #     log_as_list[f"field_{i}"] = val

            if log_pickle_filename is not None:
                # write_json(log_as_list, log_pickle_filename, verbose=verbose)
                write_compressed_pickle(log_as_list, log_pickle_filename, verbose=verbose)
        except Exception as e:
            pass 

        # delete log filename 
        delete_file(log_filename, verbose=verbose)

    return log_as_list

def read_vina_log_file(
    vina_log_file,
    compute_pKd: bool = True,
    verbose: bool = True,
    ):

    if verbose:
        print ("Reading Vina energies from Vina log file located at", vina_log_file)

    lines = []
    with open(vina_log_file, "r") as f:
        start_reading = False
        for line in map(str.rstrip, f.readlines()):
            if line ==  "-----+------------+----------+----------":
                start_reading = True
                continue
            # if re.match(r"^ {0,1,2,3}[1-9]+", line):
            if start_reading:
                line_split = line.split()
                if len(line_split) == 4:
                    lines.append(line_split)

    modes = {}
    for mode, energy, rmsd_lb, rmsd_ub in lines:
        try:
            mode = int(mode)
            energy = float(energy)
            rmsd_lb = float(rmsd_lb)
            rmsd_ub = float(rmsd_ub)
            mode_data = {
                "mode": mode,
                "energy": energy,
                "rmsd_lb": rmsd_lb,
                "rmsd_ub": rmsd_ub,
            }
            if compute_pKd:
                mode_data["pKd"] =  convert_vina_energy_to_pKd(energy)
            modes[mode] = mode_data
        except: 
            pass

    return modes

def convert_and_separate_vina_out_file(
    vina_output_filename,
    conversion_dir,
    ligand_id,
    output_format="pdb",
    output_filename="pose_",
    verbose: bool = True,
    ):
    delete_directory(conversion_dir)
    if verbose:
        print ("Converting and separting Vina out file:", vina_output_filename)
    os.makedirs(conversion_dir, exist_ok=True,)
    # read out file and convert from pdbqt to pdb
    obabel_convert(
        input_format="pdbqt",
        input_filename=vina_output_filename,
        output_format=output_format,
        output_filename=output_filename,
        output_dir=conversion_dir,
        title=ligand_id,
        multiple=True,
        append=":MODEL", # append model id (pose_num) to title (use : delimiter?) 
        verbose=verbose,
    )
    # delete vina_output_filename
    # delete_file(vina_output_filename, verbose=verbose)
    # return generator of separate ligand pdb files
    return glob.glob(os.path.join(conversion_dir, f"{output_filename}*.{output_format}"))


if __name__ == "__main__":

    # execute_vina(
    #     ligand_filename="ligand.pdbqt",
    #     target_filename="target.pdbqt",
    #     output_filename="vina_out.pdbqt",
    #     center_x=0,
    #     center_y=0,
    #     center_z=0,
    #     size_x=30,
    #     size_y=30,
    #     size_z=30,
    #     log_filename="vina_out.log",
    #     vina_variant="psovina",
    #     num_poses=20,

    # )

    # convert_and_separate_vina_out_file(
    #     vina_output_filename="vina_out.pdbqt",
    #     conversion_dir="my_conversion_dir",
    #     ligand_id="my_ligand",
    #     verbose=True,
    # )

    prepare_ligand_for_vina(
        ligand_filename="AR.pdb",
        output_filename="AR1.pdbqt"
    )

    prepare_ligand_for_vina(
        ligand_filename="AR.mol2",
        output_filename="AR2.pdbqt"
    )