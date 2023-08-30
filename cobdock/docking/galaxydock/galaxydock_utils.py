

import os, sys

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

import pandas as pd

from utils.molecules.openbabel_utils import obabel_convert
from utils.sys_utils import execute_system_command
from utils.io.io_utils import copy_file, write_json

GALAXYDOCK_VERSION = 3
GALAXYDOCK_HOME = os.path.join(PROJECT_ROOT, "bin", "docking", f"GalaxyDock{GALAXYDOCK_VERSION}")

GALAXYDOCK_OUT_DIRECTORY = f"GALAXYDOCK{GALAXYDOCK_VERSION}_out"

CALCULATE_RMSD_SCRIPT_LOCATION = os.path.join(GALAXYDOCK_HOME, "script", "calc_RMSD.py")
RUN_GALAXYDOCK_LOCATION = os.path.join(GALAXYDOCK_HOME, "script", "run_GalaxyDock3.py")

GALAXYDOCK_MAX_POSES = 100

GALAXYDOCK_N_PROC = int(os.environ.get("GALAXYDOCK_N_PROC", default=1))
GALAXYDOCK_TIMEOUT = os.environ.get("GALAXYDOCK_TIMEOUT", default="1h")

def prepare_ligand_for_galaxydock(
    input_filename: str,
    output_filename: str = None,
    overwrite: bool = False,
    verbose: bool = True,
    ):
    if verbose:
        print ("Preparing ligand", input_filename, "for GalaxyDock by converting to mol2 format")
    stem, ext = os.path.splitext(input_filename)
    ext = ext.replace(".", "")
    if output_filename is None:
        output_filename = stem + ".mol2"
    if verbose:
        print ("Outputting to", output_filename)

    output_filename = obabel_convert(
        input_format=ext,
        input_filename=input_filename,
        output_format="mol2",
        output_filename=output_filename,
        overwrite=overwrite,
        verbose=verbose,
    )

    return output_filename


def prepare_target_for_galaxydock(
    input_filename: str,
    output_filename: str = None,
    overwrite: bool = False,
    verbose: bool = True,
    ):
    if verbose:
        print ("Preparing target", input_filename, "for GalaxyDock by converting to pdb format")
    stem, ext = os.path.splitext(input_filename)
    ext = ext.replace(".", "")
    if output_filename is None:
        output_filename = stem + ".pdb"
    if verbose:
        print ("Outputting to", output_filename)

    output_filename = obabel_convert(
        input_format=ext,
        input_filename=input_filename,
        output_format="pdb",
        output_filename=output_filename,
        overwrite=overwrite,
        verbose=verbose,
    )

    return output_filename

def run_GalaxyDock3_python3(
    input_filename: str,
    log_filename: str,
    home_dir: str,
    pdb_fn: str,
    lig_fn: str,
    cntr_x: float,
    cntr_y: float,
    cntr_z: float,
    n_elem_x: int,
    n_elem_y: int,
    n_elem_z: int,
    e0max: float = 1000.0,
    e1max: float = 1000000.0,
    # timeout: str = "1h",
    n_proc: int = 1,
    verbose: bool = True,
    ):

    # python3 implementation of run_GalaxyDock3.py

    sys.stdout.write("Running GalaxyDock....\n") 
    HOME = os.path.abspath(home_dir)

    EXEC_GALAXYDOCK3 = '%s/bin/GalaxyDock3'%HOME
    #
    with open('%s/input/galaxydock.in'%HOME, "r") as f:
        INPUT = f.read()
    input = INPUT.replace('[GALAXY_DOCK_HOME]', HOME)
    input = input.replace('[RECEPTOR_PDB]', os.path.relpath(pdb_fn))
    input = input.replace('[LIGAND_MOL2]', os.path.relpath(lig_fn))
    input = input.replace('[GRID_BOX_CENTER]', ' '.join(map(str, [cntr_x, cntr_y, cntr_z])))
    input = input.replace('[N_ELEM]', ' '.join(map(str, [n_elem_x, n_elem_y, n_elem_z])))
    input = input.replace('[E0MAX]', str(e0max))
    input = input.replace('[E1MAX]', str(e1max))

    #
    # if opt.cofac_fn_s != None:
    #     for cofac_fn in opt.cofac_fn_s:
    #         resName = cofac_fn.prefix()
    #         input += 'infile_mol2_topo  %s  %s\n'%(cofac_fn, resName)
    if n_proc != None:
        n_proc = int(n_proc)
        if not n_proc == 1:
            input += 'n_proc %d\n'%(n_proc)
            EXEC_GALAXYDOCK3 = '%s/bin/GalaxyDock3.openmp'%HOME

    # fout = file('galaxydock.in', 'wt')
    # fout.write(input)
    # fout.close()
    with open(input_filename, 'wt') as f:
        f.write(input)
    #
    # os.system(f'{EXEC_GALAXYDOCK3} {input_filename} > {log_filename}')
    execute_system_command(f'{EXEC_GALAXYDOCK3} {input_filename} > {log_filename}', timeout=GALAXYDOCK_TIMEOUT, verbose=verbose)
    if not os.path.exists('GD3_fb.E.info'):
        print ('''ERROR: Failed to run GalaxyDock.''')
        # sys.exit()
    #
    sys.stdout.write("Done.\n") 


    return 

def execute_galaxydock(
    ligand_filename: str,
    target_filename: str,
    output_dir: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    use_multiprocessing: bool = False,
    n_proc: int = None,
    num_poses: int = None,
    verbose: bool = True,
    ):

    if num_poses is None: # galaxydock cannot control this, keeping for consistency
        num_poses = GALAXYDOCK_MAX_POSES

    if verbose:
        print ("Executing GalaxyDock for ligand", ligand_filename, "and target", target_filename)
        print ("Using", n_proc, "process(es)")
        print ("Generating", num_poses, "binding pose(s)")
        print ("Outputting to", output_dir)
    
    # score filename 
    galaxydock_output_filename = os.path.join(
        output_dir,
        f"GD{GALAXYDOCK_VERSION}_fb.E.info",
    ) 

    # output pose filename 
    pose_mol2_file = os.path.join(
        output_dir, 
        f"GD{GALAXYDOCK_VERSION}_fb.mol2"
    )

    # pose data
    all_pose_data_filename = os.path.join(output_dir, "galaxydock_pose_data.json")


    if verbose:
        print ("Scores will be located at", galaxydock_output_filename)
        print ("Output poses will be located at", pose_mol2_file)
        print ("Pose data will be written to", all_pose_data_filename)

    if os.path.exists(all_pose_data_filename) and os.path.exists(pose_mol2_file):
        if verbose:
            print (all_pose_data_filename, "and", pose_mol2_file, "already exist, skipping docking")
        
        return pose_mol2_file, all_pose_data_filename 

    current_dir = os.getcwd()

    if not ligand_filename.endswith(".mol2"):
        ligand_filename = prepare_ligand_for_galaxydock(ligand_filename, verbose=verbose)
        if ligand_filename is None:
            return None 

    if not target_filename.endswith(".pdb"):
        target_filename = prepare_target_for_galaxydock(target_filename, verbose=verbose)
        if target_filename is None:
            return None

    ligand_filename = os.path.abspath(ligand_filename)
    target_filename = os.path.abspath(target_filename)

    os.makedirs(output_dir, exist_ok=True)

    # copy ligand and target to output_dir
    ligand_basename = os.path.basename(ligand_filename)
    target_basename = os.path.basename(target_filename)
    ligand_filename_in_output_dir = os.path.join(output_dir, ligand_basename)
    target_filename_in_output_dir = os.path.join(output_dir, target_basename)
    copy_file(ligand_filename, ligand_filename_in_output_dir, verbose=verbose)
    copy_file(target_filename, target_filename_in_output_dir, verbose=verbose)

    # run in output dir
    os.chdir(output_dir)

    ligand_filename = os.path.basename(ligand_filename)
    target_filename = os.path.basename(target_filename)

    if n_proc is None:
        n_proc = GALAXYDOCK_N_PROC

    if GALAXYDOCK_N_PROC == 1:
        use_multiprocessing = False 

    try:

        n_elem_x = int( size_x / 0.375 ) + 1
        n_elem_y = int( size_y / 0.375 ) + 1
        n_elem_z = int( size_z / 0.375 ) + 1

        pair_id = ligand_basename + "_" + target_basename
        config_filename = pair_id + ".in"
        log_filename = pair_id + ".log"

        run_GalaxyDock3_python3(
            input_filename=config_filename,
            log_filename=log_filename,
            home_dir=GALAXYDOCK_HOME,
            pdb_fn=target_filename,
            lig_fn=ligand_filename,
            cntr_x=center_x,
            cntr_y=center_y,
            cntr_z=center_z,
            n_elem_x=n_elem_x,
            n_elem_y=n_elem_y,
            n_elem_z=n_elem_z,
            verbose=verbose,
        )

    except Exception as e:
        print ("GalaxyDock exception", e)
    finally:
        # change back to current_dir
        os.chdir(current_dir)

    if verbose:
        print ("Reading GalaxyDock scores from", galaxydock_output_filename)
    galaxydock_output_df = pd.read_fwf(
        galaxydock_output_filename, 
        comment="!", 
        sep="\t", 
        index_col=0)
    # necessary
    galaxydock_output_df = galaxydock_output_df.fillna(0)

    all_pose_data = {}

    for rank in galaxydock_output_df.index:

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
            "rmsd": galaxydock_rmsd,
            "score": galaxydock_score,
            "autodock": galaxydock_autodock,
            "drug_score": galaxydock_drug_score,
            "internal_energy": galaxydock_internal_energy, 
        }

        all_pose_data[rank] = pose_data

    # write all_pose_data to file
    write_json(all_pose_data, all_pose_data_filename, verbose=verbose)

    return pose_mol2_file, all_pose_data_filename 

if __name__ == "__main__":

    pass