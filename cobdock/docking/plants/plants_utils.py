

import os 
import re
import glob

from textwrap import dedent

import numpy as np
import pandas as pd

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


from utils.molecules.openbabel_utils import obabel_convert
from utils.sys_utils import execute_system_command
from utils.io.io_utils import copy_file, delete_directory, delete_file, write_json

# from utils.molecules.pymol_utils import create_complex_with_pymol, calculate_RMSD_pymol

PLANTS_VERSION = 1.2

PLANTS_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "plants", f"PLANTS{PLANTS_VERSION}_64bit")
SPORES_LOCATION = os.path.join(PROJECT_ROOT, "bin", "docking", "plants", "SPORES_64bit")

PLANTS_NUM_POSES = 100

PLANTS_N_PROC = int(os.environ.get("PLANTS_N_PROC", default=1))
PLANTS_TIMEOUT = os.environ.get("PLANTS_TIMEOUT", default="1h")

def determine_binding_site_radius(
    # target_data: dict,
    size_x,
    size_y,
    size_z,
    max_radius: float = None
    ):
    binding_site_radius = max([size_x, size_y, size_z])
    if max_radius is not None:
        binding_site_radius = min([binding_site_radius, max_radius])
    return binding_site_radius

def prepare_for_plants(
    title: str,
    input_filename: str,
    output_filename: str = None,
    spores_mode: str = "complete",
    overwrite: bool = False,
    verbose: bool = True,
    ):
    # convert to mol2?
    # run SPORES?
    if verbose:
        print ("Preparing file", input_filename, "for PLANTS")

    stem, ext = os.path.splitext(input_filename)
    ext = ext.replace(".", "")
    if output_filename is None:
        output_filename = stem + "_PLANTS.mol2"
    
    if verbose:
        print ("Outputting to", output_filename)

    if not os.path.exists(output_filename) or input_filename == output_filename or overwrite:

        # convert to mol2 with obabel
        # output_filename will be None in the case of error
        input_filename = obabel_convert(
            input_format=ext,
            input_filename=input_filename,
            output_format="mol2",
            output_filename=output_filename,
            title=title, # rename molecule
            overwrite=overwrite,
            verbose=verbose,
        ) 

        if verbose:
            print ("Using SPORES")

        '''
        FROM SPORES
        inputfile is the structure file in pdb, sdf, mol or mol2 format,

        outputfile will be written by SPORES in mol2 format
        '''

        if input_filename == output_filename:
            output_filename = os.path.splitext(output_filename)[0]
            output_filename += "_SPORES.mol2"

        cmd = f'''\
        {SPORES_LOCATION} --mode {spores_mode} {input_filename} {output_filename}
        '''
        try:
            execute_system_command(cmd, timeout=PLANTS_TIMEOUT, verbose=verbose)
        except Exception as e:
            print ("SPORES EXCEPTION", e)
            return None

    # delete "bad" mol2 files
    for bad_mol2_filename in glob.iglob("*_bad.mol2"):
        delete_file(bad_mol2_filename, verbose=verbose)
        
    return output_filename

def write_plantsconfig(
    configfile_location: str,
    target_filename: str,
    ligand_filename: str,
    output_dir: str,
    center_x: float,
    center_y: float,
    center_z: float,
    binding_site_radius: float,
    scoring_function: str = "chemplp",
    write_multi_mol2: bool = 0,
    search_speed: int = 1,
    cluster_rmsd: float = 2.0,
    num_poses: int = PLANTS_NUM_POSES,
    verbose: bool = True,
    ):

    plantsconfig_text = dedent(f'''
    # scoring function and search settings
    scoring_function {scoring_function}
    search_speed speed{search_speed}

    # input
    protein_file {target_filename}
    ligand_file {ligand_filename}

    # output
    output_dir {output_dir}

    # write single mol2 files (e.g. for RMSD calculation)
    write_multi_mol2 {write_multi_mol2}

    # binding site definition
    bindingsite_center {center_x} {center_y} {center_z}
    bindingsite_radius {binding_site_radius}

    # cluster algorithm
    cluster_structures {num_poses}
    cluster_rmsd {cluster_rmsd}
    ''')
    # ensure directory exists
    configfile_directory = os.path.dirname(configfile_location)
    os.makedirs(configfile_directory, exist_ok=True)

    if verbose:
        print ("Writing", plantsconfig_text, "to", configfile_location)

    with open(configfile_location, "w") as f:
        f.write(plantsconfig_text)

    return configfile_location

def execute_plants(
    ligand_title: str,
    ligand_filename: str,
    target_title: str,
    target_filename: str,
    plants_output_dir: str, 
    configfile_location: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    scoring_function: str = "chemplp",
    write_multi_mol2: bool = 0,
    search_speed: int = 1,
    cluster_rmsd: float = 2.0, 
    mode: str = "screen",
    num_poses: int = None,
    prepare_ligand: bool = False,
    prepare_target: bool = False,
    verbose: bool = True,
    ):

    # default number of poses
    if num_poses is None:
        num_poses = PLANTS_NUM_POSES

    plants_data_filename = os.path.join(plants_output_dir, "plants_scores.json")
    
    if verbose:
        print ("Executing PLANTS for ligand", ligand_filename, "and target", target_filename)
        print ("Generating", num_poses, "binding pose(s)")
        print ("Outputting to directory", plants_output_dir)
        print ("Writing scores to", plants_data_filename)

    # check for existing poses

    if os.path.isdir(plants_output_dir):
        pose_mol2_filenames = [os.path.join(plants_output_dir, file) 
            for file in os.listdir(plants_output_dir) 
            if re.match(r".*conf_[0-9]+\.mol2$", file)] 
    else:
        pose_mol2_filenames = []

    num_existing_poses = len(pose_mol2_filenames)

    if num_existing_poses > 0 and os.path.exists(plants_data_filename):
        if verbose:
            print (num_existing_poses, "already exist and", plants_data_filename, "also exists, skipping docking")
        return pose_mol2_filenames, plants_data_filename

    ligand_filename = os.path.abspath(ligand_filename)
    target_filename = os.path.abspath(target_filename)
   
    if prepare_ligand or not ligand_filename.endswith(".mol2"):
        ligand_filename = prepare_for_plants(
            title=ligand_title,
            input_filename=ligand_filename,
            overwrite=True,
            verbose=verbose,
            )
        if ligand_filename is None:
            return [], plants_data_filename 

    if prepare_target or not target_filename.endswith(".mol2"):
        target_filename = prepare_for_plants(
            title=target_title,
            input_filename=target_filename,
            overwrite=True,
            verbose=verbose,
        )
        if target_filename is None:
            return [], plants_data_filename 

    delete_directory(plants_output_dir, verbose=verbose) # PLANTS DOES NOT LIKE THE DIRECTORY EXISTING

    current_dir = os.getcwd()

    plants_run_dir = os.path.dirname(plants_output_dir)
    os.makedirs(plants_run_dir, exist_ok=True)

    # change to basename and copy to run_dir
    ligand_basename = os.path.basename(ligand_filename)
    target_basename = os.path.basename(target_filename)
    plants_output_dir_basename = os.path.basename(plants_output_dir)

    # copy in ligand and target
    ligand_basename_in_plants_run_dir = os.path.join(plants_run_dir, ligand_basename)
    if not os.path.exists(ligand_basename_in_plants_run_dir):
        copy_file(ligand_filename, ligand_basename_in_plants_run_dir)
    target_basename_in_plants_run_dir = os.path.join(plants_run_dir, target_basename)
    if not os.path.exists(target_basename_in_plants_run_dir):
        copy_file(target_filename, target_basename_in_plants_run_dir)

    # determine binding_site_radius
    binding_site_radius = determine_binding_site_radius(
        size_x,
        size_y,
        size_z,
    )

    # if not os.path.exists(configfile_location):
    configfile_location = write_plantsconfig(
        configfile_location,
        ligand_filename=ligand_basename,
        target_filename=target_basename,
        output_dir=plants_output_dir_basename, 
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        binding_site_radius=binding_site_radius,
        scoring_function=scoring_function,
        write_multi_mol2=write_multi_mol2,
        search_speed=search_speed,
        cluster_rmsd=cluster_rmsd,
        num_poses=num_poses,
        verbose=verbose,
    )
    
    if verbose:
        print ("Using configfile located at", configfile_location)

    configfile_location = os.path.basename(configfile_location)

    if verbose:
        print ("Running PLANTS in", plants_run_dir)

    os.chdir(plants_run_dir)

    try:
        cmd = f'''\
        {PLANTS_LOCATION} --mode {mode} {configfile_location}
        '''
        execute_system_command(cmd, timeout=PLANTS_TIMEOUT, verbose=verbose)
    except Exception as e:
        print ("Exception running PLANTS", e)
    finally:
        os.chdir(current_dir)

    # write scores as a json file 
    ranking_filename = os.path.join(plants_output_dir, "ranking.csv")

    if os.path.exists(ranking_filename) and os.stat(ranking_filename).st_size > 0:
        plants_data_df = pd.read_csv(ranking_filename, index_col=0)
        plants_data = {
            i : {
                col.lower(): row[col]
                for col in row.index
                if col not in {"EVAL", "TIME",}
            }
            for i, (_, row) in enumerate(plants_data_df.iterrows(), start=1)
        }
        write_json(plants_data, plants_data_filename, verbose=verbose)

    pose_mol2_filenames = [
        os.path.join(plants_output_dir, file) 
        for file in os.listdir(plants_output_dir) 
        if re.match(r".*conf_[0-9]+\.mol2$", file)
    ] # return as list for multiprocessing

    return pose_mol2_filenames, plants_data_filename

if __name__ == "__main__":

    from concurrent.futures import ProcessPoolExecutor as Pool, as_completed
    # # from concurrent.futures import ThreadPoolExecutor as Pool
    
    # target_filenames = glob.glob("ai_docking_test_cases/test_case_1/psovina/4/prepared_targets/*/*/selected_chains_*.pdb")
    # target_filenames = target_filenames[:1000]

    # root_output_dir = "plants_targets"

    # with Pool(max_workers=24) as p:

    #     tasks = []

    #     for target_filename in target_filenames:
            
    #         if " " in target_filename:
    #             raise Exception(target_filename)
            
    #         target_basename = os.path.basename(target_filename)
    #         target_basename, _ = os.path.splitext(target_basename)


            
    #         output_filename = os.path.join(root_output_dir, target_basename + "_plants.mol2")
    #         if os.path.exists(output_filename, ):
    #             continue

    #         tasks.append(p.submit(
    #             prepare_for_plants,
    #             title=target_basename,
    #             input_filename=target_filename,
    #             output_filename=output_filename,
    #             verbose=True,
    #         ))

    #     for task in as_completed(tasks):
    #         task.result()


    root_output_dir = "plants"

    ligand_filename = "aspirin_plants.mol2"

    target_filenames = glob.glob(os.path.join("plants_targets", "*_SPORES.mol2"))

    with Pool(max_workers=22) as p:

        running_tasks = []

        for target_filename in target_filenames:
            target_basename = os.path.basename(target_filename)
            target_basename, _ = os.path.splitext(target_basename)
            
            target_output_dir = os.path.join(root_output_dir, target_basename)
            os.makedirs(target_output_dir, exist_ok=True)

            task = p.submit(
                execute_plants,
                ligand_title="aspirin",
                ligand_filename=ligand_filename,
                target_title=target_basename,
                target_filename=target_filename,
                plants_output_dir=target_output_dir,
                configfile_location=os.path.join(root_output_dir, target_basename + "_plantsconfig"),
                center_x=3.897,
                center_y=23.808,
                center_z=5.198,
                binding_site_radius=30,
                num_poses=1,
                verbose=True,
            )

            running_tasks.append((target_output_dir, task))

        for target_output_dir, running_task in running_tasks:
            plants_out = running_task.result()

            if plants_out is not None: # run error

                pose_output_dir = os.path.join(target_output_dir, "poses")
                os.makedirs(pose_output_dir, exist_ok=True)

                # convert and separate output_file
                pass