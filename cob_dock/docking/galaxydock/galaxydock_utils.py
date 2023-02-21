

import os


import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    os.pardir,
    os.pardir,
    ))

if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, PROJECT_ROOT)

from utils.molecules.openbabel_utils import obabel_convert
from utils.sys_utils import execute_system_command
from utils.molecules.pdb_utils import remove_all_hetero_residues_using_biopython
from utils.io.io_utils import copy_file
from utils.molecules.pymol_utils import relabel_residue

GALAXYDOCK_VERSION = 3
GALAXYDOCK_HOME = os.path.join(PROJECT_ROOT, "bin", "docking", f"GalaxyDock{GALAXYDOCK_VERSION}")

GALAXYDOCK_OUT_DIRECTORY = f"GALAXYDOCK{GALAXYDOCK_VERSION}_out"

CALCULATE_RMSD_SCRIPT_LOCATION = os.path.join(GALAXYDOCK_HOME, "script", "calc_RMSD.py")
RUN_GALAXYDOCK_LOCATION = os.path.join(GALAXYDOCK_HOME, "script", "run_GalaxyDock3.py")

# GALAXYDOCK_MAX_POSES = 25
GALAXYDOCK_MAX_POSES = 100

GALAXYDOCK_N_PROC = int(os.environ.get("GALAXYDOCK_N_PROC", default=1))

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
    remove_residues: str = False,
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

    if output_filename is not None and remove_residues:
        output_filename = remove_all_hetero_residues_using_biopython(
            pdb_id=None,
            pdb_filename=output_filename,
            output_filename=output_filename,
        )

    output_filename = relabel_residue(
        input_filename=output_filename,
        original_residue_labels=["HIE", "HID", "HIP", "HIZ"],
        new_residue_label="HIS",
        output_filename=output_filename,
        verbose=verbose,
    )

    output_filename = relabel_residue(
        input_filename=output_filename,
        original_residue_labels=["CYM", ],
        new_residue_label="CYS",
        output_filename=output_filename,
        verbose=verbose,
    )

    output_filename = relabel_residue(
        input_filename=output_filename,
        original_residue_labels=["GLH", "GLZ"],
        new_residue_label="GLU",
        output_filename=output_filename,
        verbose=verbose,
    )

    # DID, DIC -> ?

    # TYS -> TYR?
    output_filename = relabel_residue(
        input_filename=output_filename,
        original_residue_labels=["TYS", ],
        new_residue_label="TYR",
        output_filename=output_filename,
        verbose=verbose,
    )

    # MEU -> MET? 
    output_filename = relabel_residue(
        input_filename=output_filename,
        original_residue_labels=["MEU", ],
        new_residue_label="MET",
        output_filename=output_filename,
        verbose=verbose,
    )

    # LLP

    return output_filename

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
    # timeout: str = "1h", # kill after 1 hour of runtime
    timeout: str = "10m", # kill after 10 mins of runtime
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

    if verbose:
        print ("Scores will be located at", galaxydock_output_filename)
        print ("Output poses will be located at", pose_mol2_file)

    if os.path.exists(galaxydock_output_filename) and os.path.exists(pose_mol2_file):
        if verbose:
            print (galaxydock_output_filename, "and", pose_mol2_file, "already exist, skipping docking")
        
        return galaxydock_output_filename, pose_mol2_file

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

    args = f'''\
    -d {GALAXYDOCK_HOME} \
    -p {target_filename} \
    -l {ligand_filename} \
    -x {center_x} \
    -y {center_y} \
    -z {center_z} \
    -size_x {size_x} \
    -size_y {size_y} \
    -size_z {size_z} \
    '''
    if use_multiprocessing: # only for new GalaxyDock
        args += f" --n_proc {n_proc}"

    try:
        execute_system_command(f"python {RUN_GALAXYDOCK_LOCATION} {args}", timeout=timeout, verbose=verbose)
    except Exception as e:
        print ("GalaxyDock exception", e)
    finally:
        # change back to current_dir
        os.chdir(current_dir)

    return galaxydock_output_filename, pose_mol2_file

# def calculate_RMSD_galaxydock(
#     reference_ligand_filename,
#     model_ligand_filename,
#     output_directory,
#     delete_hydrogen=True,
#     add_hydrogen=False,
#     verbose: bool = False,
#     ):

#     # if not reference_ligand_filename.endswith(".mol2"):
#     reference_ligand_stem, ext = os.path.splitext(reference_ligand_filename)
#     ext = ext.replace(".", "")
#     reference_ligand_filename = obabel_convert(
#         input_format=ext,
#         input_filename=reference_ligand_filename,
#         output_format="mol2",
#         output_filename=f"{reference_ligand_stem}_galaxydock.mol2",
#         delete_hydrogen=delete_hydrogen,
#         add_hydrogen=add_hydrogen,
#         canonical=True,
#         title="reference",
#     )
#     # if not model_ligand_filename.endswith(".mol2"):
#     model_ligand_stem, ext = os.path.splitext(model_ligand_filename)
#     ext = ext.replace(".", "")
#     model_ligand_filename = obabel_convert(
#         input_format=ext,
#         input_filename=model_ligand_filename,
#         output_format="mol2",
#         output_filename=f"{model_ligand_stem}_galaxydock.mol2",
#         delete_hydrogen=delete_hydrogen,
#         add_hydrogen=add_hydrogen,
#         canonical=True,
#         title="model",
#     )
#     # assert model_ligand_filename.endswith(".mol2")
#     assert os.path.exists(reference_ligand_filename), f"{reference_ligand_filename} DOES NOT EXIST"
#     assert os.path.exists(model_ligand_filename), f"{model_ligand_filename} DOES NOT EXIST"


#     print ("CALCULATING RMSD FOR LIGAND FILES", reference_ligand_filename, model_ligand_filename)
#     print ("OUTPUTTING TO DIRECTORY", output_directory)

#     current_directory = os.getcwd()

    
#     reference_ligand_filename = os.path.abspath(reference_ligand_filename)
#     model_ligand_filename = os.path.abspath(model_ligand_filename)

#     print ("MAKING DIRECTORY", output_directory)
#     os.makedirs(output_directory, exist_ok=True)

#     reference_ligand_basename = os.path.basename(reference_ligand_filename)
#     model_ligand_basename = os.path.basename(model_ligand_filename)
    
#     reference_ligand_filename_in_dir = os.path.join(output_directory, reference_ligand_basename)
#     model_ligand_filename_in_dir = os.path.join(output_directory, model_ligand_basename)

#     copy_file(reference_ligand_filename, reference_ligand_filename_in_dir)
#     print (reference_ligand_filename, reference_ligand_filename_in_dir)
#     shutil.copy(model_ligand_filename, model_ligand_filename_in_dir)
#     print (model_ligand_filename, model_ligand_filename_in_dir)

#     os.chdir(output_directory)


#     cmd = f'''
#     python {CALCULATE_RMSD_SCRIPT_LOCATION} \
#         {GALAXYDOCK_HOME} \
#         {reference_ligand_basename} \
#         {model_ligand_basename} \
#     '''

#     execute_system_command(cmd, )

#     os.chdir(current_directory)

#     log_filename = os.path.join(output_directory, "rmsd.log")
#     if os.path.exists(log_filename):
#         rmsd = subprocess.check_output(f"awk '/RMSD/ {{print $5}}' {log_filename}", shell=True)
#         # print ("REMOVING DIRECTORY", output_directory)
#         # shutil.rmtree(output_directory)

#         try:
#             rmsd = float(rmsd)
#             return rmsd
#         except:
#             print ("RMSD CONVERSION TO FLOAT FAIL", rmsd)
#             pass 

#     return None

# def collate_galaxydock_data(
#     ligands_to_targets,
#     output_dir,
#     output_filename,
#     precision=3,
#     compute_rmsd_with_submitted_ligand=True,
#     num_complexes=1,
#     ):

#     if os.path.exists(output_filename):
#         print(output_filename, "EXISTS -- LOADING IT")
#         galaxydock_data = load_json(output_filename)

#     else:
#         galaxydock_data = {}

#         for ligand_name in ligands_to_targets:

#             if ligand_name not in galaxydock_data:
#                 galaxydock_data[ligand_name] = {}

#             submitted_ligand_location = ligands_to_targets[ligand_name]["pdb_filename"]

#             # copy ligand from docking_dir into galaxy dir
#             ligand_galaxydock_out_dir = os.path.join(
#                 output_dir,
#                 GALAXYDOCK_OUT_DIRECTORY, 
#                 ligand_name)


#             ligand_accessions = ligands_to_targets[ligand_name]["prepared_targets"]

#             for accession in ligand_accessions:

#                 if accession not in galaxydock_data[ligand_name]:
#                     galaxydock_data[ligand_name][accession] = {}

#                 ligand_pdb_targets = ligand_accessions[accession]

#                 for ligand_pdb_target in ligand_pdb_targets:

#                     # add PDB target to return dictionary
#                     if ligand_pdb_target not in galaxydock_data[ligand_name][accession]:
#                         galaxydock_data[ligand_name][accession][ligand_pdb_target] = {}

#                     prepared_target_filename = ligand_accessions[accession][ligand_pdb_target]["prepared_filename"]

#                     ligand_target_galaxydock_out_directory = os.path.join(
#                         ligand_galaxydock_out_dir, 
#                         ligand_pdb_target)
                   
#                     galaxydock_info_filename = os.path.join(
#                         ligand_target_galaxydock_out_directory,
#                         f"GD{GALAXYDOCK_VERSION}_fb.E.info")

#                     galaxydock_pose_output_filename = os.path.join(
#                         ligand_target_galaxydock_out_directory, 
#                         f"GD{GALAXYDOCK_VERSION}_fb.mol2")

#                     if os.path.exists(galaxydock_info_filename) and os.path.exists(galaxydock_pose_output_filename): 

#                         print ("Reading GalaxyDock scores from", galaxydock_info_filename)
#                         galaxydock_output_df = pd.read_fwf(
#                             galaxydock_info_filename, 
#                             comment="!", 
#                             sep="\t", 
#                             index_col=0)
#                         # necessary
#                         galaxydock_output_df = galaxydock_output_df.fillna(0)

#                         # make pose and complex directory
#                         pose_output_dir = os.path.join(
#                             ligand_target_galaxydock_out_directory, 
#                             "poses")
#                         os.makedirs(pose_output_dir, exist_ok=True)

#                         complex_output_dir = os.path.join(
#                             ligand_target_galaxydock_out_directory,
#                             "complexes")
#                         os.makedirs(complex_output_dir, exist_ok=True)

#                         # convert to pdb and split
#                         obabel_return = obabel_convert(
#                             input_format="mol2",
#                             input_filename=galaxydock_pose_output_filename,
#                             output_format="pdb",
#                             output_filename="pose_",
#                             output_dir=pose_output_dir, 
#                             multiple=True,
#                         )

#                         # iterate over pose PDB files
#                         for pose_pdb_filename in glob.iglob(os.path.join(pose_output_dir, "pose_*.pdb")):

#                             # get rank
#                             stem, ext = os.path.splitext(pose_pdb_filename)
#                             rank = int(stem.split("_")[-1])

#                             if num_complexes is not None and rank <= num_complexes:

#                                 # make complex with current pose 
#                                 complex_filename = os.path.join(
#                                     complex_output_dir,
#                                     f"complex_{rank}.pdb")
#                                 create_complex_with_pymol(
#                                     input_pdb_files=[prepared_target_filename, pose_pdb_filename],
#                                     output_pdb_filename=complex_filename,
#                                 )

#                             # compute bounding box for current pose
#                             center_of_mass = identify_centre_of_mass(pose_pdb_filename)
#                             if center_of_mass is None:
#                                 continue
#                             center_x, center_y, center_z = center_of_mass
#                             size_x, size_y, size_z = get_bounding_box_size(pose_pdb_filename)


#                             galaxydock_score = galaxydock_output_df["Energy"].loc[rank]
#                             if isinstance(galaxydock_score, str):
#                                 galaxydock_score = 0
#                             galaxydock_rmsd = galaxydock_output_df["l_RMSD"].loc[rank]
#                             if isinstance(galaxydock_rmsd, str):
#                                 galaxydock_rmsd = 0
#                             galaxydock_autodock = galaxydock_output_df["ATDK_E"].loc[rank]
#                             if isinstance(galaxydock_autodock, str):
#                                 galaxydock_autodock = 0
#                             galaxydock_internal_energy = galaxydock_output_df["INT_E"].loc[rank]
#                             if isinstance(galaxydock_autodock, str):
#                                 galaxydock_internal_energy = 0
#                             galaxydock_drug_score = galaxydock_output_df["DS_E"].loc[rank]
#                             if isinstance(galaxydock_drug_score, str):
#                                 galaxydock_drug_score = 0

#                             if isinstance(precision, int):
#                                 galaxydock_score = round(galaxydock_score, precision)
#                                 galaxydock_rmsd = round(galaxydock_rmsd, precision)
#                                 galaxydock_autodock = round(galaxydock_autodock, precision)
#                                 galaxydock_internal_energy = round(galaxydock_internal_energy, precision)
#                                 galaxydock_drug_score = round(galaxydock_drug_score, precision)

#                             pose_data = {
#                                 "center_x": center_x, 
#                                 "center_y": center_y, 
#                                 "center_z": center_z, 
#                                 "size_x": size_x, 
#                                 "size_y": size_y, 
#                                 "size_z": size_z, 
#                                 "rmsd": galaxydock_rmsd,
#                                 "score": galaxydock_score,
#                                 "autodock": galaxydock_autodock,
#                                 "drug_score": galaxydock_drug_score,
#                                 "internal_energy": galaxydock_internal_energy, 
#                             }

#                             if compute_rmsd_with_submitted_ligand:

#                                 # calculate RMSD with submitted ligand
#                                 pose_data["rmsd_submitted"] = calculate_RMSD_pymol(
#                                     reference_filename=submitted_ligand_location,
#                                     model_filename=pose_pdb_filename,
#                                 )

#                             galaxydock_data[ligand_name][accession][ligand_pdb_target][rank] = pose_data


#                     # no poses
#                     if len(galaxydock_data[ligand_name][accession][ligand_pdb_target]) == 0:
#                         missing_pose_data = {
#                             "center_x": None, 
#                             "center_y": None, 
#                             "center_z": None, 
#                             "size_x": None, 
#                             "size_y": None, 
#                             "size_z": None, 
#                             "rmsd": None,
#                             "score": None,
#                             "autodock": None,
#                             "drug_score": None,
#                             "internal_energy": None, 
#                         }
#                         if compute_rmsd_with_submitted_ligand:
#                             missing_pose_data["rmsd_submitted"] = None
#                         galaxydock_data[ligand_name][accession][ligand_pdb_target][None] = missing_pose_data

#             # delete ligand pose files
#             for ext in ("pdb", "mol2"):
#                 for filename in glob.iglob(os.path.join(ligand_galaxydock_out_dir, f"pose_*.{ext}")):
#                     print("REMOVING FILE", filename)
#                     os.remove(filename)

#         print ("WRITING GALAXYDOCK DATA TO", output_filename)
#         write_json(galaxydock_data, output_filename)

#     return galaxydock_data

if __name__ == "__main__":

    pass