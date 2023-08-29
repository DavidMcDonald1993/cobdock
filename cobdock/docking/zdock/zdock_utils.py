


import os 
import shutil
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

from utils.molecules.pymol_utils import create_complex_with_pymol

from utils.molecules.openbabel_utils import obabel_convert
from utils.sys_utils import execute_system_command
from utils.io.io_utils import delete_file, move_file, copy_file, delete_directory

ZDOCK_DIRECTORY = os.path.join(PROJECT_ROOT, "bin", "docking", "zdock")
ZDOCK_LOCATION = os.path.join(ZDOCK_DIRECTORY, "zdock")

MARK_SUR_DIRECTORY = ZDOCK_DIRECTORY
MARK_SUR_LOCATION = os.path.join(MARK_SUR_DIRECTORY, "mark_sur")
UNICHARMM_LOCATION = os.path.join(MARK_SUR_DIRECTORY, "uniCHARMM")

CREATE_PL_DIRECTORY = ZDOCK_DIRECTORY # must also contain create_lig binary
CREATE_PL_LOCATION = os.path.join(CREATE_PL_DIRECTORY, "create.pl")
CREATE_LIG_LOCATION = os.path.join(CREATE_PL_DIRECTORY, "create_lig")

ZDOCK_NUM_POSES = 100

ZDOCK_N_PROC = int(os.environ.get("ZDOCK_N_PROC", default=1))
ZDOCK_TIMEOUT = os.environ.get("ZDOCK_TIMEOUT", default="1h")

def link_to_directory(
    file_to_link: str,
    output_directory: str,
    verbose: bool = True,
    ):
    os.makedirs(output_directory, exist_ok=True)
    file_to_link_basename = os.path.basename(file_to_link)
    output_filename = os.path.join(output_directory, file_to_link_basename)
    if os.path.exists(file_to_link) and not os.path.exists(output_filename):
        if verbose:
            print ("Linking file", file_to_link,  "to directory", output_directory)
        try:
            os.symlink(file_to_link, output_filename)
        except:
            pass 
    return output_filename


def link_unicharmm(
    output_directory: str,
    verbose: bool = True,
    ):
    return link_to_directory(
        file_to_link=UNICHARMM_LOCATION,
        output_directory=output_directory,
        verbose=verbose,
    )


def link_create_lig(
    output_directory: str,
    verbose: bool = True,
    ):
    return link_to_directory(
        file_to_link=CREATE_LIG_LOCATION,
        output_directory=output_directory,
        verbose=verbose,
    )

def prepare_for_zdock(
    input_filename: str,
    output_filename: str = None,
    overwrite: bool = False,
    verbose: bool = True,
    ):
    input_filename = os.path.abspath(input_filename)
  
    if output_filename is None:
        stem, _ = os.path.splitext(input_filename)
        output_filename = stem + "_mark_sur_out.pdb"

    if verbose:
        print("Preparing file", input_filename, "for ZDOCK")
        print ("Outputting to", output_filename)

    if not os.path.exists(output_filename) or input_filename == output_filename or overwrite:
        
        output_filename = os.path.abspath(output_filename)

        current_dir = os.getcwd()

        # input_directory = os.path.dirname(input_filename)
        output_directory = os.path.dirname(output_filename)

        input_basename = os.path.basename(input_filename)
        output_basename = os.path.basename(output_filename)

        # copy input file into output directory
        input_filename_in_output_directory = os.path.join(
            output_directory, input_basename,
        )
        if input_filename != input_filename_in_output_directory:
            copy_file(input_filename, input_filename_in_output_directory, verbose=verbose)

        # ensure uniCHARMM exists in output_directory
        link_unicharmm(output_directory, verbose=verbose)

        # run in output directory
        os.chdir(output_directory)

        cmd = f"{MARK_SUR_LOCATION} {input_basename} {output_basename}"
        try:
            execute_system_command(
                cmd, 
                timeout=ZDOCK_TIMEOUT, # sometimes mark_sur hangs for some reason
                verbose=verbose
                )
        except Exception as e:
            print ("PREPARE FOR ZDOCK exception", e)

        os.chdir(current_dir)

        # delete input_filename_in_output_directory if it is not the output file
        if os.path.exists(input_filename_in_output_directory) and input_filename_in_output_directory != output_filename:
            delete_file(input_filename_in_output_directory, verbose=verbose)

    if not os.path.exists(output_filename):
        return None # error

    return output_filename

def execute_zdock(
    ligand_filename: str, 
    target_filename: str,
    output_filename: str,
    fix: bool = True,
    num_poses: int = None,
    verbose: bool = True,
    ):

    if not ligand_filename.endswith(".pdb"):
        ligand_filename = prepare_for_zdock(ligand_filename, timeout=timeout, verbose=verbose)
        if ligand_filename is None:
            return None
    if not target_filename.endswith(".pdb"):
        target_filename = prepare_for_zdock(target_filename, timeout=timeout, verbose=verbose)
        if target_filename is None:
            return None

    if num_poses is None:
        num_poses = ZDOCK_NUM_POSES

    if verbose:
        print ("Executing ZDOCK for ligand", ligand_filename, "and target", target_filename)
        print ("Generating", num_poses, "binding pose(s)")
        print ("Outputting to", output_filename)

    if os.path.exists(output_filename):
        if verbose:
            print (output_filename, "already exists, skipping docking")
        return output_filename
    
    ligand_filename = os.path.abspath(ligand_filename)
    target_filename = os.path.abspath(target_filename)
    output_filename = os.path.abspath(output_filename)

    ligand_filename_directory = os.path.dirname(ligand_filename)
    current_dir = os.getcwd()

    assert ligand_filename_directory == os.path.dirname(target_filename), "LIGAND AND TARGET MUST BE IN THE SAME DIRECTORY"

    ligand_basename = os.path.basename(ligand_filename)
    target_basename = os.path.basename(target_filename)
    output_basename = os.path.basename(output_filename)

    os.chdir(ligand_filename_directory)

    args = f'''\
    {"-F " if fix else ""} \
    -N {num_poses} \
    -o {output_basename} \
    -L {ligand_basename} \
    -R {target_basename} \
    '''

    cmd = f"{ZDOCK_LOCATION} {args}"

    try:
        execute_system_command(
            cmd,
            allow_non_zero_return=True, # TODO: return to this
            timeout=ZDOCK_TIMEOUT,
            verbose=verbose,
        )
        if os.path.exists(output_basename) and os.path.abspath(output_basename) != output_filename: #' output filename is an abspath
            shutil.move(output_basename, output_filename)
    except Exception as e:
        print ("ZDOCK exception", e)

    finally:
        os.chdir(current_dir)

    return output_filename

def run_create_pl(
    input_filename: str,
    pose_output_dir: str,
    target_filename: str,
    complex_output_dir: str,
    num_complexes: int,
    verbose: bool = True,
    ):
  

    input_filename_dir = os.path.dirname(input_filename)
    input_basename = os.path.basename(input_filename)

    target_filename = os.path.abspath(target_filename)
    
    pose_output_dir = os.path.abspath(pose_output_dir)
    complex_output_dir = os.path.abspath(complex_output_dir)
    current_dir = os.getcwd()

    if verbose:
        print ("Running create.pl on file", input_filename)
        print ("outputting poses to directory", pose_output_dir)
        print ("outputting complexes to directory", complex_output_dir)


    # cannot already exist for some reason
    delete_directory(complex_output_dir, verbose=verbose,)
    os.makedirs(complex_output_dir, exist_ok=True)

    # create a link to create_lig
    create_lig_location = link_create_lig(input_filename_dir, verbose=verbose)

    os.chdir(input_filename_dir)

    try:
        cmd = f"{CREATE_PL_LOCATION} {input_basename}"
        execute_system_command(cmd, verbose=verbose)
    except Exception as e:
        print ("create.pl exception", e)


    # move poses
    pose_files = glob.glob("*.out.*")
    for pose_file in pose_files:
        pose_id = int(pose_file.split(".")[-1])
        new_pose_filename = os.path.join(
            pose_output_dir,
            f"pose_{pose_id}.pdb"
        )
        move_file(pose_file, new_pose_filename, verbose=verbose)

        # cleanup pose file with obabel
        new_pose_filename = obabel_convert(
            input_format="pdb",
            input_filename=new_pose_filename,
            output_format="pdb",
            output_filename=new_pose_filename,
            verbose=verbose,
        )

        # create complex
        if num_complexes is not None and pose_id <= num_complexes:
            complex_filename = os.path.join(
                complex_output_dir,
                f"complex_{pose_id}.pdb"
            )
            create_complex_with_pymol(
                input_pdb_files=[target_filename, new_pose_filename],
                output_pdb_filename=complex_filename,
                verbose=verbose,
            )


    # change back to original dir
    os.chdir(current_dir)

    # remove create_lig link
    delete_file(create_lig_location)

    # return pose filenames
    return glob.glob(os.path.join(pose_output_dir, "*.pdb"))


# def most_frequent_element(l):
#     return max(set(l), key=l.count)

# def get_most_frequent_ligand_symbol(ligand_structure_filename):

#     for atom_type in ("ATOM", "HETATM"):
#         all_symbols = subprocess.check_output(f"awk '/{atom_type}/ {{print $4}}' {ligand_structure_filename}", shell=True)
#         all_symbols = all_symbols.decode("utf-8")
#         all_symbols = all_symbols.split("\n")
#         if len(all_symbols) > 4:
#             break

#     return most_frequent_element(all_symbols)

# def process_single_pose_zdock(
#     pose_num,
#     zdock_score,
#     ligand_pose_filename,
#     compute_rmsd_with_submitted_ligand,
#     submitted_ligand_location,
#     ):

#     centre_of_mass = identify_centre_of_mass(ligand_pose_filename)
#     if centre_of_mass is None:
#         return {}
#     center_x, center_y, center_z = centre_of_mass
    
#     size_x, size_y, size_z = get_bounding_box_size(ligand_pose_filename)

#     pose_data = {
#         "center_x": center_x, 
#         "center_y": center_y, 
#         "center_z": center_z, 
#         "size_x": size_x, 
#         "size_y": size_y, 
#         "size_z": size_z, 
#         "score": zdock_score,
#     }
#     if compute_rmsd_with_submitted_ligand:
#         pose_data["rmsd"] = calculate_RMSD_pymol(
#             submitted_ligand_location,
#             ligand_pose_filename,
#         )
        
#     return {
#         pose_num : pose_data
#     }

# def collate_zdock_data(
#     ligands_to_targets,
#     output_dir,
#     output_filename,
#     compute_rmsd_with_submitted_ligand=True,
#     num_complexes=1,
#     ):

#     if os.path.exists(output_filename):
#         print(output_filename, "already exists -- loading it")
#         zdock_data = load_json(output_filename)

#     else:
        
#         zdock_data = {}

#         for ligand_name in ligands_to_targets:

#             if ligand_name not in zdock_data:
#                 zdock_data[ligand_name] = {}

#             submitted_ligand_location = ligands_to_targets[ligand_name]["pdb_filename"]

#             ligand_zdock_out_directory = os.path.join(
#                 output_dir, 
#                 ZDOCK_OUT_DIRECTORY, 
#                 ligand_name,
#                 )

#             ligand_accessions = ligands_to_targets[ligand_name]["prepared_targets"]

#             for accession in ligand_accessions:

#                 if accession not in zdock_data[ligand_name]:
#                     zdock_data[ligand_name][accession] = {}

#                 ligand_pdb_targets = ligand_accessions[accession]
            
#                 for ligand_pdb_target in ligand_pdb_targets:

#                     if ligand_pdb_target not in zdock_data[ligand_name][accession]:
#                         zdock_data[ligand_name][accession][ligand_pdb_target] = {}

#                     target_prepared_filename = ligand_accessions[accession][ligand_pdb_target]["prepared_filename"]

#                     target_out_file = os.path.join(
#                         ligand_zdock_out_directory, 
#                         f"{ligand_pdb_target}.out")

#                     # check that ZDOCK ran for this target
#                     if os.path.exists(target_out_file):
    
#                         print (f"Processing target_out_file {target_out_file}")

#                         ligand_target_zdock_out_directory = os.path.join(
#                             ligand_zdock_out_directory, 
#                             ligand_pdb_target)
#                         os.makedirs(ligand_target_zdock_out_directory, exist_ok=True)

#                         # create pose and complex dir
#                         pose_output_dir = os.path.join(
#                             ligand_target_zdock_out_directory,
#                             "poses",)
#                         os.makedirs(pose_output_dir, exist_ok=True)

#                         complex_output_dir = os.path.join(
#                             ligand_target_zdock_out_directory,
#                             "complexes")
#                         os.makedirs(complex_output_dir, exist_ok=True)
                        
#                         # read zdock out file and get score  
#                         print ("Reading ZDOCK energies from", target_out_file)
#                         target_zdock_output_df = pd.read_csv(
#                             target_out_file, 
#                             index_col=None, 
#                             skiprows=4, # based on ZDock output header  
#                             sep="\t", 
#                             header=None,
#                             names=["rx", "ry", "rz", "tx", "ty", "tz", "energy"])

#                         zdock_energies = target_zdock_output_df["energy"] 

#                         print ("Generating poses and complexes from", target_out_file )

#                         pose_pdb_filenames = run_create_pl(
#                             input_filename=target_out_file, 
#                             pose_output_dir=pose_output_dir,
#                             target_filename=target_prepared_filename,
#                             complex_output_dir=complex_output_dir,
#                             num_complexes=num_complexes,
#                             )


#                         # for pose_num in range(1, ZDOCK_NUM_POSES+1):
#                         for pose_pdb_filename in pose_pdb_filenames:
                            
#                             stem, ext = os.path.splitext(pose_pdb_filename)
#                             pose_num = int(stem.split("_")[-1])

#                             zdock_score = zdock_energies.iloc[pose_num - 1]
                        
#                             zdock_data[ligand_name][accession][ligand_pdb_target].update(
#                                 process_single_pose_zdock(
#                                     pose_num=pose_num,
#                                     zdock_score=zdock_score,
#                                     ligand_pose_filename=pose_pdb_filename,
#                                     compute_rmsd_with_submitted_ligand=compute_rmsd_with_submitted_ligand,
#                                     submitted_ligand_location=submitted_ligand_location,
#                                 )
#                             )
                    
#                     if len(zdock_data[ligand_name][accession][ligand_pdb_target]) == 0:
#                         zdock_data[ligand_name][accession][ligand_pdb_target][None] = {
#                             "center_x": None, 
#                             "center_y": None, 
#                             "center_z": None, 
#                             "size_x": None, 
#                             "size_y": None, 
#                             "size_z": None, 
#                             "rmsd": None,
#                             "score": None,
#                         }

#         print ("Writing ZDOCK data to", output_filename)
#         write_json(zdock_data, output_filename)

#     return zdock_data

if __name__ == "__main__":
    # ligands = ["aspirin"]
    # targets = ["3EWT"]

    # zdock_data = collate_zdock_data(
    #     ligands=ligands, 
    #     targets=targets,
    #     docking_directory=docking_directory,)

    # from utils.io.io_utils import write_json


    # write_json(zdock_data, os.path.join(PROJECT_ROOT, "ZDOCK_OUT.json")) 

    # ligand_filename = f"{DATA_ROOT_DIR}/training/negative_prepared/chunk_0/docking/ZDOCK_out/4KPR_ZON/4KPR_ZON_MARK_SUR_OUT.pdb"
    # target_filename = f"{DATA_ROOT_DIR}/training/negative_prepared/chunk_0/docking/ZDOCK_out/4KPR_ZON/4KPR_prepared_MARK_SUR_OUT.pdb"
    # output_filename = "test_ZDOCK_OUT.out"

    # prepare_for_zdock(
    #     input_filename=f"{DATA_ROOT_DIR}/training/negative_prepared/chunk_0/docking/ZDOCK_out/4KPR_ZON/4KPR_ZON.pdb",
    #     # output_filename="test_MARK_SUR.pdb",

    # )

    # output_filename = execute_zdock(
    #     ligand_filename=os.path.abspath(ligand_filename),
    #     target_filename=os.path.abspath(target_filename),
    #     output_filename=os.path.abspath(output_filename),
    # )

    # run_create_pl(
    #     input_filename=f"{DATA_ROOT_DIR}/training/negative_prepared/chunk_0/docking/ZDOCK_out/4KPR_ZON/4KPR.out",
    #     output_dir="CREATE_PL_out",
    # )

    # collate_zdock_data(
    #     ligands={"aspirin"},
    #     targets={"3FFD"},
    # )

    # prepare_for_zdock(
    #     # input_filename="1a0t.pdb",
    #     input_filename="test_compounds/aspirin.pdb",
    #     # input_filename="input.pdb",
    #     # output_filename="mark_sur_test/DELTEME.pdb",
    #     )

    # execute_zdock(
    #     ligand_filename=""
    # )

    print (link_unicharmm("my_test_dir"))