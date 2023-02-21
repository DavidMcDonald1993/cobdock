if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        )))

import os

import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.sys_utils import execute_system_command
from utils.io.io_utils import sanitise_filename, read_smiles, delete_file, write_json
from utils.molecules.rdkit_utils import smiles_to_SDF_3D_rdkit_single_molecule

def obabel_convert(
    input_format: str,
    input_filename: str, 
    output_format: str,
    output_filename: str = None,
    multiple: bool = False,
    output_dir: str = None,
    gen_3d: bool = False,
    add_hydrogen: bool = False,
    delete_hydrogen: bool = False,
    pH: float = None,
    title: str = None,
    append: str = None,
    overwrite: bool = False,
    main_job_id: bool = None,
    verbose: bool = False,
    ):
    """Wrapper for obabel command.
    Used to convert molecule formats.

    Parameters
    ----------
    input_format : str
        Input format type
    input_filename : str
        Filename of input file
    output_format : str
        Desired output format type
    output_filename : str, optional
        Desired output filename, by default None
    multiple : bool, optional
        Flag to generate multiple output files, by default False
    output_dir : str, optional
        Directory to output multiple files to, by default None
    add_hydrogen : bool, optional
        Flag to make all hydrogen explicit, by default False
    delete_hydrogen : bool, optional
        Flag to make all hydrogen implicit, by default False
    title : str, optional
        Optional title of the molecule, by default None
    overwrite : bool, optional
        Flag to overwrite the output file if it already exists, by default False
    main_job_id : bool, optional
        Job ID to associate with, by default None
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    str
        The output filename or None if obabel failed
    """

    # can handle multiple inputs 
    if isinstance(input_filename, str):
        input_filename = [input_filename]
    
    if input_format is None:
        _, input_format = os.path.splitext(input_filename[0])
        input_format = input_format.replace(".", "")


    if output_filename is None: # output to same directory as input, if NOT multiple 
        stem, ext = os.path.splitext(input_filename[0])
        output_filename = stem + "." + output_format

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, output_filename)

    args = []
    if input_format is not None:
        args.append(f"-i {input_format}")
    if input_filename is not None:
        # args.append(f"{input_filename}")
        args.append(" ".join(input_filename))
    if output_format is not None:
        args.append(f"-o {output_format}")
        if not output_filename.endswith(output_format):
            output_filename += f".{output_format}"
    if output_filename is not None:
        args.append(f"-O {output_filename}")
    if multiple:
        args.append("-m")

    # gen 3D
    if gen_3d:
        args.append("--gen3d")

    # handle hydrogens
    if pH is not None:
        args.append(f"-p {pH}")
    elif add_hydrogen:
        args.append("-h")
    elif delete_hydrogen:
        args.append("-d")
   
    if title is not None:
        args.append(f"--title {title}")
    if append is not None:
        args.append(f"--append {append}")

    args = " ".join(args)

    cmd = f"obabel {args}"
    if not os.path.exists(output_filename) or input_filename[0] == output_filename or overwrite:
        if verbose:
            print ("Using OpenBabel to convert", len(input_filename), "file(s), first is:", input_filename[0], 
                "from format", input_format, "to format", output_format)
            print ("Writing to directory", output_dir)
            print ("Writing to file", output_filename)
        try:
            execute_system_command(cmd, main_job_id=main_job_id, verbose=verbose)
        except Exception as e:
            print ("OPENBABEL exception", e)
            return None
    return output_filename

def convert_single_smiles_to_3D(
    smiles_record,
    output_dir,
    output_format,
    desired_output_filename: str,
    add_hydrogen: bool = True,
    delete_hydrogen: bool = False,
    pH: float = None,
    smiles_key: str = "smiles",
    molecule_identifier_key: str = "molecule_id",
    overwrite: bool = False,
    verbose: bool = False,
    ):
    if isinstance(smiles_record, tuple):
        smi, molecule_identifier = smiles_record
    elif isinstance(smiles_record, dict):
        smi = smiles_record[smiles_key]
        molecule_identifier = smiles_record[molecule_identifier_key]
    else:
        raise NotImplementedError
    # sanitise molname for filename
    molecule_identifier_sanitised = sanitise_filename(molecule_identifier)
    
    if desired_output_filename is None:
        mol_output_filename = os.path.join(
            output_dir, 
            f"{molecule_identifier_sanitised}.{output_format}")
    else:
        mol_output_filename = desired_output_filename

    # do nothing if mol_output_filename already exists
    # alway run if overwrite == True
    if not os.path.exists(mol_output_filename) or overwrite:

        sdf_filename = os.path.join(
            output_dir, 
            f"{molecule_identifier_sanitised}.sdf") # generate name based on molecule id in file
    
        # first convert to SDF format using RDKit
        sdf_filename = smiles_to_SDF_3D_rdkit_single_molecule(
            smi=smi, 
            molecule_identifier=molecule_identifier_sanitised, 
            sdf_filename=sdf_filename, 
            add_hydrogen=add_hydrogen,
            overwrite=overwrite,
            verbose=verbose,
            )
        
        # use obabel to convert from SDF to other 3D formats
        if output_format != "sdf":
            
            mol_output_filename = obabel_convert(
                input_format="sdf", 
                input_filename=sdf_filename, 
                output_format=output_format, 
                output_filename=mol_output_filename, 
                add_hydrogen=add_hydrogen,
                delete_hydrogen=delete_hydrogen,
                pH=pH,
                title=molecule_identifier_sanitised,
                overwrite=overwrite,
                verbose=verbose,
                )
            delete_file(sdf_filename, verbose=verbose)
        else:
            # output format is SDF
            mol_output_filename = sdf_filename 

    
    # relabel residue

    return {molecule_identifier: mol_output_filename}

def smiles_to_3D(
    supplied_molecules, 
    output_dir: str, 
    output_format: str = "pdb",
    desired_output_filename: str = None, # lazy way to enforce correct filename --> only use on SMILES file containing single molecule
    add_hydrogen: bool = True,
    delete_hydrogen: bool = False,
    pH: float = None,
    assume_clean_input: bool = False,
    delimiter: str = "\t",
    smiles_key: str = "smiles",
    molecule_identifier_key: str = "molecule_id",
    overwrite: bool = False,
    n_proc: int = 10,
    verbose: bool = False,
    ):
    """Convert to two dimensional molecules to 3D using RDKit and OpenBabel

    Parameters
    ----------
    supplied_molecules : list
        SMILES strings of molecules to convert, alternatively a string filename of SMILES to load
    output_dir : str
        Direcotry to output to
    output_format : str, optional
        Desired 3D output format, one of "pdb", "mol2", "sdf", by default "pdb"
    desired_output_filename : str, optional
        Desired output filename, by default None
    delete_hydrogen : bool, optional
        Flag to remove all hydrogens, by default False
    assume_clean_input : bool, optional
        Flag for clean input file broken by delimiter, by default False
    delimiter : str, optional
        Delimiter in SMILES file, by default "\t"
    smiles_key : str, optional
        Key of SMILES in list of dicts, by default "smiles"
    molecule_identifier_key : str, optional
        Key of molecule ID in list of dicts, by default "molecule_id"
    overwrite : bool, optional
        Flag to overwrite an existing file if it exists
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    dict
        Mapping from molecule to filename

    Raises
    ------
    ValueError
        Thrown if an output filename is provided for multiple molecules 
    NotImplementedError
        Thrown if list of molecules does not contain tuples or dicts
    """
    '''
    input is (smi, mol_name) pairs or filename of SMILES file
    '''

    if verbose:
        print ("Converting SMILES to 3D structure format", output_format)
    assert output_format in {"mol2", "sdf", "pdb"}
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(supplied_molecules, str):
        # smiles is a filename
        supplied_molecules = read_smiles(
            supplied_molecules,
            return_list=True,
            assume_clean_input=assume_clean_input,
            delimiter=delimiter,
            smiles_key=smiles_key,
            molecule_identifier_key=molecule_identifier_key,
        )

    num_supplied_molecules = len(supplied_molecules)
    if num_supplied_molecules == 0:
        if verbose:
            print ("No molecules supplied")
        return {}

    if verbose:
        print ("Converting", num_supplied_molecules, "molecule(s) to", output_format, "format")
    
    if desired_output_filename is not None:

        desired_mol_name, _ = os.path.splitext(os.path.basename(desired_output_filename))
        if verbose:
            print ("Output filename has been supplied:", desired_output_filename)
            print ("Desired molecule name is", desired_mol_name)
        # assert len(smiles) == 1, "YOU MUST NOT SPECIFY AN OUTPUT FILENAME FOR MORE THAN ONE MOLECULE!"
        # let's try to be nice -- allow multiple molecules in file so long as one matches desired_mol_name
        mols_in_smiles = {
            # mol_name: (smi, mol_name)
            record[molecule_identifier_key]: record 
                for record in supplied_molecules
        }

        if desired_mol_name in mols_in_smiles:
            supplied_molecules = [
                mols_in_smiles[desired_mol_name]
            ] # remove any other molecules
        # elif len(mols_in_smiles) == 1:
        else: # take first molecule only?
            name = list(mols_in_smiles)[0]
            smi = mols_in_smiles[name][smiles_key]
            supplied_molecules = [{
                smiles_key: smi,
                molecule_identifier_key: desired_mol_name,  
            }] # rename
        # else:
        #     raise ValueError("YOU MUST NOT SPECIFY AN OUTPUT FILENAME FOR MORE THAN ONE MOLECULE!")

    
    with ThreadPoolExecutor(max_workers=n_proc) as p:

        running_tasks = []

        for smiles_record in supplied_molecules:

            task = p.submit(
                convert_single_smiles_to_3D,
                smiles_record=smiles_record,
                output_dir=output_dir,
                output_format=output_format,
                desired_output_filename=desired_output_filename,
                add_hydrogen=add_hydrogen,
                delete_hydrogen=delete_hydrogen,
                pH=pH,
                smiles_key=smiles_key,
                molecule_identifier_key=molecule_identifier_key,
                overwrite=overwrite,
                verbose=verbose
            )

            running_tasks.append(task)

    all_filenames = {}

    for task in as_completed(running_tasks):

        task_result = task.result()

        all_filenames.update(task_result)

    if verbose:
        print ("Conversion from 2D to 3D complete")
    return all_filenames

def generate_conformer_with_obabel(
    input_filename: str,
    output_filename: str = None,
    n_steps: int = 200,
    n_geometry_steps: int = 100,
    overwrite: bool = False,
    verbose: bool = False,
    ):
    """Use OpenBabel to generate a conformer

    Parameters
    ----------
    input_filename : str
        Filename of molecule
    output_filename : str, optional
        Desired output filename, will be generated based on input_filename if not given, by default None
    n_steps : int, optional
        Number of optimisation steps, by default 200
    n_geometry_steps : int, optional
        Number of geometry steps, by default 100
    overwrite : bool, optional
        Flag to overwrite output filename if it exists, by default False

    Returns
    -------
    str
        Output filename
    """

    if output_filename is None:
        stem, ext = os.path.splitext(input_filename)
        output_filename = stem + "_conf" + ext 

    if not os.path.exists(output_filename) or input_filename == output_filename or overwrite:
        cmd = f"obconformer {n_steps} {n_geometry_steps} {input_filename} > {output_filename}"
        execute_system_command(cmd, verbose=verbose)

    if os.stat(output_filename).st_size == 0:
        return None 

    return output_filename


def generate_rotamer_with_obabel(
    input_filename: str,
    output_filename: str = None,
    overwrite: bool = False,
    verbose: bool = False,
    ):
    """Use OpenBabel to generate a rotamer from an input molecule

    Parameters
    ----------
    input_filename : str
        Filename of molecule
    output_filename : str, optional
        Desired output filename, will be generated based on input_filename if not given, by default None
    overwrite : bool, optional
        Flag to overwrite output filename if it exists, by default False

    Returns
    -------
    str
        Output filename
    """

    if output_filename is None:
        stem, ext = os.path.splitext(input_filename)
        output_filename = stem + "_rot" + ext 

    if not os.path.exists(output_filename) or input_filename == output_filename or overwrite:
        cmd = f"obrotamer {input_filename} > {output_filename}"
        execute_system_command(cmd, verbose=verbose)

    if os.stat(output_filename).st_size == 0:
        return None

    return output_filename

# def pqr_to_pdb(
#     input_filename: str,
#     output_filename: str = None,
#     main_job_id: int = None,
#     verbose: bool = False,
#     ):
#     """Use OpenBabel to convert from pqr to PDB format.

#     Parameters
#     ----------
#     input_filename : str
#         Input PQR filename
#     main_job_id : int, optional
#         Job ID to associate with, by default None
#     verbose : bool, optional
#         Flag to print updates to the console, by default False

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     if verbose:
#         print ("Converting", input_filename, "to PDB format")
#     if not input_filename.endswith(".pqr"):
#         print ("Input format may be incorrect", input_filename)
#     print ("OUTPUT FILENAME", output_filename)
#     output_filename = obabel_convert(
#         input_format="pqr",
#         input_filename=input_filename,
#         output_format="pdb",
#         output_filename=output_filename,
#         main_job_id=main_job_id,
#         verbose=verbose,
#     )

#     print ("AFTER", input_filename, output_filename)

#     # delete input filename
#     # if input_filename != output_filename:
#     #     delete_file(input_filename, verbose=verbose)


#     return output_filename

def convert_3D_to_smiles(
    molecule_structure_filename,
    verbose: bool = False,
    ):

    if verbose:
        print ("Converting 3D structure file", molecule_structure_filename, "to SMILES string")

    try:

        stem, ext = os.path.splitext(molecule_structure_filename)
        ext = ext.replace(".", "")
        cmd = f"obabel -i{ext} {molecule_structure_filename} -osmi 2> /dev/null | awk '{{print $1}}'"

        if verbose:
            print ("Executing command", cmd)

        smiles_string = subprocess.check_output(cmd, shell=True)
        # convert to string 
        smiles_string = smiles_string.decode()
        # remove new line character
        smiles_string = smiles_string.strip()
        if verbose:
            print ("Obtained SMILES string", smiles_string)
        return smiles_string

    except Exception as e:
        print (molecule_structure_filename, "to SMILES exception", e)
    return None

    

# def calculate_RMSD_obabel(
#     ligand_filename_1, 
#     ligand_filename_2,
#     output_filename=None,
#     output_dir=None,
#     return_if_fail=float("inf"),
#     ):

#     print ("USING OPENBABEL TO CALC RMSD FOR FILES", ligand_filename_1, "AND", ligand_filename_2)
#     assert os.path.exists(ligand_filename_1), f"{ligand_filename_1} DOES NOT EXIST"
#     assert os.path.exists(ligand_filename_2), f"{ligand_filename_2} DOES NOT EXIST"

#     if output_filename is None:
#         output_filename = "RMSD.out"

#     if output_dir is not None and output_filename is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         output_filename = os.path.join(output_dir, output_filename)

#     cmd = f"obrms {ligand_filename_1} {ligand_filename_2}"
#     # if minimum:
#     #     cmd += " -m"
#     if output_filename is not None:
#         cmd += f" > {output_filename}"

#     execute_system_command(cmd)

#     rmsd = return_if_fail
#     if output_filename is not None:
#         print ("READING RMSD FROM", output_filename)
#         # read RMSD from file
#         with open(output_filename, "r") as f:
#             text = f.read().rstrip()
#             try:
#                 rmsd = float(text.split()[-1])
#             except:
#                 print ("EXCEPTION", text, text.split())
#                 pass
#         print ("REMOVING FILE", output_filename)
#         # os.remove(output_filename)
#     if output_dir is not None:
#         print("REMOVING DIRECTORY", output_dir)
#         # shutil.rmtree(output_dir) 
#     rmsd = min(return_if_fail, rmsd)
#     print ("OBTAINED RMSD", rmsd)
#     return rmsd


if __name__ == "__main__":


    # convert dud molecules 

    # root_dir = "/media/david/Elements/data/ai_blind_docking/dud/databases/dud_ligands2006"

    # import glob 

    # for compressed_mol2_filename in glob.iglob(os.path.join(root_dir, "*.mol2.gz")):
        
    #     compressed_mol2_base_name = os.path.basename(compressed_mol2_filename)

    #     target = compressed_mol2_base_name.split(".mol2.gz")[0]

    #     target_output_dir = os.path.join(root_dir, target)
    #     os.makedirs(target_output_dir)


    #     obabel_convert(
    #         input_format="mol2",
    #         input_filename=compressed_mol2_filename,
    #         output_filename=os.path.join(target_output_dir, target),
    #         output_format="mol2",
    #         multiple=True,
    #         verbose=True,
    #     )

    # smiles = convert_3D_to_smiles(
    #     molecule_structure_filename="test_compounds"
    # )

    # smiles_to_3D(
    #     supplied_molecules=[ 
    #         {
    #             "molecule_id": "DB03403",
    #             "smiles": "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    #         }
    #     ],
    #     output_format="mol2",
    #     output_dir="test_compounds",
    #     desired_output_filename="test_compounds/DB03403.mol2"
    # )

    import glob
    input_poses = glob.glob("drug_safety_MRGPRX1_ensemble_new/psovina/4/0/True/local_docking/Q96LB2/8DWC/binding_site_1/*/psovina/poses/pose_1.pdb")

    obabel_convert(
        input_format=None,
        input_filename=input_poses,
        output_format="mol2",
        output_filename="checkme.mol2",
        verbose=True,
    )
