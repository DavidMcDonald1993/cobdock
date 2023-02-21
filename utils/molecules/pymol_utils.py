
import os 
import sys
import os.path

if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir, 
        os.path.pardir,
    )))

import math

from pymol import cmd

from utils.molecules.openbabel_utils import obabel_convert

def rgyrate(
    selection: str,
    ):
    """Pymol utility function for radius of gyration.

    Taken from https://www.researchgate.net/post/How_to_calculate_hydrodynamic_radius_for_a_PDB_structure

    Parameters
    ----------
    selection : str
        Atom selection
    
    Returns
    -------
    float
        Radius of gyration
    """

    # from pymol import cmd

    # Get the atoms for the selection
    model=cmd.get_model(selection).atom
    # Extract the coordinates
    x=[i.coord for i in model]
    # Get the masses
    mass=[i.get_mass() for i in model]
    # Mass-weighted coordinates
    xm=[(m*i,m*j,m*k) for (i,j,k),m in zip(x,mass)]
    # Sum of masses
    tmass=sum(mass)
    # First part of the sum under the sqrt
    rr=sum(mi*i+mj*j+mk*k for (i,j,k),(mi,mj,mk) in zip(x,xm))
    # Second part of the sum under the sqrt
    mm=sum((sum(i)/tmass)**2 for i in zip(*xm))
    # Radius of gyration
    rg=math.sqrt(rr/tmass-mm)
    return rg

def cleanup_with_pymol(
    input_pdb_filename: str,
    output_pdb_filename: str = None,
    verbose: bool = False,
    ):
    """Remove all hetero atoms from a PDB file using PyMol.

    Parameters
    ----------
    input_pdb_filename : str
        The PDB to clean
    output_pdb_filename : str, optional
        The path to save the cleaned PDB structure to, by default None
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        Path of cleaned structure
    """

    # from pymol import cmd


    if output_pdb_filename is None:
        stem, ext = os.path.splitext(input_pdb_filename)
        output_pdb_filename = stem + "_pymol_cleaned" + ext

    # refresh program completely to allow re-reading previously read files
    cmd.reinitialize()

    if verbose:
        print ("Using PyMol to clean up PDB file", input_pdb_filename, )
        print ("Outputting cleaned structure to", output_pdb_filename)
    # upload to pymol
    cmd.load(input_pdb_filename)
    cmd.remove("resn HOH") # get rid of water
    # cmd.h_add("all")" # add all hydrogen
    cmd.h_add(selection="acceptor or donors") #only charged H is added to the target
    cmd.save(output_pdb_filename)
    cmd.remove("all") # something like closing the file (remove all atoms from memory)
    
    return output_pdb_filename

def create_complex_with_pymol(
    input_pdb_files: list,
    output_pdb_filename: str,
    verbose: bool = False,
    ):
    """Use PyMol to combine multiple PDB files into one complex file.

    Parameters
    ----------
    input_pdb_files : list
        A list of PDB files to combine into a complex
    output_pdb_filename : str
        The path to save the resulting complex to
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        Path of the resulting complex
    """

    # from pymol import cmd

    # refresh program completely to allow re-reading previously read files
    cmd.reinitialize()

    if verbose:
        print ("Writing complex containing", len(input_pdb_files), "structures")

    for input_pdb_file in input_pdb_files:
        if verbose:
            print ("Reading structure from", input_pdb_file)
        cmd.load(input_pdb_file)

    # change UNL residue ID
    # cmd.alter("resn 'UNL'", "chain='A'")  # 
    cmd.alter("resn 'UNL'", "resn='LIG'") 
    
    if verbose:
        print ("Writing complex to", output_pdb_filename)
    cmd.save(output_pdb_filename) 
    
    cmd.remove("all")

    return output_pdb_filename

def calculate_RMSD_pymol(
    reference_filename: str,
    model_filename: str,
    precision: int = 3,
    align: bool = False,
    ensure_same_residues: bool = True,
    remove_string: str = None,
    return_if_fail: float = None,
    verbose: bool = False,
    ):
    """Use PyMol to compute RMSD between two 3D structures.

    Parameters
    ----------
    reference_filename : str
        Path to PDB file of the reference structure
    model_filename : str
        Path to the PDB file of the model structure
    align : bool, optional
        Flag to perform aligning of the ligand before computing RMSD
    precision : int, optional
        Number of decimal places to round resulting RMSD to, by default 3
    return_if_fail : float, optional
        Number to return in the case of an error, by default None
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    float
        The computed RMSD between the two structures
    """
        
    # from pymol import cmd


    if not reference_filename.endswith(".pdb"):
        reference_basename, ext = os.path.splitext(reference_filename)
        ext = ext.replace(".", "")
        reference_filename = obabel_convert(
            input_format=ext,
            input_filename=reference_filename,
            output_format="pdb",
            output_filename=reference_basename + ".pdb"
        )

    if not model_filename.endswith(".pdb"):
        model_basename, ext = os.path.splitext(model_filename)
        ext = ext.replace(".", "")
        model_filename = obabel_convert(
            input_format=ext,
            input_filename=model_filename,
            output_format="pdb",
            output_filename=model_basename + ".pdb"
        )

    if verbose:
        print ("Using PyMol to compute RMSD for:", reference_filename, model_filename)

    # refresh program completely to allow re-reading previously read files
    cmd.reinitialize()

    # load the files into PyMol
    cmd.load(reference_filename, object="_reference", 
        # state=0,
        )
    cmd.load(model_filename, object="_model", 
        # state=0,
        )

    # set residue ID the same 
    if ensure_same_residues:
        cmd.alter("all", "resn='LIG'")
        cmd.alter("all", "resi=''")
        cmd.alter("all", "resv=1")

        # set name to element
        cmd.alter("all", "name=elem")

        # remove sequence id
        cmd.alter("all", "segi=''")

    # remove chain
    cmd.alter("all", "chain=''")

    # remove hydrogen
    cmd.remove("hydro")
    # add hydrogen
    # cmd.h_add("all")

    if remove_string is not None:
        cmd.remove(remove_string)

    cmd.sort("all")

    try:
        if align:
            if verbose:
                print ("Aligning molecules")
            rmsd = cmd.rms("_reference", "_model")
        else:
            rmsd = cmd.rms_cur("_reference", "_model")
        if verbose:
            print ("Obtained an RMSD value of", rmsd)

        cmd.remove("all")

        if isinstance(precision, int):
            rmsd = round(rmsd, precision)

        return rmsd 
    except Exception as e:
        print ("PYMOL RMSD exception", e)
        # raise e
        return return_if_fail

def calculate_radius_of_gyration(
    pdb_filename: str,
    verbose: bool = False,
    ):
    """Use PyMol to compute the radius of gyration for a given PDB file.

    Parameters
    ----------
    pdb_filename : str
        The path of the file to compute radius of gyration for
    verbose : bool, optional
        Flag to print updates to the consolde, by default False

    Returns
    -------
    float
        Radius of gyration
    """

    # from pymol import cmd


    if verbose:
        print ("Using PyMol to determine radius of gyration for file", pdb_filename)

    # reset pymol
    cmd.reinitialize()

    # load in molecule
    cmd.load(pdb_filename, object="_ligand")

    radius_of_gyration = rgyrate("_ligand")
    if verbose:
        print("Obtained radius of gyration:", radius_of_gyration)

    cmd.remove("all")

    return radius_of_gyration

def alter_molecule(
    input_filename: str,
    output_filename: str,
    selection: str,
    changes: list,
    verbose: bool = True,
    ):


    # from pymol import cmd

    if isinstance(changes, str):
        changes = [changes]

    # reset pymol
    cmd.reinitialize()

    # load in molecule
    cmd.load(input_filename, )

    for change in changes:
        cmd.alter(selection, change)

    cmd.save(output_filename)

    cmd.remove("all")

    return output_filename

def relabel_residue(
    input_filename: str,
    original_residue_labels: list,
    new_residue_label: str,
    output_filename: str = None,
    verbose: bool = False,
    ):
    """Use PyMol to relabel any residues in a given structure with a label in the list/set `original_residue_labels`
    to `new_residue_label`.

    Parameters
    ----------
    input_filename : str
        Path of the PDB file to relabel
    original_residue_labels : list
        List or set of residue labels to relabel
    new_residue_label : str
        The new residue label to use
    output_filename : str, optional
        Path to write the resultant structure, by default None.
        Will be generated if a path is not provided.
    verbose : bool, optional
        Flag to print updates to the consolde, by default False

    Returns
    -------
    str
        Path of the resultant structure
    """

    if verbose:
        print ("Relabelling all residues:", original_residue_labels, "to", new_residue_label)

    if isinstance(original_residue_labels, str):
        original_residue_labels = [original_residue_labels]

    if output_filename is None:
        stem, ext = os.path.splitext(input_filename)
        output_filename = f"{stem}_relabelled_{original_residue_labels}_{new_residue_label}{ext}"

    # change residue ID the same 
    selection = " | ".join((f"resn {res}" for res in original_residue_labels))
    changes = [f"resn='{new_residue_label}'"]

    return alter_molecule(
        input_filename=input_filename,
        output_filename=output_filename,
        selection=selection,
        changes=changes,
        verbose=verbose,
    )

def convert_to_ligand(
    input_filename: str,
    output_filename: str,
    verbose: bool = True,
    ):

    if verbose:
        print ("Converting", input_filename, "to ligand")

    selection = "all"
    changes = [ 
        "resn = 'UNL'",
        "chain = ''",
        "resi = 1",
        "type = 'HETATM'"
    ]

    return alter_molecule(
        input_filename=input_filename,
        output_filename=output_filename,
        selection=selection,
        changes=changes,
        verbose=verbose,
    )

def convert_file_with_pymol(
    input_filename,
    output_filename,
    verbose: bool = True,
    ):

    # from pymol import cmd


    if verbose:
        print ("Using PyMol to convert", input_filename, "to", output_filename)

    # reset pymol
    cmd.reinitialize()

    # load in molecule
    cmd.load(input_filename, )

    # save the file
    cmd.save(output_filename)

    cmd.remove("all")

    return output_filename


if __name__ == "__main__":

    # from utils.io.io_utils import load_json
    
    # data = load_json("checkme.json")

    # receptors = data["receptors"]

    # poses = data["top_poses"]

    # output_dir = "test_complex_dir"

    # receptor_dir = os.path.join(output_dir, "receptors")
    # os.makedirs(receptor_dir, exist_ok=True)

    # poses_dir = os.path.join(output_dir, "poses")
    # os.makedirs(poses_dir, exist_ok=True)

    # complexes_dir = os.path.join(output_dir, "complexes")
    # os.makedirs(complexes_dir, exist_ok=True)


    # print (len(receptors), len(poses))

    # receptor = receptors[0]

    # receptor_filename = os.path.join(receptor_dir, "receptor.pdb")
    # print ("Writing receptor to", receptor_filename)
    # with open(receptor_filename, "w") as f:
    #     f.write(receptor["pdb_text"])

    # # poses
    # for pose in poses:
    #     pose_name = pose["molecule_id"]

    #     pose_filename = os.path.join(poses_dir, pose_name + ".pdbqt")
    #     print ("Writing pose to", pose_filename)

    #     with open(pose_filename, "w") as f:
    #         f.write(pose["pdbqt_text"])

    #     complex_filename = os.path.join(complexes_dir, pose_name + ".pdb")

    #     create_complex_with_pymol(
    #         input_pdb_files=[receptor_filename, pose_filename],
    #         output_pdb_filename=complex_filename,
    #         verbose=True
    #     )

    create_complex_with_pymol(
        ["multigrow_test_dir/preparation/prepared_targets/P49327/6NNA_A/6NNA_A_prepared_protonated_7.4.pdb", "docking_result.pdbqt"],
        "my_complex.pdb",
    )