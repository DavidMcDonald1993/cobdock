if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        )))

import os

# from utils.io.io_utils import delete_directory, delete_file
# from utils.molecules.openbabel_utils import obabel_convert 

from utils.sys_utils import execute_system_command

# python API is also possible for more options / control

def fix_pdb_file_with_pdbfixer(
    pdb_filename: str,
    output_filename: str,
    add_atoms: str = "all", # all/heavy/hydrogen/none
    keep_heterogens: str = "none", # all/water/none
    replace_nonstandard_residues: bool = True,
    add_missing_residues: bool = True,
    pH: float = 7.0,
    # check out the documentation for waterbox options (for MD)
    verbose: bool = True,
    ):

    add_atoms = add_atoms.lower()
    if add_atoms not in {"all", "heavy", "hydrogen", "none"}:
        add_atoms = "none"
    keep_heterogens = keep_heterogens.lower()
    if keep_heterogens not in {"all", "water", "none"}:
        keep_heterogens = "all"

    if verbose:
        print ("Running pdbfixer on pdb file", pdb_filename)
        print ("Adding atoms:", add_atoms)
        print ("Keeping heterogens:", keep_heterogens)
        print ("Outputting to", output_filename)
    
    cmd = f"pdbfixer {pdb_filename} --output {output_filename} --add-atoms {add_atoms} --keep-heterogens {keep_heterogens} --ph={pH}"
    if replace_nonstandard_residues:
        cmd += " --replace-nonstandard"
    if add_missing_residues:
        cmd += "  --add-residues"
    if verbose:
        cmd += " --verbose"

    try:
        execute_system_command(cmd, verbose=verbose)
    except Exception as e:
        print ("PDBFixer exception", e)
        # raise e
        return None 

    return output_filename

if __name__ == "__main__": 
    fix_pdb_file_with_pdbfixer(
        # pdb_filename="1htp.pdb",
        # pdb_filename="6nna_A.pdb",
        pdb_filename="AF-B1MC30-test.pdb",
        # pdb_filename="checkme.pdb",
        output_filename="checkme_3.pdb",
        add_atoms="all",
        keep_heterogens="none",
        add_missing_residues=True,
        replace_nonstandard_residues=True,
        pH=10,
        verbose=True,
    )