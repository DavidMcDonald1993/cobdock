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

from Bio.PDB import PDBParser, PDBIO, PDBList, Structure, Model, Chain, Residue, Atom, NeighborSearch
from Bio.PDB.PDBIO import Select
from Bio.PDB.parse_pdb_header import parse_pdb_header

import scoria

import numpy as np

from utils.molecules.openbabel_utils import obabel_convert
from utils.io.io_utils import copy_file, delete_directory, delete_file, write_json, load_compressed_pickle, gzip_file
from utils.sys_utils import execute_system_command

import collections

# vdW radii taken from:
# https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
#
# Radii for CL, K, NA, etc are _not_ ionic radii.
#
# References:
# A. Bondi (1964). "van der Waals Volumes and Radii".
# M. Mantina, A.C. et al., J. Phys. Chem. 2009, 113, 5806.
ATOMIC_RADII = collections.defaultdict(lambda: 2.0)
ATOMIC_RADII.update(
    {
        "H": 1.200,
        "HE": 1.400,
        "C": 1.700,
        "N": 1.550,
        "O": 1.520,
        "F": 1.470,
        "NA": 2.270,
        "MG": 1.730,
        "P": 1.800,
        "S": 1.800,
        "CL": 1.750,
        "K": 2.750,
        "CA": 2.310,
        "NI": 1.630,
        "CU": 1.400,
        "ZN": 1.390,
        "SE": 1.900,
        "BR": 1.850,
        "CD": 1.580,
        "I": 1.980,
        "HG": 1.550,
    }
)

# map three letter AA codes to one letter AA code
# taken from PyBioMed.PyGetMol.GetProtein
AA3_TO_AA1 = [
    ("ALA", "A"),
    ("CYS", "C"),
    ("ASP", "D"),
    ("GLU", "E"),
    ("PHE", "F"),
    ("GLY", "G"),
    ("HIS", "H"),
    ("HSE", "H"),
    ("HSD", "H"),
    ("ILE", "I"),
    ("LYS", "K"),
    ("LEU", "L"),
    ("MET", "M"),
    ("MSE", "M"),
    ("ASN", "N"),
    ("PRO", "P"),
    ("GLN", "Q"),
    ("ARG", "R"),
    ("SER", "S"),
    ("THR", "T"),
    ("VAL", "V"),
    ("TRP", "W"),
    ("TYR", "Y"),
]
# convert to dict
AA3_TO_AA1 = {
    k: v 
    for k, v in AA3_TO_AA1
}
AA1_TO_AA3 = {
    v: k 
    for k, v in AA3_TO_AA1.items()
}

# map accession to cofactor
ACCESSION_TO_COFACTOR_FILENAME = "data/databases/pdb/accession_to_cofactor_id.pkl.gz"
ACCESSION_TO_COFACTOR = load_compressed_pickle(ACCESSION_TO_COFACTOR_FILENAME, verbose=False) # will be {} if file does not exist

# set of all cofactors in PDB
ALL_COFACTORS_FILENAME = "data/databases/pdb/all_cofactors.pkl.gz"
ALL_COFACTORS = load_compressed_pickle(ALL_COFACTORS_FILENAME, verbose=False)

# map PDB ligand type to ID
LIGAND_TYPE_TO_ID_FILENAME = "data/databases/pdb/ligand_type_to_id.pkl.gz"
LIGAND_TYPE_TO_ID = load_compressed_pickle(LIGAND_TYPE_TO_ID_FILENAME, verbose=False)

class ModelSelect(Select):

    def __init__(
        self, 
        model_ids,
        ):
        """Initialiser 

        Parameters
        ----------
        chain_ids : set
            Set of chain IDs to keep. A list, int or str can also be supplied.
        """
        super(ModelSelect, self).__init__()
        if isinstance(model_ids, int) or isinstance(model_ids, str):
            model_ids = {model_ids}
        self.model_ids = model_ids

    
    def accept_model(
        self, 
        model: Model,
        ):
        """Custom select model function.

        Parameters
        ----------
        model : Model
            Biopython model object to check

        Returns
        -------
        bool
            Flag to keep model
        """
        return model.id in self.model_ids


class ChainIDSelect(Select):


    def __init__(
        self, 
        chain_ids,
        ):
        """Initialiser 

        Parameters
        ----------
        chain_ids : set
            Set of chain IDs to keep. A list, int or str can also be supplied.
        """
        super(ChainIDSelect, self).__init__()
        if isinstance(chain_ids, int) or isinstance(chain_ids, str):
            chain_ids = {chain_ids}
        self.chain_ids = chain_ids

    
    def accept_chain(
        self, 
        chain: Chain,
        ):
        """Custom select chain function.

        Parameters
        ----------
        chain : Chain
            Biopython chain object to check

        Returns
        -------
        bool
            Flag to keep chain
        """
        return chain.id in self.chain_ids

class ChainSelect(Select):

    def __init__(
        self, 
        chains,
        clean_chains: bool = True,
        ):
        """Initialiser 

        Parameters
        ----------
        chains : set
            Set of chains to keep. A list, int or str can also be supplied.
        clean_chains : bool, optional
            Flag to remove hetero residues from chains 
        """
        super(ChainSelect, self).__init__()
        if isinstance(chains, int) or isinstance(chains, str):
            chains = {chains}
        self.chains = chains
        self.clean_chains = clean_chains

    
    def accept_chain(
        self, 
        chain: Chain,
        ):
        """Custom select chain function.

        Parameters
        ----------
        chain : Chain
            Biopython chain object to check

        Returns
        -------
        bool
            Flag to keep chain
        """
        return chain in self.chains

    def accept_residue(
        self, 
        residue: Residue,
        ):
        """Custom accept function for residues.
        Returns true if the hetero_flag for the residue is " "

        Parameters
        ----------
        residue : Residue
            Biopython residue to check

        Returns
        -------
        bool
            Flag to keep residue
        """
        # accept all residues
        if not self.clean_chains:
            return True
        # accept non-hetero residues
        hetero_flag, sequence_identifier, insertion_code = residue.id
        return hetero_flag == " "


# class ResidueSelect(Select):


#     def __init__(
#         self, 
#         residues,
#         ):
#         """Initialiser 

#         Parameters
#         ----------
#         residues : set
#             Set of residues to keep. A list or str can also be supplied.
#         """
#         super(ResidueSelect, self).__init__()
#         if isinstance(residues, int) or isinstance(residues, str):
#             residues = {residues}
#         self.residues = residues

    
#     def accept_residue(
#         self, 
#         residue: Residue,
#         ):
#         """Custom select residue function.

#         Parameters
#         ----------
#         residue : Residue
#             Biopython residue object to check

#         Returns
#         -------
#         bool
#             Flag to keep residue
#         """
#         return residue in self.residues

class ResidueSelect(Select):


    def __init__(
        self, 
        residue_ids,
        ):
        """Initialiser 

        Parameters
        ----------
        residue_ids : set
            Set of (resname, residue_id) pairs to keep. 
        """
        super(ResidueSelect, self).__init__()
        self.residue_ids = residue_ids

    
    def accept_residue(
        self, 
        residue: Residue,
        ):
        """Custom select residue function.

        Parameters
        ----------
        residue : Residue
            Biopython residue object to check

        Returns
        -------
        bool
            Flag to keep residue
        """
        resname = residue.resname
        _, residue_id, _ = residue.id
        return (resname, residue_id) in self.residue_ids

class ChainResidueSelect(Select):

    def __init__(
        self, 
        chain_residues,
        ):
        """Initialiser 

        Parameters
        ----------
        chain_residues : set
            Set of chain_id, resname, residue tuples to keep.
        """
        super(ChainResidueSelect, self).__init__()
        if isinstance(chain_residues, list):
            chain_residues = set(chain_residues)
        self.chain_residues = chain_residues

    def accept_residue(
        self, 
        residue: Residue,
        ):
        """Custom select residue function.

        Parameters
        ----------
        residue : Residue
            Biopython residue object to check

        Returns
        -------
        bool
            Flag to keep residue
        """

        chain_id = residue.get_parent().id
        _, residue_id, _ = residue.id
        resname = residue.resname

        return (chain_id, resname, residue_id) in self.chain_residues

class AtomSelect(Select):

    def __init__(
        self, 
        atoms,
        ):
        """Initialiser 

        Parameters
        ----------
        atoms : set
            Set of atoms to keep. A list or str can also be supplied.
        """
        super(AtomSelect, self).__init__()
        if isinstance(atoms, int) or isinstance(atoms, str):
            atoms = {atoms}
        self.atoms = atoms

    
    def accept_atom(
        self, 
        atom: Atom,
        ):
        """Custom select residue function.

        Parameters
        ----------
        atom : Atom
            Biopython Atom object to check

        Returns
        -------
        bool
            Flag to keep atom
        """
        return atom in self.atoms

class CleanResidueSelect(Select):

    '''
    custom select class to only accept residues with hetero_flag == " "
    '''
    
    def __init__(self, ):
        """Initialiser
        """
        super(CleanResidueSelect, self).__init__()
    
    def accept_residue(
        self, 
        residue: Residue,
        ):
        """Custom accept function for residues.
        Returns true if the hetero_flag for the residue is " "

        Parameters
        ----------
        residue : Residue
            Biopython residue to check

        Returns
        -------
        bool
            Flag to keep residue
        """
        hetero_flag, sequence_identifier, insertion_code = residue.id
        return hetero_flag == " "

def get_up_to_date_list_of_pdb_ids(
    as_json: bool = False,
    verbose: bool = False,
    ):
    """Use Biopython to retrieve a complete list of PDB IDs.

    Parameters
    ----------
    as_json : bool, optional
        Flag to return a list of dicts, by default False
    verbose : bool
        Flag to print updates to the console, by default False.

    Returns
    -------
    list
        A list of all IDs for all available structure in the PDB database.
    """
    if verbose:
        print ("Retrieving a complete list of PDB IDs")
    pdbl = PDBList()
    pdb_ids = pdbl.get_all_entries()
    if as_json:
        pdb_ids = [ 
            {"pdb_id": pdb_id}
            for pdb_id in pdb_ids
        ]
    return pdb_ids


def download_pdb_structure_using_pdb_fetch(
    pdb_id: str,
    pdb_filename,
    verbose: bool = False,
    ):
    """Use pdb_fetch to download PDB structure, given a PDB ID (can be used as an altrnative to Biopython)

    Parameters
    ----------
    pdb_id : str
        PDB ID of structure to download
    pdb_filename : _type_
        File path to write to 
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        File path of structure
    """
    # if os.path.exists(pdb_filename):
    #     return pdb_filename
    pdb_id = pdb_id.strip()
    if len(pdb_id) != 4:
        print (pdb_id, "contains too many characters")
        return None
    # download pdb file
    cmd = f"pdb_fetch {pdb_id} > {pdb_filename}"
    try:
        # 1 minute timeout
        execute_system_command(cmd, timeout="1m", verbose=verbose)
    except Exception as e:
        print ("pdb_fetch exception", e)
        return None 
    return pdb_filename

def download_pdb_file_using_biopython(
    pdb_id: str, 
    download_dir: str,
    verbose: bool = False,
    ):
    """Use Biopython package to download the PDB file for the provided pdb_id.
    The file will be downloaded into directory `download_dir`/`pdb_id`.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the structure to download
    download_dir : str
        The root download directory
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        The filename of the downloaded file
    """
    pdb_id = str(pdb_id)
    pdb_id = pdb_id.upper()
    
    download_dir = os.path.join(
        download_dir, 
        pdb_id)
    os.makedirs(download_dir, exist_ok=True)

    if verbose:
        print ("Downloading PDB file for", pdb_id, "to directory", download_dir, "using Biopython")
  
    pdbl = PDBList()
    return pdbl.retrieve_pdb_file(
        pdb_id,
        file_format="pdb", 
        pdir=download_dir)

def get_structure_biopython(
    pdb_id: str,
    pdb_filename: str,
    output_dir: str,
    verbose: bool = False,
    ):

    if verbose:
        print ("Getting structure for PDB ID", pdb_id)
        print ("Using PDB filename", pdb_filename)
    
    if pdb_filename is None:
        # download the structure first 
        if verbose:
            print ("No filename provided, downloading structure to", output_dir)
        pdb_filename = download_pdb_file_using_biopython(
            pdb_id=pdb_id,
            download_dir=output_dir,
            verbose=verbose,
        )

    # set structure and return it 
    parser = PDBParser()
    return parser.get_structure(
        pdb_id, 
        pdb_filename)


def get_number_of_models_in_pdb_file(
    pdb_id: str,
    pdb_filename: str,
    output_dir: str,
    verbose: bool = False,
    ):
    return len(get_structure_biopython(
        pdb_id=pdb_id,
        pdb_filename=pdb_filename,
        output_dir=output_dir,
        verbose=verbose,
    ))

def write_single_model(
    pdb_id: str,
    pdb_filename: str,
    output_dir: str,
    output_filename: str = None,
    selected_model_id: int = None,
    verbose: bool = False,
    ):
   
    structure = get_structure_biopython(
        pdb_id=pdb_id,
        pdb_filename=pdb_filename,
        output_dir=output_dir,
        verbose=verbose,
    ) 

    all_model_ids = [model.id for model in structure]

    # select first model if it is not selected
    if selected_model_id is None or selected_model_id not in all_model_ids:
        selected_model_id = all_model_ids[0]

    if output_filename is None:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(
            output_dir,
            f"{pdb_id}_model_{selected_model_id}.pdb")
    
    if verbose:
        print ("Writing first model", selected_model_id, "in PDB file", pdb_filename, "to file", output_filename)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filename, ModelSelect(selected_model_id))

    return output_filename

def write_all_models(
    pdb_id: str,
    pdb_filename: str,
    output_dir: str,
    verbose: bool = False,
    ):
    """Write all models in a PDB file into separate PDB files in a specified directory.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the structure
    pdb_filename : str
        The filename to write the PDB file to
    output_dir : str
        The directory to write the model PDB files to
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    list
        A list of written model filenames
    """

    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print ("Extracting all models in file", pdb_filename, "to directory", output_dir)

    structure = get_structure_biopython(
        pdb_id=pdb_id,
        pdb_filename=pdb_filename,
        output_dir=output_dir,
        verbose=verbose,
    )

    model_filenames = []

    for model in structure:

        model_id = model.id

        model_filename = os.path.join(
            output_dir, 
            f"{pdb_id}-model-{model_id}.pdb")

        io = PDBIO()
        io.set_structure(model)
        print ("writing model to", model_filename)
        io.save(model_filename)

        model_filenames.append(model_filename)

    return model_filenames

def select_chains_from_pdb_file(
    pdb_id: str,
    pdb_filename: str,
    chain_ids,
    output_filename: str = None,
    verbose: bool = False,
    ):
    """Select chain(s) from a PDB file

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the structure
    pdb_filename : str
        The filename to write the PDB file to
    chain_ids : list
        Chain IDs to keep
    output_filename : str, optional
        Filename to write the output PDB file to, by default None
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        Filename of output PDB file
    """

    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]

    if output_filename is None:
        stem, ext = os.path.splitext(pdb_filename)
        chain_id_str = "".join(chain_ids)
        output_filename = stem + f"_{chain_id_str}" + ext

    if verbose:
        print ("Selecting chains", chain_ids, "from PDB file", pdb_filename)
        print ("writing chain(s) to", output_filename)
    
    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filename, ChainIDSelect(chain_ids))

    return output_filename
    
def get_number_of_atoms_in_pdb_file(
    pdb_id: str,
    pdb_filename: str,
    verbose: bool = False,
    ):
    """Count the number of atoms in a PDB file

    Parameters
    ----------
    pdb_id : str
        the PDB ID of the structure
    pdb_filename : str
        The filename to write the PDB file to

    Returns
    -------
    int
        The number of atoms in the file
    """

    if verbose:
        print ("Getting number of atoms in PDB file", pdb_filename)

    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    num_atoms = 0
    for _ in structure.get_atoms():
        num_atoms += 1
    return num_atoms

def get_number_of_residues_in_PDB_file(
    pdb_id: str,
    pdb_filename: str,
    verbose: bool = False,
    ):
    """Count the number of residues in a PDB file

    Parameters
    ----------
    pdb_id : str
        the PDB ID of the structure
    pdb_filename : str
        The filename to write the PDB file to

    Returns
    -------
    int
        The number of residues in the file
    """

    if verbose:
        print ("Counting number of residues in PDB file", pdb_filename)

    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    num_residues = 0
    for _ in structure.get_residues():
        num_residues += 1
    return num_residues

def get_all_chain_ids_in_a_PDB_file(
    pdb_id: str,
    pdb_filename: str,
    ):
    """Return a list containing the IDs of all of the chains in a PDB file

    Parameters
    ----------
    pdb_id : str
        the PDB ID of the structure
    pdb_filename : str
        The filename to write the PDB file to

    Returns
    -------
    list
        All of the chain IDs in the supplied PDB file
    """

    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    chain_ids = []
    for chain in structure.get_chains():
        chain_id = chain.id.strip()
        if chain_id == "":
            continue
        chain_ids.append(chain.id)
    return chain_ids

def remove_all_hetero_residues_using_biopython(
    pdb_id: str,
    pdb_filename: str,
    output_filename: str = None,
    ):
    """Uses CleanResidueSelect to remove all hetero residues from a PDB file.

    Parameters
    ----------
    pdb_id : str
        the PDB ID of the structure
    pdb_filename : str
        The filename to write the PDB file to
    output_filename : str, optional
        The filename to save the cleaned PDB file to, will be generated and returned if not supplied, by default None

    Returns
    -------
    str
        The output filename
    """

    if output_filename is None:
        stem, ext = os.path.splitext(pdb_filename)
        output_filename = stem + "_cleaned" + ext

    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)
    
    # write all models to file (removing all hetero-residues)
    io = PDBIO()
    io.set_structure(structure)
    print ("Writing cleaned structure to", output_filename)
    io.save(output_filename, CleanResidueSelect())
    print ("Write successful",)


    return output_filename

def search_natural_ligand_in_pdb_structure(
    structure: Structure,
    desired_model: str = None,
    desired_chain: str = None,
    desired_ligand_symbol: str = None,
    desired_sequence_identifier: str = None,
    return_closest_chain: bool = False,
    min_atoms: int = 5,
    verbose: bool = False,
    ):
    
    if desired_chain is not None:
        desired_chain = str(desired_chain)
        desired_chain = desired_chain.upper()
    if desired_ligand_symbol is not None:
        desired_ligand_symbol = str(desired_ligand_symbol)
        desired_ligand_symbol = desired_ligand_symbol.upper() 

    for model in structure:

        model_id = model.id
        if desired_model is not None and model_id != desired_model:
            if verbose:
                print("Skipping model ID:", model_id,)
            continue

        for chain in model:

            chain_id = chain.id
            if desired_chain is not None and chain_id != desired_chain:
                if verbose:
                    print ("Skipping chain ID:", chain_id, )
                continue

            for residue in chain:
                hetero_flag, sequence_identifier, insertion_code = residue.id

                if desired_sequence_identifier is not None and sequence_identifier != desired_sequence_identifier:
                    if verbose:
                        print ("Skipping sequence identifier:", sequence_identifier,)
                    continue

                # skip all non-hetero residues
                if not hetero_flag.startswith("H_"):
                    continue
                    
                # skip glucose
                if hetero_flag in {"H_GLC", }: # ignore glucose
                    continue

                # check for number of atoms
                residue_atoms = list(residue.get_atoms())
                if isinstance(min_atoms, int) and len(residue_atoms) < min_atoms:
                    continue
                    
                ligand_symbol = residue.get_resname()

                # skip sulfate ion
                if ligand_symbol == "SO4":
                    continue

                if desired_ligand_symbol is not None and ligand_symbol != desired_ligand_symbol:
                    # skip ligand
                    continue

                if verbose:
                    print ("Found ligand", ligand_symbol, 
                        "in chain", chain_id, "in model", model_id, 
                        "with sequence identifier", sequence_identifier)

                # return current chain
                if return_closest_chain:
                    return residue, chain
                # just return residue 
                return residue
    
    # failed to find suitable natural ligand
    raise ValueError("Could not find suitable natural ligand")

def extract_natural_ligand_and_chain(
    pdb_id: str, 
    pdb_filename: str,
    output_dir: str,
    desired_model: str = None,
    desired_chain: str = None,
    desired_ligand_symbol: str = None,
    desired_sequence_identifier: str = None,
    min_atoms: int = None,
    return_closest_chain: bool = True,
    ligand_filename: str = None,
    chain_filename: str = None,
    verbose: bool = False,
    ):
    """Write a PDB file for a co-crytalised natural ligand in a PDB file and write a PDB for the corresponding chain.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the target
    pdb_filename : str
        The PDB file
    output_dir : str
        Optional directory name to write the files to
    desired_model : str, optional
        The model of the desired natural ligand, by default None
    desired_chain : str, optional
        The chain of the desired natural ligand, by default None
    desired_ligand_symbol : str, optional
        The symbol of the desired natural ligand, by default None
    desired_sequence_identifier : str, optional
        The sequence identifier of the desired natural ligand, by default None
    min_atoms : int, optional
        Minimum number of atoms in residue, by default None
    return_chain : bool, optional
        Flag to return the chain in addition to the natural ligand, by default True
    ligand_filename : str, optional
        Filename to save the ligand (in PDB format) to, by default None
    chain_filename : str, optional
        Filename to save the chain (in PDB format) to, by default None
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        The filename(s) of the ligand (and corresponding chain)

    """

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    chain, residue = search_natural_ligand_in_pdb_structure(
        structure=structure,
        desired_model=desired_model,
        desired_chain=desired_chain,
        desired_ligand_symbol=desired_ligand_symbol,
        desired_sequence_identifier=desired_sequence_identifier,
        min_atoms=min_atoms,
        return_closest_chain=True,
        verbose=verbose,
    )

    chain_id = chain.id
    ligand_symbol = residue.get_resname()
    hetero_flag, sequence_identifier, insertion_code = residue.id

    # write ligand to file
    io = PDBIO()
    io.set_structure(residue)
    if ligand_filename is None: # auto-generate ligand_filename from model/chain ID, ligand symbol etc.
        ligand_filename = os.path.join(
            output_dir, 
            f"{pdb_id}-chain-{chain_id}-{ligand_symbol}-{sequence_identifier}-{insertion_code}.pdb")
    if verbose:
        print ("writing ligand to", ligand_filename)
    io.save(ligand_filename)

    if not return_closest_chain: # only return ligand_filename
        return ligand_filename
    
    # write chain to file (removing all hetero-residues)
    io = PDBIO()
    io.set_structure(chain)
    if chain_filename is None:
        chain_filename = os.path.join(
            output_dir, 
            f"{pdb_id}-chain-{chain_id}.pdb")
    if verbose:
        print ("writing chain to", chain_filename)
    # remove hetero residues and save
    io.save(chain_filename, CleanResidueSelect())
    
    # return both chain and ligand filename
    return chain_filename, ligand_filename


def extract_closest_chain(
    pdb_id: str, 
    pdb_filename: str,
    point,
    output_dir: str,
    desired_model: str = None,
    desired_chain: str = None,
    verbose: bool = False,
    ):
    """Identify the closest chain (with respect to Euclidean distance) to a point. 

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the target
    pdb_filename : str
        The filename of the PDB file
    point : np.ndarray
        Euclidean co-ordinates of the point
    output_dir : str
        The directory to write the chains to
    desired_model : str, optional
        The model of interest in the PDB file, by default None
    desired_chain : str, optional
        The chain of interest in the PDB file (providing this will skip selection by distance), by default None
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        The filename of the closest chain
    """
    os.makedirs(output_dir, exist_ok=True)

    if point is not None and not isinstance(point, np.ndarray):
        point = np.array(point)
    
    if desired_chain is not None:
        desired_chain = str(desired_chain)
        desired_chain = desired_chain.upper()
        if verbose:
            print ("Selecting chain", desired_chain)
    else:
        if verbose:
            print ("Identifying closest chain to point", point, "and writing chains to", output_dir)
    
    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    chain_distances = {}

    for model in structure:

        model_id = model.id
        if desired_model is not None and model_id != desired_model:
            if verbose:
                print("Skipping model ID:", model_id)
            continue

        for chain in model:

            chain_id = chain.id
           
             # write chain to file (removing all hetero-residues)
            io = PDBIO()
            io.set_structure(chain)
            chain_filename = os.path.join(output_dir, 
                f"{pdb_id}-model-{model_id}-chain-{chain.id}.pdb")
            if verbose:
                print ("writing chain to", chain_filename)
            io.save(chain_filename, CleanResidueSelect()) # use CleanResidueSelect to clean chain (remove residues)

            if desired_chain is not None and chain_id == desired_chain:
                return chain_filename # found chain

            # read chain as scoria molecule and compute distance to point
            chain_centre_of_mass = identify_centre_of_mass(
                mol_path=chain_filename,
                geometric=True,)
            chain_centre_of_mass = np.array(chain_centre_of_mass)

            # determine euclidean distance
            distance_to_point = np.linalg.norm(chain_centre_of_mass - point)
            chain_distances[chain_filename] = distance_to_point

    # select filename with smallest distance
    return min(chain_distances, key=chain_distances.get)

def identify_contact_residues_for_natural_ligand(
    pdb_id: str, 
    pdb_filename: str,
    natural_ligand_id: str = None,
    natural_ligand_pdb_filename: str = None,
    atom_centers = None,
    radius: int = 4,
    desired_model: str = None,
    desired_chain: str = None,
    desired_ligand_symbol: str = None,
    desired_sequence_identifier: str = None,
    min_atoms: int = 5,
    verbose: bool = False,
    ):

    # load protein structur
    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    # construct NeighborSearch instance using all atoms in structure
    neighbor = NeighborSearch(list(structure.get_atoms())) 

    if atom_centers is None: # no list of atoms provided

        # load natural ligand from PDB file
        if natural_ligand_pdb_filename is not None:
            # load from pdb file
            if verbose:
                print ("Loading natural ligand from file", natural_ligand_pdb_filename)
            natural_ligand_structure = parser.get_structure(
                natural_ligand_id,
                natural_ligand_pdb_filename,
            )
            # get residue (should be single residue in structure)
            natural_ligand_residue = list(natural_ligand_structure.get_residues())[0]

            atom_centers = [ 
                atom.get_coord()
                for atom in natural_ligand_residue.get_atoms()
                if atom.element != "H" # only heavy atoms
            ]

        else: # search for natural ligand in target PDB file using chain/symbol/sequence identifier etc.
            if verbose:
                print ("Identifying a natural ligand in the PDB structure")
            # find in protein structure
            natural_ligand_residue = search_natural_ligand_in_pdb_structure(
                structure=structure,
                desired_model=desired_model,
                desired_chain=desired_chain,
                desired_ligand_symbol=desired_ligand_symbol,
                desired_sequence_identifier=desired_sequence_identifier,
                min_atoms=min_atoms,
                return_closest_chain=False,
                verbose=verbose,
            )

            atom_centers = [ 
                atom.get_coord()
                for atom in natural_ligand_residue.get_atoms()
                if atom.element != "H" # only heavy atoms
            ]
    elif verbose:
        print ("Using", len(atom_centers), "pre-defined atom location(s)")


    # build set of residues within `radius`A from any atom in atom_centers list
    selected_residues = set()

    # for atom in natural_ligand_residue.get_atoms():
    for atom_center in atom_centers:
        selected_residues.update(
            neighbor.search(
                center=atom_center,
                radius=radius, 
                level='R', # residue
            )
        ) 
    # remove hetero residues from selected_residues
    selected_residues = set(
        filter(
            lambda selected_residue: selected_residue.id[0] == " ",
            selected_residues,    
        ),
    )

    # convert to list of (chain_id, resname, residue_id) tuples
    selected_residues = {
        (
            selected_residue.get_parent().id, 
            selected_residue.resname, 
            selected_residue.id[1]
        )
        for selected_residue in selected_residues
    }


    # get list of unique chains
    selected_chain_ids = { 
        # selected_residue.get_parent().id
        chain_id
        for chain_id, resname, residue_id in selected_residues 
        # for chain_id, residue_id in selected_residues 
    }

    return selected_residues, selected_chain_ids

def build_binding_site_based_on_distance_to_natural_ligand(
    pdb_id: str, 
    pdb_filename: str,
    output_pocket_filename: str,
    output_chains_filename: str = None,
    natural_ligand_id: str = None,
    natural_ligand_pdb_filename: str = None,
    atom_centers = None,
    radius: int = 8,
    desired_model: str = None,
    desired_chain: str = None,
    desired_ligand_symbol: str = None,
    desired_sequence_identifier: str = None,
    min_atoms: int = 5,
    verbose: bool = False,
    ):

    if verbose:
        print ("Building a binding site using pdb_filename", pdb_filename,
            "and radius", radius)

    # load protein structur
    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    # returns set of (chain_id, resname, residue_id) tuples
    selected_residues, selected_chain_ids = identify_contact_residues_for_natural_ligand(
        pdb_id=pdb_id,
        pdb_filename=pdb_filename,
        natural_ligand_id=natural_ligand_id,
        natural_ligand_pdb_filename=natural_ligand_pdb_filename,
        atom_centers=atom_centers,
        radius=radius,
        desired_model=desired_model,
        desired_chain=desired_chain,
        desired_ligand_symbol=desired_ligand_symbol,
        desired_sequence_identifier=desired_sequence_identifier,
        min_atoms=min_atoms,
        verbose=verbose,
    )

    # write selected residues to file
    io = PDBIO()
    io.set_structure(structure)

    if verbose:
        print ("Saving pocket to", output_pocket_filename) 

    # save residues in structure that are in selected residues 
    # ChainResidueSelect requires (chain_id, resname, residue_id) tuples
    io.save(output_pocket_filename, ChainResidueSelect(selected_residues))

    if output_chains_filename is None:
        return output_pocket_filename

    # select chains participating in binding site
    # # build set of chains within `radius`A from any atom in atom_list
    # selected_chains = set()
    # for atom_center in atom_centers:
    #     selected_chains.update(
    #         neighbor.search(
    #             center=atom_center,
    #             radius=radius, 
    #             level='C', # chain
    #         )
    #     ) 

    io = PDBIO()
    io.set_structure(structure)

    if verbose:
        print ("Saving selected chain(s) to", output_chains_filename)

    # # save chains in structure that are in selected chains 
    # io.save(output_chains_filename, ChainSelect(selected_chains, clean_chains=True,))
    
    io.save(output_chains_filename, ChainIDSelect(selected_chain_ids))

    return output_pocket_filename, output_chains_filename



def get_all_natural_ligands_from_pdb_file(
    pdb_id: str = None,
    pdb_filename: str = None,
    output_dir: str = "pdb_files",
    min_heavy_atoms: int = 4,
    delete_output_dir: bool = True,
    compress_ligands: bool = True,
    verbose: bool = False,
    ):
    """Extract all natural ligands from a given PDB file.
    If a file is not supplied, a PDB ID can be supplied to have file download.

    Parameters
    ----------
    pdb_id : str, optional
        The PDB ID of the target, by default None
    pdb_filename : str, optional
        The filename of the PDB structure, by default None
    output_dir : str, optional
        The directory to download to, by default "pdb_files"
    min_atoms : int, optional
        The minimum number of atoms in a natural ligand, by default 4
    cleanup : bool, optional
        Flag to delete any downloaded files after running, by default True
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    list
        A list of dicts describing all of the natural ligands (with at least `min_atoms` atoms)
        in the PDB file. 
    """
    if pdb_filename is None:
        assert pdb_id is not None, "pdb_filename or pdb_id must be provided"
        # download PDB file
        pdb_filename = download_pdb_file_using_biopython(
            pdb_id,
            download_dir=output_dir)
    if pdb_id is None:
        pdb_file_basename = os.path.basename(pdb_filename)
        pdb_id, _ = os.path.splitext(pdb_file_basename)
    pdb_id = str(pdb_id)
    pdb_id = pdb_id.upper()

    output_dir = os.path.join(output_dir, f"{pdb_id}_natural_ligands")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print ("Extracting all natural ligands for PDB ID", pdb_id,
            "containing at least", min_heavy_atoms, "atoms")
        print ("Outputting to", output_dir)

    filter_by_ligand_type = "NON-POLYMER" in LIGAND_TYPE_TO_ID

    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id, 
        pdb_filename)

    all_ligand_data = []

    for model in structure:

        model_id = model.id

        for chain in model:

            chain_id = chain.id

            for residue in chain:
                hetero_flag, sequence_identifier, insertion_code = residue.id
            
                if not hetero_flag.startswith("H_"):
                    continue
                    
                ligand_id = residue.get_resname()

                if min_heavy_atoms is not None:
                    num_heavy_atoms = 0
                    for atom in residue:
                        if atom.element == "H":
                            continue
                        num_heavy_atoms += 1
                    if num_heavy_atoms < min_heavy_atoms:
                        continue
                
                # # skip ions
                # if ligand_id in {"SO4", "ACT"}:
                #     continue
                # # skip anions
                # if ligand_id in {"FLC"}:
                #     continue

                # skip cholesterol/testosterone/heme/glucose?
                # if ligand_id in {"CLR", "TES", "HEM", "GLC"}:
                #     continue

                if filter_by_ligand_type and ligand_id not in LIGAND_TYPE_TO_ID["NON-POLYMER"]:
                    continue # skip ligand

                # write ligand to file
                io = PDBIO()
                io.set_structure(residue)
                ligand_filename = os.path.join(
                    output_dir, 
                    f"{pdb_id}_{model_id}_{chain_id}_{ligand_id}_{sequence_identifier}.pdb")
                if verbose:
                    print ("writing ligand to", ligand_filename)
                io.save(ligand_filename)

                try:
                    center_x, center_y, center_z = identify_centre_of_mass(
                        mol_path=ligand_filename, 
                        geometric=True,
                        precision=3,
                        verbose=verbose)
                except Exception as e:
                    center_x, center_y, center_z = None, None, None 

                size_x, size_y, size_z = get_bounding_box_size(
                    ligand_filename, 
                    allowance=None,
                    verbose=verbose,
                    )

                # determine location of atoms 
                atom_centers = []
                for atom in residue.get_atoms():
                    atom_centers.append(atom.get_coord().tolist())
               
                ligand_data = {
                    "model": model_id,
                    "chain": chain_id,
                    "ligand_id": ligand_id,
                    "sequence_identifier": sequence_identifier,
                    "center_x": center_x,
                    "center_y": center_y,
                    "center_z": center_z,
                    "size_x": size_x,
                    "size_y": size_y,
                    "size_z": size_z,
                    "volume": size_x * size_y * size_z,
                    "atom_centers": atom_centers,
                    "num_atoms": len(atom_centers),
                }

                if compress_ligands:
                    # compress ligand_pdb files
                    compressed_ligand_filename = gzip_file(
                        input_filename=ligand_filename,
                        delete_original_file=False,
                        verbose=verbose,
                    )
                    ligand_data["pdb_filename"] = ligand_filename
                    ligand_data["compressed_pdb_filename"] = compressed_ligand_filename

                all_ligand_data.append(ligand_data)


    if delete_output_dir: # delete all created PDB files 
        delete_directory(output_dir, verbose=verbose)

    return all_ligand_data

def read_header_from_pdb_file(
    pdb_id: str = None,
    pdb_filename: str = None,
    output_dir: str = "pdb_files",
    cleanup: bool = True,
    verbose: bool = False, 
    ):
    """Extract the header from a PDB file.

    Parameters
    ----------
    pdb_id : str, optional
        The PDB ID of the file, by default None
    pdb_filename : str, optional
        The filepath of the PDB structure, by default None
    output_dir : str, optional
        The directory to download to, by default "pdb_files"
    cleanup : bool, optional
        Flag to delete any downloaded files after running, by default True
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    dict
        The headers from the file
    """

    if pdb_filename is None:
        assert pdb_id is not None 
        # download PDB file
        pdb_filename = download_pdb_file_using_biopython(
            pdb_id,
            download_dir=output_dir)
    if pdb_id is None:
        pdb_file_basename = os.path.basename(pdb_filename)
        pdb_id, _ = os.path.splitext(pdb_file_basename)
    pdb_id = str(pdb_id)
    pdb_id = pdb_id.upper()

    output_dir = os.path.join(output_dir, pdb_id)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print ("Extracting header for PDB ID", pdb_id)
        print ("Outputting to directory", output_dir)

    pdb_header = parse_pdb_header(pdb_filename)

    if cleanup:
        delete_directory(output_dir, verbose=verbose)

    # cleanup header
    # remove keys
    for key in (
        "idcode", 
        "structure_reference", 
        "missing_residues",
        "journal_reference",
        "author",
        "keywords",
        "journal",
        ):
        if key in pdb_header:
            del pdb_header[key]

    # convert some values to list
    for key in ("compound", "source"):
        if key in pdb_header:
            pdb_header[key] = [ 
                pdb_header[key][k]
                for k in pdb_header[key]
            ] 

    return pdb_header

def load_scoria_molecule(
    mol_path: str,
    verbose: bool = False,
    ):
    """Use the `scoria` library to build a molecule model from the 3D structure file.

    Parameters
    ----------
    mol_path : str
        The path to the 3D file. Will be converted to PDB format, if it is not already.
    verbose : bool, optional
        Flag to print updates to a console, by default False

    Returns
    -------
    scoria.Molecule
        A `scoria` molecule object constructed from the given file
    """


    if verbose:
        print ("Building scoria molecule from", mol_path)
    
    # convert file if necessary
    if not mol_path.endswith(".pdb"):
        stem, ext = os.path.splitext(mol_path)
        ext = ext.replace(".", "") # remove .
        mol_path = obabel_convert(
            input_format=ext,
            input_filename=mol_path,
            output_format="pdb",
            output_filename=stem + ".pdb")

    return scoria.Molecule(mol_path)
    
def define_target_binding_site_using_biopython(
    pdb_filename: str,
    scale: float = 1.0,
    precision: int = 3,
    verbose: bool = False,
    ):
    """Use the PDB file of a ligand to define a bounding box for docking

    Parameters
    ----------
    ligand_pdb_filename : str
        The filename to base the bounding box on
    precision : int, optional
        Rounding precision, by default 3
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    dict
        The location and size of the computed bounding box
    """

    if verbose:
        print ("Determinning bounding box location and size using PDB file:", pdb_filename)

    # load structure
    parser = PDBParser()
    structure = parser.get_structure(
        "my-target",
        pdb_filename
    )
    
    # determine atom locations 
    atom_locations = np.array([
        a.coord 
        for a in structure.get_atoms()
    ])

    # geometric mean
    center_x, center_y, center_z = atom_locations.mean(axis=0)

    # bounding box
    max_atom_locations = atom_locations.max(axis=0)
    min_atom_locations = atom_locations.min(axis=0)
    size_x, size_y, size_z = np.ceil(np.abs(min_atom_locations - max_atom_locations) * scale)

    bounding_box = {
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }

    if isinstance(precision, int):
        bounding_box = {
            k: round(v, precision)
            for k, v in bounding_box.items()
        }

    return bounding_box

def define_target_binding_site_using_scoria(
    pdb_filename: str,
    precision: int = 3,
    verbose: bool = False,
    ):
    """Use the PDB file of a ligand to define a bounding box for docking

    Parameters
    ----------
    ligand_pdb_filename : str
        The filename to base the bounding box on
    precision : int, optional
        Rounding precision, by default 3
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    dict
        The location and size of the computed bounding box
    """

    if verbose:
        print ("Determinning bounding box location and size using PDB file:", pdb_filename)

    # create a scoria mol object from the ligand pdb file
    mol = load_scoria_molecule(
        mol_path=pdb_filename,
        verbose=verbose)

    # determine location using geometric mean
    center_x, center_y, center_z = identify_centre_of_mass(
        mol_path=pdb_filename,
        mol=mol, 
        geometric=True,
        precision=precision,
        verbose=verbose,
        )

    # determine bounding box size
    size_x, size_y, size_z = get_bounding_box_size(mol=mol, verbose=verbose)

    bounding_box = {
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }

    return bounding_box


def identify_centre_of_mass(
    mol_path = None,
    mol: scoria.Molecule = None, 
    precision = 3,
    geometric: bool = True,
    verbose: bool = False,
    ):
    """Use scoria to compute the center of mass / geometric center for a PDB file or pre-constructed scoria Molecule.

    Parameters
    ----------
    mol_path : str, optional
        Path of 3D molecule file, by default None
    mol : scoria.Molecule, optional
        Pre-constructed scoria Molecule, by default None
    precision : int, optional
        Rounding precision, by default 3
    geometric : bool, optional
        Flag to compute geometric mean, rather than center of mass, by default True
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    tuple
        Computed x, y, z co-ordinates of the center
    """

    if verbose:
        print ("Using scoria to compute center of molecule")
    
    if mol is None: # construct molecule
        assert mol_path is not None and isinstance(mol_path, str)
        mol = load_scoria_molecule(mol_path)
        
    if not geometric:
        if verbose:
            print ("Calculating mass center")
        try:
            center_x, center_y, center_z = mol.get_center_of_mass().data
        except Exception as e:
            print ("Mass center failed")
            print ("Exception was", e)
            geometric = True

    if geometric:
        if verbose:
            print ("Calculating geometric center",)
        try:
            center_x, center_y, center_z = mol.get_geometric_center()
        except Exception as e:
            print ("geometric center calculation error", e)
            return None

    # round using precision
    if isinstance(precision, int):
        center_x = round(center_x, precision)
        center_y = round(center_y, precision)
        center_z = round(center_z, precision)

    if verbose:
        print ("Determined center as mass:", center_x, center_y, center_z)

    return center_x, center_y, center_z

def get_bounding_box_size(
    mol, 
    scale: float = 1., 
    allowance: float = 3.,
    min_dimension_size: float = None,
    verbose: bool = False,
    ): 
    """Use `scoria` to compute the size of the bounding box required to contain the molecule.
    Optionally, scale the box by `scale`, and add `allowance` to all cartesian co-ordinates.

    Parameters
    ----------
    mol : scoria.Molecule / str
        A pre-contructed scoria Molecule, or a path to a 3D structure file.
    scale : float, optional
        Scalar to apply to all dimensions, by default 1.
    allowance : float, optional
        Extra Angstroms to add to all dimensions, by default 3.
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    np.ndarray
        A numpy array of shape (3,) containing the x, y, z bounding box sizes 
    """
    if not isinstance(mol, scoria.Molecule): # pdb file input
        mol = load_scoria_molecule(mol, verbose=verbose)
    if verbose:
        print ("determining bounding box of molecule")
    bounding_box = mol.get_bounding_box()
    box_size = np.ceil(np.abs(bounding_box[0] - bounding_box[1]) * scale)
    if allowance is not None:
        bounding_box += allowance 
    if min_dimension_size is not None:
        box_size = np.maximum(box_size, min_dimension_size)
    if verbose:
        print ("Determined bounding box size:", *box_size)
    return box_size

# read PDB file utils
def read_pdb_file_with_awk(
    awk_queries,
    return_if_missing: object = None,
    param_type: object = float,
    verbose: bool = False,
    ):
    """Use AWK to parse a PDB file.
    TODO: improve with a package.

    Parameters
    ----------
    awk_queries : list/str
        List of AWK queries
    return_if_missing : object, optional
        The value to return if the provided queries fail, by default None
    param_type : object, optional
        The type of the returned value, by default float
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    object
        A value of type `param_type` if a query is successful, else `return_if_missing`
    """
    if isinstance(awk_queries, str):
        awk_queries = [awk_queries]
    if verbose:
        print ("Executing awk queries", awk_queries)
    for awk_query in awk_queries:
        awk_string = subprocess.check_output(awk_query, shell=True).decode("utf-8")
        matching_lines = awk_string.split("\n")
        for matching_line in matching_lines:
            try:
                return param_type(matching_line.split()[-1])
            except:
                pass 
    return return_if_missing

def get_is_mutant_from_pdb_file(
    pdb_filename: str,
    is_mutant_if_missing: bool = False,
    verbose: bool = False,
    ):
    """Read a PDB file and determine if it is a mutant.

    Parameters
    ----------
    pdb_filename : str
        The PDB file to read
    is_mutant_if_missing : bool, optional
        Default is_mutant value if not specified, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    bool
        Whether or not the file is a mutant
    """
    if verbose:
        print ("Determining if file", pdb_filename, "is a mutant")
    try:
        is_mutant = subprocess.check_output(f"awk '/MUTATION:/' {pdb_filename}", shell=True).decode("utf-8")
        is_mutant = bool(is_mutant.split()[-1] in {"YES", "YES;"})
    except Exception as e:
        is_mutant = is_mutant_if_missing
    return is_mutant

def get_is_engineered_from_pdb_file(
    pdb_filename: str,
    is_engineered_if_missing: bool = False,
    verbose: bool = False,
    ):
    """Read a PDB file and determine if it is a engineered.

    Parameters
    ----------
    pdb_filename : str
        The PDB file to read
    is_engineered_if_missing : bool, optional
        Default is_engineered value if not specified, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    bool
        Whether or not the file is engineered
    """
    if verbose:
        print ("Determining if file", pdb_filename, "is engineered")
    try:
        is_engineered = subprocess.check_output(f"awk '/ENGINEERED:/' {pdb_filename}", shell=True).decode("utf-8")
        is_engineered = bool(is_engineered.split()[-1] in {"YES", "YES;"})
    except Exception as e:
        is_engineered = is_engineered_if_missing
    return is_engineered

def get_r_value_observed_from_pdb_file(
    pdb_filename: str,
    r_value_observed_if_missing: float = float("inf"),
    verbose: bool = False,
    ):
    """Read a PDB file and determine the observed r-value.

    Parameters
    ----------
    pdb_filename : str
        The PDB file to read
    r_value_observed_if_missing : bool, optional
        Default r_value_observed value if not specified, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    float
        The observed r-value
    """

    if verbose:
        print ("Determining observed r-value for PDB file", pdb_filename)

    awk_queries = [
        f"awk '!/BIN/&&/R VALUE/&&/WORKING/&&/TEST/' {pdb_filename}",
        f"awk '!/BIN/&&/R VALUE/&&/\(WORKING SET\)/' {pdb_filename}"
    ]

    r_value_observed = read_pdb_file_with_awk(
        awk_queries,
        return_if_missing=r_value_observed_if_missing,
        param_type=float,
        verbose=verbose,
    )
    
    return r_value_observed

def get_r_value_free_from_pdb_file(
    pdb_filename: str,
    r_value_free_if_missing: float = float("inf"),
    verbose: bool = False,
    ):
    """Read a PDB file and determine the free r-value.

    Parameters
    ----------
    pdb_filename : str
        The PDB file to read
    r_value_free_if_missing : bool, optional
        Default r_value_free value if not specified, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    float
        The free r-value
    """

    if verbose:
        print ("Determining free r-value for PDB file", pdb_filename)

    awk_queries = [
        f"awk '!/BIN/&&/FREE R VALUE                     :/' {pdb_filename}",
    ]

    r_value_free = read_pdb_file_with_awk(
        awk_queries,
        return_if_missing=r_value_free_if_missing,
        param_type=float,
    )
    return r_value_free

def get_completeness_from_pdb_file(
    pdb_filename: str,
    completeness_if_missing: float = 0.,
    verbose: bool = False,
    ):
    """Read a PDB file and determine the completeness.

    Parameters
    ----------
    pdb_filename : str
        The PDB file to read
    completeness_if_missing : bool, optional
        Default completeness value if not specified, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    float
        The completeness of the PDB file
    """

    if verbose:
        print ("Determining completeness of PDB file", pdb_filename)

    awk_queries = [
        f"awk '!/SHELL/&&!/BIN/&&/COMPLETENESS/' {pdb_filename}",
    ]
    completeness = read_pdb_file_with_awk(
        awk_queries,
        return_if_missing=completeness_if_missing,
        param_type=float,
    )
    return completeness

def get_resolution_from_pdb_file(
    pdb_filename: str,
    resolution_if_missing: float = float("inf"),
    verbose: bool = False,
    ):
    """Read a PDB file and determine the resolution.

    Parameters
    ----------
    pdb_filename : str
        The PDB file to read
    resolution_if_missing : bool, optional
        Default resolution value if not specified, by default False
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    float
        The resolution of the PDB file
    """

    if verbose:
        print ("Determining completeness of PDB file", pdb_filename)


    awk_queries = [
        f"awk '/RESOLUTION RANGE HIGH/' {pdb_filename}",
    ]
    resolution = read_pdb_file_with_awk(
        awk_queries,
        return_if_missing=resolution_if_missing,
        param_type=float,
    )
    return resolution

def get_cofactors_for_accessions(
    accessions: list,
    verbose: bool = True,
    ):
    if isinstance(accessions, str):
        accessions = [accessions]
    if verbose:
        print ("Getting co-factor IDs for accessions", accessions)

    return {
        accession: (
            ACCESSION_TO_COFACTOR[accession]
            if accession in ACCESSION_TO_COFACTOR
            else None # None value for no cofactors
        )
        for accession in accessions
    }

# def check_for_natural_ligand_in_pdb_file():
#     '''
#     TO HELP YAVUZ
#     '''

#     import pandas as pd 

#     positive_pairs_df = pd.read_csv("positive_pairs.csv", index_col=0)

#     pdb_id_to_ligand_symbol_filename = "pdb_id_to_ligand_id.json"

#     if os.path.exists(pdb_id_to_ligand_symbol_filename):
#         pdb_id_to_ligand_symbol = load_json(pdb_id_to_ligand_symbol_filename)
#     else:

#         pdb_id_to_ligand_symbol = {}

#         for _, row in positive_pairs_df.iterrows():

#             pdb_id = row["PDB entry"]
#             ligand_symbol = row["Ligand_symbol"]
#             if pd.isnull(pdb_id) or pd.isnull(ligand_symbol):
#                 continue

#             if pdb_id not in pdb_id_to_ligand_symbol:
#                 pdb_id_to_ligand_symbol[pdb_id] = set()

#             pdb_id_to_ligand_symbol[pdb_id].add(ligand_symbol)

#         # convert to list
#         pdb_id_to_ligand_symbol = {pdb_id: sorted(ligand_symbol) for pdb_id, ligand_symbol in pdb_id_to_ligand_symbol.items()}

#         write_json(pdb_id_to_ligand_symbol, pdb_id_to_ligand_symbol_filename)

#     import json
#     from utils.queries.targets_queries import get_natural_ligands_for_pdb_id_query

#     natural_ligands_from_db_dict_filename = "natural_ligands_from_db_dict.json"

#     # if os.path.exists(natural_ligands_from_db_dict_filename):
#     #     natural_ligands_from_db_dict = load_json(natural_ligands_from_db_dict_filename)
#     # else:

#     natural_ligands_from_db = get_natural_ligands_for_pdb_id_query(pdb_id_to_ligand_symbol)

#     natural_ligands_from_db_dict = {}
#     for record in natural_ligands_from_db:
        
#         natural_ligands_for_pdb_id = record["natural_ligands"] 
#         if natural_ligands_for_pdb_id is None:
#             continue

#         pdb_id = record["pdb_id"]
#         if pdb_id not in natural_ligands_from_db_dict:
#             natural_ligands_from_db_dict[pdb_id] = {}
        
#         natural_ligands_for_pdb_id = json.loads(natural_ligands_for_pdb_id)
#         for natural_ligand_for_pdb_id in natural_ligands_for_pdb_id:
#             ligand_symbol = natural_ligand_for_pdb_id["ligand"]
#             natural_ligands_from_db_dict[pdb_id][ligand_symbol] = natural_ligand_for_pdb_id



#     pdb_ids_not_in_db = {
#         pdb_id
#         for pdb_id in pdb_id_to_ligand_symbol
#         if pdb_id not in natural_ligands_from_db_dict}

#     pdb_ids_not_in_db = {
#         pdb_id: get_pdb_header_and_natural_ligands_for_one_pdb_id(pdb_id)
#         for i, pdb_id in enumerate(pdb_ids_not_in_db)
#     }

#     # add new rows to DB
#     add_to_pdb_table(pdb_ids_not_in_db)

#     print ("UPDATING DICT", len(natural_ligands_from_db_dict))
#     for pdb_id, data in pdb_ids_not_in_db.items():
#         if data is None:
#             continue
#         natural_ligands_for_pdb_id = data["natural_ligands"]
#         if natural_ligands_for_pdb_id is None:
#             continue
#         if pdb_id not in natural_ligands_from_db_dict:
#             natural_ligands_from_db_dict[pdb_id] = {}
#         for natural_ligand_for_pdb_id in natural_ligands_for_pdb_id:
#             ligand_symbol = natural_ligand_for_pdb_id["ligand"]
#             natural_ligands_from_db_dict[pdb_id][ligand_symbol] = natural_ligand_for_pdb_id
    
    
#     write_json(natural_ligands_from_db_dict, natural_ligands_from_db_dict_filename)
    

#     # update df
#     def contains_ligand_symbol(row):
#         pdb_id = row["PDB entry"]
#         ligand_symbol = row["Ligand_symbol"]

#         if pd.isnull(pdb_id) or pd.isnull(ligand_symbol):
#             return False

#         if pdb_id not in natural_ligands_from_db_dict:
#             return False

#         return ligand_symbol in natural_ligands_from_db_dict[pdb_id]


#     positive_pairs_df["contains_ligand_symbol"] = positive_pairs_df.apply(contains_ligand_symbol, axis=1)

#     positive_pairs_df.to_csv("positive_pairs_with_contains_ligand_symbol.csv")

#     print (positive_pairs_df.shape)
#     positive_pairs_df = positive_pairs_df.loc[positive_pairs_df["contains_ligand_symbol"]]
#     positive_pairs_df.to_csv("positive_pairs_filtered.csv")
#     print (positive_pairs_df.shape)

def download_all_pdbs(
    output_dir="pdbs",
    verbose: bool = True,
    ):

    if verbose:
        print ("Downloading all PDBs to directory", output_dir)

    pdb_list = PDBList(pdb=output_dir)

    #TODO



if __name__ == "__main__":


    download_all_pdbs()


    # print (get_cofactors_for_accessions(["P09874", "Q65WW7"]))

    # all_chains = get_all_chain_ids_in_a_PDB_file(
    #     "6nna",
    #     None,
    # )
    # print (all_chains)

    # select_chains_from_pdb_file(
    #     pdb_id="6nna",
    #     pdb_filename="6NNA.pdb",
    #     output_filename="6NNA_B.pdb",
    #     chain_ids=["B"],
    #     verbose=True,
    # )

    # print (get_bounding_box_size("dock_test/pockets/P49327/6NNA/P49327_6NNA_A_KUA_2202_4.pdb"))

    # from utils.molecules.pymol_utils import convert_file_with_pymol
    # from ai_blind_docking.algorithms.pose_ranking.pose_ranking_utils import convert_pocket_pdb_file_to_poc_format

    # from concurrent.futures import ProcessPoolExecutor, as_completed

    # with ProcessPoolExecutor(10) as p:

    #     running_tasks = []

        
    #     # make pocket_library for scPDB

    #     root_output_dir = "/shared/pocket_library_new"
    #     molecule_mol2_output_dir = os.path.join(root_output_dir, "MOL2")

    #     os.makedirs(molecule_mol2_output_dir, exist_ok=True)

    #     scpdb_data_dir = "/home/david/Downloads/scPDB"


    #     all_targets = sorted(os.listdir(scpdb_data_dir))

    #     for target in all_targets:
    #         target_data_dir = os.path.join(scpdb_data_dir, target)

    #         target_molecule_mol2_filename_in_target_data_dir = os.path.join(target_data_dir, "ligand.mol2")
    #         assert os.path.exists(target_molecule_mol2_filename_in_target_data_dir)

    #         target_molecule_pdb_filename_in_target_data_dir = os.path.join(target_data_dir, "ligand.pdb")
    #         if not os.path.exists(target_molecule_pdb_filename_in_target_data_dir):
    #             convert_file_with_pymol(
    #                 target_molecule_mol2_filename_in_target_data_dir,
    #                 target_molecule_pdb_filename_in_target_data_dir,
    #             )

    #         target_mol2_filename_in_target_data_dir = os.path.join(target_data_dir, "protein.mol2")
    #         assert os.path.exists(target_mol2_filename_in_target_data_dir)

    #         target_pdb_filename_in_target_data_dir = os.path.join(target_data_dir, "protein.pdb")
    #         # create as required
    #         if not os.path.exists(target_pdb_filename_in_target_data_dir):
    #             # target_pdb_filename_in_target_data_dir = obabel_convert(
    #             #     input_format="mol2",
    #             #     input_filename=target_mol2_filename_in_target_data_dir,
    #             #     output_format="pdb",
    #             #     output_filename=target_pdb_filename_in_target_data_dir,
    #             #     verbose=True,
    #             # )
    #             convert_file_with_pymol(
    #                 target_mol2_filename_in_target_data_dir,
    #                 target_pdb_filename_in_target_data_dir,
    #             )


    #         # copy ligand structure into mol2 dir
    #         target_molecule_mol2_filename_in_mol2_dir = os.path.join(molecule_mol2_output_dir, f"{target}.mol2")
    #         if not os.path.exists(target_molecule_mol2_filename_in_mol2_dir):
    #             copy_file(target_molecule_mol2_filename_in_target_data_dir, target_molecule_mol2_filename_in_mol2_dir)


    #         for radius in (
    #             4, 
    #             6, 
    #             7,
    #             8,
    #         ):

    #             radius_output_dir = os.path.join(root_output_dir, "POC")
    #             if radius > 4:
    #                 radius_output_dir += f"_{radius}A"
    #             os.makedirs(radius_output_dir, exist_ok=True)

    #             output_pocket_filename = os.path.join(radius_output_dir, f"{target}.poc")

    #             if os.path.exists(output_pocket_filename):
    #                 continue

    #             output_pdb_filename = os.path.join(radius_output_dir, f"{target}.pdb")
                

    #             # build model of pocket in pdb format
    #             running_tasks.append(
    #                 p.submit(
    #                     build_binding_site_based_on_distance_to_natural_ligand,
    #                     pdb_id=target,
    #                     pdb_filename=target_pdb_filename_in_target_data_dir,
    #                     output_pocket_filename=output_pdb_filename,
    #                     output_chains_filename=None,
    #                     natural_ligand_pdb_filename=target_molecule_pdb_filename_in_target_data_dir,
    #                     radius=radius,
    #                     verbose=True,
    #                 )
    #             )

    #     for task in as_completed(running_tasks):
    #         pocket_pdb_filename = task.result()
    #         pocket_poc_filename = pocket_pdb_filename.replace(".pdb", ".poc")

    #         # convert to POC
    #         pocket_poc_filename = convert_pocket_pdb_file_to_poc_format(
    #             pocket_pdb_filename=pocket_pdb_filename,
    #             output_poc_filename=pocket_poc_filename,
    #             verbose=True,
    #         )

    #         delete_file(pocket_pdb_filename, verbose=True,)
    #         # rename to have pdb extension
    #         os.rename(pocket_poc_filename, pocket_pdb_filename)
