
import os, re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir,
))

if __name__ == "__main__":

    import sys
    sys.path.insert(1, PROJECT_ROOT)

import math

import numpy as np
import pandas as pd

from io import BytesIO

import sys
import logging

import rdkit
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem, SimDivFilters
from rdkit.Chem import ChemicalFeatures, AllChem, PandasTools
from rdkit.Chem.Descriptors import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.Crippen import *
from rdkit.Chem.Lipinski import *

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl  # type: ignore
delete_color = mpl.colors.to_rgb("#F06060")
modify_color = mpl.colors.to_rgb("#1BBC9B")
import matplotlib.pyplot as plt

from rdkit.Chem import rdFMCS as MCS  # type: ignore
from rdkit.Chem.Draw import MolToImage

from rdkit.Chem.AllChem import GenerateDepictionMatching2DStructure, Compute2DCoords

from rdkit.Chem.BRICS import BRICSDecompose

from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

# filtering 
from rdkit.Chem.FilterCatalog import *

from utils.io.io_utils import read_smiles, write_json, load_json

BASE_FEATURES_LOCATION  = os.path.join(PROJECT_ROOT, "data", "rdkit", "BaseFeatures.fdef")
if os.path.exists(BASE_FEATURES_LOCATION):
    FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(BASE_FEATURES_LOCATION)
else:
    FEATURE_FACTORY = None

log = logging.getLogger(__name__)

LIPINSKI_THRESHOLDS = {
    "molecular_weight": {
        "lte": 500,
    },
    "log_p": {
        "lte": 5,
    },
    "num_hydrogen_donors": {
        "lte": 5,
    },
    "num_hydrogen_acceptors": {
        "lte": 10,
    },
}

GHOSE_THRESHOLDS = {
    "molecular_weight": {
        "gte": 160, "lte": 480,
    },
    "log_p": {
        "gte": -0.4, "lte": 5.6,
    },
    "num_atoms": {
        "gte": 20, "lte": 70,
    },
    "molar_refractivity": {
        "gte": 40, "lte": 130,
    },
}

VERBER_THRESHOLDS = {
    "num_rotatable_bonds": {
        "lte": 10,
    }, 
    "polar_surface_area": {
        "lte": 140,
    }
}

REOS_THRESHOLDS = {
    "molecular_weight": {
        "gte": 200, "lte": 500,
    },
    "log_p": {
        "gte": -5, "lte": 5,
    },
    "num_hydrogen_donors": {
        "gte": 0, "lte": 5,
    },
    "num_hydrogen_acceptors": {
        "gte": 0, "lte": 10,
    },
    "charge": {
        "gte": -2, "lte": 2,
    },
    "num_rotatable_bonds": {
        "gte": 0, "lte": 8,
    },
    "num_heavy_atoms": {
        "gte": 15, "lte": 50,
    },
}

RULE_OF_THREE_THRESHOLDS = {
    "molecular_weight": {
        "lte": 300,
    },
    "log_p": {
        "lte": 3,
    },
    "num_hydrogen_donors": {
        "lte": 3,
    },
    "num_hydrogen_acceptors": {
        "lte": 3,
    },
    "num_rotatable_bonds": {
        "lte": 3,
    }
}

DRUG_LIKE_THRESHOLDS = {
    "molecular_weight": {
        "lte": 400,
    },
    "number_of_rings": {
        "gte": 1, # > 0
    },
    "num_rotatable_bonds": {
        "lte": 4, # < 5
    },
    "num_hydrogen_donors": {
        "lte": 5,
    },
    "num_hydrogen_acceptors": {
        "lte": 10,
    },
    "log_p": {
        "lte": 5,
    },
}

def build_rdkit_mol(
    smiles: str,
    largest_fragment: bool = True,
    ):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not largest_fragment:
        return mol

    # determine largest fragment
    largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()

    return largest_fragment_chooser.choose(mol)


def MolSupplier(
    mol_filename: str = None, 
    molecule_structure_lines: str = None,
    sanitize: bool = True,
    ):

    if mol_filename is not None:
        with open(mol_filename, 'r') as f:
            molecule_structure_lines = [line for line in f.readlines()]

    if molecule_structure_lines is None:
        return []

    mol_start_indexes = [-1] + [index for index, p in enumerate(molecule_structure_lines) 
        if p.startswith("$$$$")] + [len(molecule_structure_lines)]

    for start, end in zip(mol_start_indexes, mol_start_indexes[1:]):
        block = "".join(molecule_structure_lines[start+1: end])
        m = Chem.MolFromMolBlock(block, sanitize=sanitize)
        yield m

def Mol2MolSupplier(
    mol2_filename: str = None, 
    molecule_structure_lines: str = None,
    sanitize: bool = True,
    ):
    # source is https://chem-workflows.com/articles/2020/03/23/building-a-multi-molecule-mol2-reader-for-rdkit-v2/
    # but going to rewrite 
#     mols=[]
    if mol2_filename is not None:
        with open(mol2_filename, 'r') as f:
            molecule_structure_lines=[line for line in f.readlines()]

#     start=[index for (index,p) in enumerate(doc) if '@<TRIPOS>MOLECULE' in p]
#     finish=[index-1 for (index,p) in enumerate(doc) if '@<TRIPOS>MOLECULE' in p]
#     finish.append(len(doc))
    if molecule_structure_lines is None:
        return []
    
    # use "in" instead of startswith to handle poorly formatted files?
    mol_start_indexes = [index for index, p in enumerate(molecule_structure_lines) if '@<TRIPOS>MOLECULE' in p] + [len(molecule_structure_lines)]

    for start, end in zip(mol_start_indexes, mol_start_indexes[1:]):
        block = "".join(molecule_structure_lines[start: end])
        m = Chem.MolFromMol2Block(block, sanitize=sanitize)
        yield m
#         mols.append(m)
#     return(mols)

def load_multiple_molecules_from_file(
    structure_filename: str,
    mol_format: str = None,
    verbose: bool = True,
    ):

    if mol_format is None:
        _, mol_format = os.path.splitext(structure_filename)
        # remove .
        mol_format = mol_format.replace(".", "")

    mol_format = mol_format.lower()

    if verbose:
        print ("Loading molecule(s) in format", mol_format, "from file", structure_filename)

    if not os.path.exists(structure_filename):
        print (structure_filename, "does not exist!")
        return []

    if mol_format == "mol":
        # return Chem.MolFromMolFile(structure_filename)
        # return generator
        return MolSupplier(structure_filename)
    elif mol_format == "mol2":
        # return Chem.MolFromMol2File(structure_filename)
        return Mol2MolSupplier(structure_filename)
    elif mol_format == "sdf":
        return Chem.ForwardSDMolSupplier(structure_filename)
    elif mol_format == "pdb":
        return [Chem.MolFromPDBFile(structure_filename)]
    # elif mol_format in {"smi", "txt"}: # TODO doesn't work right now
    #     return Chem.SmilesMolSupplier(structure_filename)
    else:
        print ("Format", mol_format, "is not implemented yet!")
        return []
        # raise NotImplementedError(mol_format)

def load_multiple_molecules_from_string(
    molecule_structure_string: str,
    mol_format: str, # must be given
    verbose: bool = True,
    ):

    mol_format = mol_format.lower()

    molecule_structure_lines = molecule_structure_string.splitlines(keepends=True)

    if verbose:
        print ("Loading molecule(s) in string in format", mol_format,)

    if molecule_structure_string is None:
        print ("molecule structure string is None ")
        return []

    if mol_format == "mol":
        # return Chem.MolFromMolFile(structure_filename)
        # return generator
        return MolSupplier(molecule_structure_lines=molecule_structure_lines)
    elif mol_format == "mol2":
        # return Chem.MolFromMol2File(structure_filename)
        return Mol2MolSupplier(molecule_structure_lines=molecule_structure_lines)
    elif mol_format == "sdf":
        return Chem.ForwardSDMolSupplier(BytesIO(molecule_structure_string.encode()))
    elif mol_format == "pdb":
        return [Chem.MolFromPDBBlock(molecule_structure_string)]
    # elif mol_format in {"smi", "txt"}: # TODO doesn't work right now
    #     return Chem.SmilesMolSupplier(structure_filename)
    else:
        print ("Format", mol_format, "is not implemented yet!")
        return []
        # raise NotImplementedError(mol_format)


def smiles_to_SDF_2D_rdkit(
    smiles_filename: str,
    sdf_filename: str = None,
    smiles_key: str = "smiles",
    molecule_identifier_key: str = "molecule_identifier",
    delimiter: str = "\t",
    molecule_col_name: str = "molecule",
    ):
    """Use RDKit to convert a smiles file into two-dimensional SDF format.

    Parameters
    ----------
    smiles_filename : str
        Smiles filename to convert
    sdf_filename : str, optional
        Output SDF filename, by default None
    smiles_key : str, optional
        Key to identify smiles, by default "smiles"
    molecule_identifier_key : str, optional
        Key to identify molecules, by default "molecule_identifier"
    delimiter : str, optional
        Delimiter used in the SMILES file, by default "\t"
    molecule_col_name : str, optional
        Key to store RDKit molecule in DataFrame, by default "molecule"

    Returns
    -------
    str
        Filename of created SDF file
    """

    if sdf_filename is None:
        stem, _ = os.path.splitext(smiles_filename)
        sdf_filename = stem + ".sdf"

    # read in smiles as dataframe
    mols_as_df = read_smiles(
        smiles_filename=smiles_filename,
        remove_invalid_molecules=True,
        clean_molecules_with_rdkit=False,
        return_series=False,
        return_list=False,
        molecule_identifier_key=molecule_identifier_key,
        smiles_key=smiles_key,
        delimiter=delimiter,
    )
    PandasTools.AddMoleculeColumnToFrame(mols_as_df, smilesCol=smiles_key, molCol=molecule_col_name)

    PandasTools.WriteSDF(mols_as_df, sdf_filename, molColName=molecule_col_name, properties=list(mols_as_df.columns))

    return sdf_filename

def smiles_to_SDF_3D_rdkit_single_molecule(
    smi: str,
    molecule_identifier: str,
    sdf_filename: str,
    add_hydrogen: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    ):
    """Convert a single molecule to 3D and save in SDF format to `sdf_filename`.

    Parameters
    ----------
    smi : str
        SMILES string of molecule
    molecule_identifier : str
        Name of molecule
    sdf_filename : str
        Filename to write to
    add_hydrogen : bool, optional
        Flag to make all hydrogen atoms explicit, by default True
    overwrite : bool, optional
        Flag to overwrite sdf_filename if it already exists, by default False

    Returns
    -------
    str
        Output SDF filename
    """
    if not sdf_filename.endswith(".sdf"):
        sdf_filename += ".sdf"
    if verbose:
        print ("Writing molecule to SDF file", sdf_filename)
    if not os.path.exists(sdf_filename) or overwrite:
        # mol = Chem.MolFromSmiles(smi)
        mol = build_rdkit_mol(smi)
        mol.SetProp("_Name", molecule_identifier)
        if add_hydrogen:
            if verbose:
                print ("Adding hydrogen to SDF file")
            mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,)
        with open(sdf_filename, "w") as f:
            f.writelines(Chem.MolToMolBlock(mol))
    return sdf_filename

def remove_atoms(mol, list_of_idx_to_remove):
    """
    This function removes atoms from an rdkit mol based on
    a provided list. The RemoveAtom function in Rdkit requires
    converting the mol to an more editable version of the rdkit mol
    object (Chem.EditableMol).

    Inputs:
    :param rdkit.Chem.rdchem.Mol mol: any rdkit mol
    :param list list_of_idx_to_remove: a list of idx values to remove
                                        from mol
    Returns:
    :returns: rdkit.Chem.rdchem.Mol new_mol: the rdkit mol as input but with
                                            the atoms from the list removed
    """

    if mol is None:
        return None

    try:
        atoms_to_remove = list_of_idx_to_remove
        atoms_to_remove.sort(reverse=True)
    except:
        return None

    try:
        em1 = Chem.EditableMol(mol)
        for atom in atoms_to_remove:
            em1.RemoveAtom(atom)

        new_mol = em1.GetMol()

        return new_mol
    except:
        return None

def BRICS_decompose_smiles(
    smi: str,
    min_fragment_size: int = 3,
    return_mols: bool = True,
    keep_non_leaf_nodes: bool = False,
    ):
    """Perform BRICS decomposition of a single molecule described by `smi`.

    Parameters
    ----------
    smi : str
        SMILES string describing molecule
    min_fragment_size : int, optional
        Minimum number of atoms in fragment, by default 3

    Returns
    -------
    generator
        Generator containing all fragments
    """
    # mol = Chem.MolFromSmiles(smi)
    mol = build_rdkit_mol(smi)
    if mol is None:
        return None
    return BRICSDecompose(
        mol,
        minFragmentSize=min_fragment_size,
        returnMols=return_mols,
        keepNonLeafNodes=keep_non_leaf_nodes,
    )

def BRICS_decompose_smiles_using_RDKit(
    smiles,
    keep_original_molecule_in_output: bool = True,
    min_fragment_size: int = 3,
    keep_non_leaf_nodes: bool = False,
    smiles_key: str ="smiles",
    molecule_identifier_key: str = "molecule_id",
    verbose: bool = True,
    ):
    """USe RDKit to BRICs decompose a list of molecules

    Parameters
    ----------
    smiles : list
        List of molecules
    keep_original_molecule_in_output : bool, optional
        Flag to keep each original molecule in the output set, by default True

    Returns
    -------
    list
        List of (id, smi) tuples
    """

    if verbose:
        print ("Perfoming BRICS decomposition of", len(smiles), "SMILES using min_fragment_size =", min_fragment_size)

    # expr = re.compile(r'\[[0-9]+\*\]')
    # empty_brackets = re.compile(r"\(\)")

    # store as dict for uniqueness
    decomposed_smiles = dict()

    if isinstance(smiles, dict):
        smiles = smiles.items()

    for molecule_id_smi in smiles:

        if isinstance(molecule_id_smi, tuple):
            molecule_id, smi = molecule_id_smi
        elif isinstance(molecule_id_smi, dict) and smiles_key in molecule_id_smi and molecule_identifier_key in molecule_id_smi:
            molecule_id = molecule_id_smi[molecule_identifier_key]
            smi = molecule_id_smi[smiles_key]

        else:
            continue

        if keep_original_molecule_in_output and smi not in decomposed_smiles:
            decomposed_smiles[smi] = molecule_id

        fragment_mols = BRICS_decompose_smiles(
            smi,
            min_fragment_size=min_fragment_size,
            keep_non_leaf_nodes=keep_non_leaf_nodes,
        )
        if fragment_mols is None:
            continue
        for i, fragment_mol in enumerate(fragment_mols, start=1):


            # remove atoms (better than regex change)
            # taken from fragmenter_of_smi_mol.py
            list_to_remove = []
            for atom in fragment_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    list_to_remove.append(atom.GetIdx())

            fragment_mol = remove_atoms(fragment_mol, list_to_remove)

            for atom in fragment_mol.GetAtoms():
                atom.SetIsotope(0)

            # convert to smiles
            fragment_smiles = Chem.MolToSmiles(fragment_mol, isomericSmiles=True)

            # some sanity check
            # fragment_mol = Chem.MolFromSmiles(fragment_smiles)
            fragment_mol = build_rdkit_mol(fragment_smiles)
            if fragment_mol is None:
                continue

            # remove asterix
            # fragment = expr.sub("", fragment)
            # remove any empty brackets
            # fragment = empty_brackets.sub("", fragment)

            if fragment_smiles not in decomposed_smiles:
                decomposed_smiles[fragment_smiles] = f"{molecule_id}_BRICS_fragment_{i}"

    decomposed_smiles = [
        {
            molecule_identifier_key: molecule_id,
            smiles_key: smi,
        }
        for smi, molecule_id in decomposed_smiles.items()
    ]

    return decomposed_smiles

def map_molecule_property_string_to_function(
    property_string: str,
    precision: int = 4,
    ):

    if property_string == "smiles": # isomeric?
        return lambda mol: Chem.MolToSmiles(mol, isomericSmiles=True)
    elif property_string == "inchi":
        return Chem.MolToInchi
    elif property_string == "inchikey":
        return Chem.MolToInchiKey
    elif property_string == "molecular_formula":
        return CalcMolFormula
    elif property_string == "molecular_weight":
        return lambda mol: round(MolWt(mol), precision)
    elif property_string == "heavy_atom_molecular_weight":
        return lambda mol: round(HeavyAtomMolWt(mol), precision)
    elif property_string == "bond_count":
        return lambda mol: mol.GetNumBonds()
    elif property_string == "number_of_rings":
        return CalcNumRings
    elif property_string == "num_radical_electons":
        return NumRadicalElectrons
    elif property_string == "num_valence_electons":
        return NumValenceElectrons
    elif property_string == "num_hetero_atoms":
        return CalcNumHeteroatoms
    elif property_string == "num_hetero_cycles":
        return CalcNumHeterocycles
    elif property_string == "log_p":
        return lambda mol: round(MolLogP(mol), precision)
    elif property_string == "molar_refractivity":
        return lambda mol: round(MolMR(mol), precision)
    elif property_string == "num_rotatable_bonds":
        return CalcNumRotatableBonds
    elif property_string == "num_hydrogen_acceptors":
        return NumHAcceptors
    elif property_string == "num_hydrogen_donors":
        return NumHDonors
    elif property_string == "num_atoms":
        return Chem.rdchem.Mol.GetNumAtoms
    elif property_string == "num_heavy_atoms":
        return HeavyAtomCount        
    elif property_string == "polar_surface_area":
        return lambda mol: round(TPSA(mol), precision)
    elif property_string == "num_aromatic_rings":
        return CalcNumAromaticRings
    elif property_string == "charge":
        return Chem.GetFormalCharge
    elif property_string == "scaffold_smiles":
        return lambda mol: Chem.MolToSmiles(GetScaffoldForMol(mol))
    else:
        return None # property not implemented

def compute_molecule_properties(
    smi: str = None,
    mol: rdkit.Chem.rdchem.Mol = None,
    properties: list = [ 
        "molecular_formula",
        "molecular_weight",
        "log_p",
        "num_hydrogen_acceptors",
        "num_hydrogen_donors",
        "num_rotatable_bonds",
        "num_atoms",
        "num_heavy_atoms",
        "polar_surface_area",
        "num_aromatic_rings",
        "bond_count",
        "number_of_rings",
        "molar_refractivity",
        "charge",
        "scaffold_smiles",
    ],
    precision: int = 4,
    largest_fragment: bool = True,
    verbose: bool = False,
    ):
    """Compute a set molecule properties from the molecule described either by `smi` or `mol`.

    Parameters
    ----------
    smi : str, optional
        SMILES string of molecule, by default None
    mol : rdkit.Chem.rdchem.Mol, optional
        Preinitialised RDkit molecule instance, by default None
    properties : dict, optional
        Dictionary mapping property names to functions to compute them,
        by default { "inchikey": Chem.MolToInchiKey, "molecular_formula": CalcMolFormula,
            "molecular_weight": lambda mol: round(MolWt(mol), 3), "heavy_atom_molecular_weight": lambda mol: round(HeavyAtomMolWt(mol), 3),
            "bond_count": lambda mol: mol.GetNumBonds(), "number_of_rings": CalcNumRings, "num_radical_electons": NumRadicalElectrons,
            "num_valence_electons": NumValenceElectrons, "num_hetero_atoms": CalcNumHeteroatoms, "num_hetero_cycles": CalcNumHeterocycles, }

    Returns
    -------
    dict
        Mapping from property name to property value for the submitted molecule
    """

    # convert properties to dict
    properties_dict = {}
    for property_string in properties:
        property_function = map_molecule_property_string_to_function(property_string, precision=precision)
        if property_function is None:
            continue
        properties_dict[property_string] = property_function

    if verbose:
        print ("Obtaining molecular properties of a single molecule")

    if mol is None:
        if smi is not None:
            # mol = Chem.MolFromSmiles(smi)
            mol = build_rdkit_mol(smi, largest_fragment=largest_fragment)

    if mol is None: # mol failed to generate for whatever reason
        return {
            property_name: None
            for property_name in properties_dict
        }
    
    return {
        property_name: property_function(mol)
        for property_name, property_function in properties_dict.items()
    }

def diversity_molecule_selection(
    smiles, # list of smiles
    sample_size: int,
    radius: int = 3,
    seed: int = 0,
    verbose: bool = False,
    ):
    """Use fingerprints to select `sample_size` diverse molecules from the list or set of molecules `smiles`.

    Parameters
    ----------
    smiles : list/set
        List or set of molecule SMILES
    radius : int, optional
        Radius of morgan fingerprints to use for diversity selection, by default 3
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    generator
        Iterable containing indices of selected molecules

    """

    assert isinstance(smiles, list) or isinstance(smiles, set)

    n_smiles = len(smiles)
    if sample_size >= n_smiles:
        # not enough molecules
        return range(n_smiles)

    if verbose:
        print ("Selecting", sample_size, "diverse molecule(s) from a set of", n_smiles, "molecule(s) using seed", seed)

    fps = []
    for smi in smiles:
        # mol = Chem.MolFromSmiles(smi)
        mol = build_rdkit_mol(smi)
        if mol is None: # TODO
            raise Exception("INVALID MOLECULE IN DIVERSITY SELECTION")
        fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, ))

    assert len(fps) == n_smiles

    mmp = SimDivFilters.MaxMinPicker()
    return mmp.LazyBitVectorPick(fps, poolSize=n_smiles, pickSize=sample_size, seed=seed)

# def build_2D_gobbi_pharmacophore_fingerprints(
#     smiles,
#     # min_points=2,
#     # max_points=3,
#     # bins=[(0,2),(2,5),(5,8)],
#     ):
#     # signature_factory = SigFactory(FEATURE_FACTORY, minPointCount=min_points, maxPointCount=max_points)
#     # signature_factory.SetBins(bins)
#     # signature_factory.Init()

#     return sp.csr_matrix([
#         Generate.Gen2DFingerprint(Chem.MolFromSmiles(smi), Gobbi_Pharm2D.factory)
#             for smi in smiles
#     ], dtype=bool)

def smi_to_inchikey(
    smi: str,
    ):
    """Convert SMILES string to inchikey using RDKit.

    Parameters
    ----------
    smi : str
        SMILES string

    Returns
    -------
    str
        INCHIKEY of molecule
    """
    # mol = Chem.MolFromSmiles(smi)
    mol = build_rdkit_mol(smi)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)

def LoadSDF_in_chunks(
    sdf_filename,
    idName='ID',
    mol_col_name='ROMol',
    includeFingerprints=False,
    isomericSmiles=True,
    smiles_name=None,
    embedProps=False,
    removeHs=True,
    strictParsing=True,
    chunk_size=1000,
    ):
    '''Read file in SDF format and return as Pandas data frame.
    If embedProps=True all properties also get embedded in Mol objects in the molecule column.
    If molColName=None molecules would not be present in resulting DataFrame (only properties
    would be read).
    '''
    if isinstance(sdf_filename, str):
        if sdf_filename.lower()[-3:] == ".gz":
            import gzip
            f = gzip.open(sdf_filename, "rb")
        else:
            f = open(sdf_filename, 'rb')
            close = f.close
    else:
        f = sdf_filename
        close = None  # don't close an open file that was passed in

    records = []
    indices = []
    for i, mol in enumerate(
        Chem.ForwardSDMolSupplier(f, sanitize=(mol_col_name is not None), removeHs=removeHs,
                                strictParsing=strictParsing)):
        if mol is None:
            continue
        try:
            row = dict((k, mol.GetProp(k)) for k in mol.GetPropNames())
        except UnicodeDecodeError:
            continue
        if mol_col_name is not None and not embedProps:
            for prop in mol.GetPropNames():
                mol.ClearProp(prop)
        if mol.HasProp('_Name'):
            row[idName] = mol.GetProp('_Name')
        if smiles_name is not None:
            try:
                row[smiles_name] = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
            except:
                log.warning('No valid smiles could be generated for molecule %s', i)
                row[smiles_name] = None
        if mol_col_name is not None and not includeFingerprints:
            row[mol_col_name] = mol
        # elif molColName is not None:
        #     row[molColName] = _MolPlusFingerprint(mol)
        records.append(row)
        indices.append(i)

        if len(records) == chunk_size:
            yield pd.DataFrame(records, index=indices)
            records = []
            indices = []

    if close is not None:
        close()

    if len(records) > 0:
        assert len(records) < chunk_size
        yield pd.DataFrame(records, index=indices)

def write_molecules_to_sdf_file(
    all_molecules: list,
    output_sdf_filename: str,
    return_id_to_molecule: bool = True,
    verbose: bool = False,
    ):

    if not output_sdf_filename.endswith(".sdf"):
        output_sdf_filename += ".sdf"

    if verbose:
        print ("Writing molecules to SDF file", output_sdf_filename)
        if return_id_to_molecule:
            print ("Returning id_to_molecule")

    id_to_molecule = {}

    writer = Chem.SDWriter(output_sdf_filename)
    for i, m in enumerate(all_molecules):
        id_to_molecule[i] = m.GetProp('_Name')
        writer.write(m)
    writer.close()

    if return_id_to_molecule:
        return id_to_molecule, output_sdf_filename

    return output_sdf_filename

# All below are for drawing counterfactuals

# from multiprocessing import Process, Manager

def moldiff(template, query, mcs_timeout=10):
    """Compare the two rdkit molecules.

    :param template: template molecule
    :param query: query molecule
    :return: list of modified atoms in query, list of modified bonds in query
    """
    r = MCS.FindMCS([template, query], timeout=mcs_timeout, verbose=True)

    substructure = rdkit.Chem.MolFromSmarts(r.smartsString)
    raw_match = query.GetSubstructMatches(substructure)
    template_match = template.GetSubstructMatches(substructure)
    # flatten it
    match = list(raw_match[0])
    template_match = list(template_match[0])

    # need to invert match to get diffs
    inv_match = [i for i in range(query.GetNumAtoms()) if i not in match]

    # get bonds
    bond_match = []
    for b in query.GetBonds():
        if b.GetBeginAtomIdx() in inv_match or b.GetEndAtomIdx() in inv_match:
            bond_match.append(b.GetIdx())

    # now get bonding changes from deletion

    def neigh_hash(a):
        return "".join(sorted([n.GetSymbol() for n in a.GetNeighbors()]))

    for ti, qi in zip(template_match, match):
        if neigh_hash(template.GetAtomWithIdx(ti)) != neigh_hash(
            query.GetAtomWithIdx(qi)
        ):
            inv_match.append(qi)

    return inv_match, bond_match


def moldiff_process(template, query, return_dict):
    """Compare the two rdkit molecules.

    :param template: template molecule
    :param query: query molecule
    :return: list of modified atoms in query, list of modified bonds in query
    """
    r = MCS.FindMCS([template, query])

    substructure = rdkit.Chem.MolFromSmarts(r.smartsString)
    raw_match = query.GetSubstructMatches(substructure)
    template_match = template.GetSubstructMatches(substructure)
    # flatten it
    match = list(raw_match[0])
    template_match = list(template_match[0])

    # need to invert match to get diffs
    inv_match = [i for i in range(query.GetNumAtoms()) if i not in match]

    # get bonds
    bond_match = []
    for b in query.GetBonds():
        if b.GetBeginAtomIdx() in inv_match or b.GetEndAtomIdx() in inv_match:
            bond_match.append(b.GetIdx())

    # now get bonding changes from deletion

    def neigh_hash(a):
        return "".join(sorted([n.GetSymbol() for n in a.GetNeighbors()]))

    for ti, qi in zip(template_match, match):
        if neigh_hash(template.GetAtomWithIdx(ti)) != neigh_hash(
            query.GetAtomWithIdx(qi)
        ):
            inv_match.append(qi)

    # return inv_match, bond_match
    return_dict["inv_match"] = inv_match
    return_dict["bond_match"] = bond_match

def generate_cf_images(
    # smiles,
    counterfactual_list,
    smiles_key: str = "smiles",
    mol_size=(200, 200),
    fontsize=10,
    ):
    if len(counterfactual_list) == 0:
        return [], []

    # get aligned images
    ms = [
        # Chem.MolFromSmiles(counterfactual[smiles_key]) 
        build_rdkit_mol(counterfactual[smiles_key])
        for counterfactual in counterfactual_list
    ]
    dos = rdkit.Chem.Draw.MolDrawOptions()
    dos.useBWAtomPalette()
    dos.minFontSize = fontsize
    # compute coords of base mol
    Compute2DCoords(ms[0])
    imgs = []

    # list of counteractuals that were able to complete
    # initialise with original molecule
    counterfactual_list_complete = counterfactual_list[:1]

    for cf, m in zip(counterfactual_list[1:], ms[1:]):
        GenerateDepictionMatching2DStructure(
            m, ms[0], acceptFailure=True
        )

        aidx, bidx = moldiff(ms[0], m)

        imgs.append(
            MolToImage(
                m,
                size=mol_size,
                options=dos,
                highlightAtoms=aidx,
                highlightBonds=bidx,
                highlightColor=modify_color if len(bidx) > 0 else delete_color,
            )
        )

        # add completed cf to counterfactual_list_complete
        counterfactual_list_complete.append(cf)

    if len(ms) > 1:
        # set location of first query?
        GenerateDepictionMatching2DStructure(
            ms[0], ms[1], acceptFailure=True
        )

    # png only
    imgs.insert(0, MolToImage(ms[0], size=mol_size, options=dos))
    return counterfactual_list_complete, imgs


def plot_counterfactuals(
    counterfactual_list,
    plot_filename: str = None,
    fig = None,
    figure_kwargs: dict = None,
    mol_size = (200, 200),
    mol_fontsize: int = 10,
    nrows: int = None,
    ncols: int = None,
    smiles_key: str = "smiles",
    verbose: bool = True,
    ):

    if verbose:
        print ("Plotting",  len(counterfactual_list), "counterfactuals")


    # write_json(counterfactual_list, "counterfactual_list.json")

    # raise Exception

    counterfactual_list, imgs = generate_cf_images(
        counterfactual_list=counterfactual_list,
        mol_size=mol_size,
        fontsize=mol_fontsize,
        smiles_key=smiles_key,
    )

    print ("generated", len(counterfactual_list), "images")

    if nrows is not None:
        R = nrows
    else:
        R = math.ceil(math.sqrt(len(imgs)))
    if ncols is not None:
        C = ncols
    else:
        C = math.ceil(len(imgs) / R)
    if fig is None:
        if figure_kwargs is None:
            figure_kwargs = {"figsize": (12, 8)}
        fig, axs = plt.subplots(R, C, **figure_kwargs)
    else:
        axs = fig.subplots(R, C)
    axs = axs.flatten()
    # for i, (img, e) in enumerate(zip(imgs, exps)):
    for i, (current_target_cf, img) in enumerate(zip(counterfactual_list, imgs)):
        if "similarity" in current_target_cf:
            similarity = current_target_cf["similarity"]
        else:
            similarity = None
        if "yhat" in current_target_cf:
            yhat = current_target_cf["yhat"]
        else:
            yhat = None
        title = "Base" if i==0 else f"Similarity = {similarity:.2f}"
        title += f"\nf(x) = {yhat:.3f}"
        axs[i].set_title(title)
        axs[i].imshow(np.asarray(img), gid=f"rdkit-img-{i}")
        axs[i].axis("off")
    for j in range(i, C * R):
        axs[j].axis("off")
        axs[j].set_facecolor("white")
    plt.tight_layout()

    if plot_filename is not None:
        plt.savefig(plot_filename)
    plt.close()

    return plot_filename

# stereoisomers
def enumerate_stereoisomers(
    molecules: list,
    max_isomers_per_molecule: int = 100,
    try_embedding: bool = True, # ensure molecule is sensible
    molecule_identifier_key: str = "molecule_id",
    smiles_key: str = "smiles",
    verbose: bool = True,
    ):

    if verbose:
        print ("Enumerating stereoisomers for", len(molecules), "molecule(s)")

    # isomer options
    opts = StereoEnumerationOptions(tryEmbedding=try_embedding, unique=True, onlyUnassigned=True, maxIsomers=max_isomers_per_molecule)

    all_isomers = []

    for molecule in molecules:
        if molecule_identifier_key not in molecule:
            continue
        molecule_id = molecule[molecule_identifier_key]
        if smiles_key not in molecule:
            continue
        molecule_smiles = molecule[smiles_key]

        # mol = Chem.MolFromSmiles(molecule_smiles)
        mol = build_rdkit_mol(molecule_smiles)

        for i, isomer in enumerate( EnumerateStereoisomers(mol, options=opts), start=1):

            isomer_id = f"{molecule_id}_isomer_{i}"

            all_isomers.append({
                molecule_identifier_key: isomer_id,
                smiles_key: Chem.MolToSmiles(isomer, isomericSmiles=True),
            })

    return all_isomers

# begin functions relating to violations
def list_violations(
    filter_catalogue,
    smi: str = None,
    mol = None,
    ):

    # initialise params
    params = FilterCatalogParams()
    # add filter catalogue
    params.AddCatalog(filter_catalogue)
    # create final catalogue
    catalog = FilterCatalog(params)

    all_violations = []

    if mol is None:
        if smi is None:
            print ("Missing smiles!")
            return all_violations
        # mol = Chem.MolFromSmiles(smi)
        
        mol = build_rdkit_mol(smi)

    if mol is None:
        print ("Failed to sanitise molecule")
        return all_violations
    # add hydrogen
    mol = Chem.AddHs(mol)

    for violation in catalog.GetMatches(mol):

        all_violations.append({
            "filter_set": violation.GetProp("FilterSet"), 
            "scope": violation.GetProp("Scope"),
            "description": violation.GetDescription(),
            # "reference": violation.GetProp("Reference"),
        })
    return all_violations

def list_ALL_violations(
    smi: str = None,
    mol = None,
    ):
    return list_violations(
        filter_catalogue=FilterCatalogParams.FilterCatalogs.ALL,
        smi=smi,
        mol=mol,
    )

def list_PAINS_violations(
    smi: str = None,
    mol = None,
    ):
    return list_violations(
        filter_catalogue=FilterCatalogParams.FilterCatalogs.PAINS,
        smi=smi,
        mol=mol,
    )


def list_BRENK_violations(
    smi: str = None,
    mol = None,
    ):
    return list_violations(
        filter_catalogue=FilterCatalogParams.FilterCatalogs.BRENK,
        smi=smi,
        mol=mol,
    )

def list_NIH_violations(
    smi: str = None,
    mol = None,
    ):
    return list_violations(
        filter_catalogue=FilterCatalogParams.FilterCatalogs.NIH,
        smi=smi,
        mol=mol,
    )

def list_lipinski_violations(
    molecule_properties: dict,
    ):
    # Lipinski:
    #     Molecular Weight <= 500
    #     LogP <= 5
    #     H-Bond Donor Count <= 5
    #     H-Bond Acceptor Count <= 10


    all_violations = []

    for property_name, property_data in LIPINSKI_THRESHOLDS.items():
        if property_name not in molecule_properties:
            continue
        property_value = molecule_properties[property_name]

        max_value = property_data["lte"]
        if property_value > max_value:

            all_violations.append({
                "property_name": property_name,
                "maximum_value": max_value,
                "value": property_value,
            })


    return all_violations

def compute_maccs_fp(smi):
    return MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi)) 

def compute_morg3_fp(smi):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=3, nBits=1024 )



if __name__ == "__main__":

    # mol = build_rdkit_mol("CC1=CC=CC([C@@H]2CC[C@H](N3CCN(C4=CN=CC=C4)CC3)CC2)=C1.OC(C(F)(F)F)=O.OC(C(F)(F)F)=O")

    # raise Exception(Chem.MolToSmiles(mol, isomericSmiles=True))

    # smiles = "C[C@H](NC(=O)Cc1cc(F)cc(F)c1)C(=O)N[C@@H](C(=O)OC(C)(C)C)c2ccccc2"

    # mol_properties = compute_molecule_properties(smi=smiles)

    # print (mol_properties)

    # structure_filename = "../aiengine/checkme.mol"
    # structure_filename = "../aiengine/test_compounds/3KBF.pdb"
    # structure_filename = "../aiengine/test_compounds/Seriniquinone.sdf"
    # with open(structure_filename, "r") as f:
    #     molecule_structure_string = f.read()

    # # molecule_structure_string = molecule_structure_string.split("\n")

    # mols = load_multiple_molecules_from_file(
    #     structure_filename=structure_filename,
    # # mols = load_multiple_molecules_from_string(
    #     # molecule_structure_string=molecule_structure_string,
    #     mol_format="sdf",
    # )

    # for i, m in enumerate(mols, start=1):
    #     if m is None:
    #         print ("none")
    #         continue
    #     molecule_id = m.GetProp("_Name")
    #     if molecule_id == "":
    #         molecule_id = f"molecule_{i}"
    #     print (molecule_id, Chem.MolToSmiles(m, isomericSmiles=True))


    # from utils.io.io_utils import load_json


    # all_cfs_path = "/home/david/aiainsights/aiengine/cfs_test.json"
    # all_cfs = load_json(all_cfs_path, verbose=True)

    # ar_P10275_selected_cfs = all_cfs["AR"]["P10275"]["selected_cfs"]

    # plot_filename = "my_cfs_plot"



    # plot_counterfactuals(
    #     counterfactual_list=ar_P10275_selected_cfs,
    #     plot_filename=plot_filename,
    # )

    # plot_counterfactuals(
    #     counterfactual_list=ar_P10275_selected_cfs[:2],
    #     plot_filename=plot_filename + "2",
    # )


    # counterfactual_list = load_json("counterfactual_list.json")

    # cfs, imgs = generate_cf_images(
    #     counterfactual_list=counterfactual_list,
    #     mol_size=(200,200),
    #     fontsize=10,
    # )

    # print (len(imgs))

    # molecules = [
    #     {
    #         "molecule_id": "my_mol",
    #         "smiles": "BrC=CC1OC(C2)(F)C2(Cl)C1",
    #     },
    #     {
    #         "molecule_id": "aspirin",
    #         "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    #     }
    # ]

    # smiles = [mol["smiles"] for mol in molecules]


    # diversity_molecule_selection(
    #     smiles=smiles, 
    #     sample_size=1,
    #     verbose=True,
    # )


    # isomers = enumerate_stereoisomers(molecules, verbose=True,)

    # print (isomers)

    # fragments = BRICS_decompose_smiles_using_RDKit(molecules, verbose=True)

    # print (fragments)

    # smiles = "O=C(Cn1cnc2c1c(=O)n(C)c(=O)n2C)N/N=C/c1c(O)ccc2c1cccc2"

    # molecule_properties = compute_molecule_properties(smi=smiles)

    # print (molecule_properties)

    # print (list_lipinski_violations(molecule_properties))

    # pains_violations = list_BRENK_violations(
    #     smi="O=C(Cn1cnc2c1c(=O)n(C)c(=O)n2C)N/N=C/c1c(O)ccc2c1cccc2"
    # )

    # print (pains_violations)
    
    from rdkit.Chem.rdmolfiles import MolFromPDBFile

    from rdkit.Chem.rdMolAlign import GetBestRMS

    mol1 = MolFromPDBFile("aspirin.pdb")
    mol2 = MolFromPDBFile("aspirin_2.pdb")
    assert mol2 is not None

    print (GetBestRMS(mol1, mol2))

