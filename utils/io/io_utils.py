if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir)))


import os, gzip, json, shutil, re

import pickle as pkl

import numpy as np
import pandas as pd
from scipy import sparse as sp

from rdkit import Chem

from decimal import Decimal

import secrets 

from json import JSONDecodeError
from xmltodict import parse

from utils.sys_utils import execute_system_command, SUDO_PASSWORD

def get_token(
    token_nbytes: int = 16, # token is double length
    ): 
    """Generate a random hexadeximal string token using `secrets` package.

    Parameters
    ----------
    token_nbytes : int, optional
        Number of bytes to use for token, by default 16

    Returns
    -------
    str
        Random hexadecimal string.
    """
    return secrets.token_hex(token_nbytes)

def generate_password(
    password_nbytes: int = 8,
    ):
    """Generate password by calling `get_token` function.

    Parameters
    ----------
    password_nbytes : int, optional
        Number of bytes to use for password, by default 8

    Returns
    -------
    str
        Random hexadecimal string.
    """
    return get_token(password_nbytes)

def generate_job_id(
    job_id_nbytes: int = 16,
    ):
    """Generate job ID by calling `get_token` function.

    Parameters
    ----------
    job_id_nbytes : int, optional
        Number of bytes to use for job ID, by default 16

    Returns
    -------
    str
        Random hexadecimal string.
    """
    return get_token(job_id_nbytes)


def dataframe_to_dict(
    df, 
    **kwargs,
    ):
    """COnvert tabular data contained in a Pandas dataframe to a list of dicts.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to convert, or a path to it.

    Returns
    -------
    list
        List of records in DataFrame, each described by dicts
    """
    if not isinstance(df, pd.DataFrame): # df is a filename/file
        print ("Converting tabular data located to", df, "to a list of dicts")
        df =  pd.read_csv(df, **kwargs)
    
    # TODO: improve Yavuz's code
    df.columns = [c.replace(".", "") for c in df.columns]

    # mask out nan values since NaN is not valid JSON
    df = df.fillna("null")

    return [
        row.to_dict()
        for _, row in df.iterrows()
    ]

def sanitise_filename(
    filename, 
    max_length: int = 100,
    ):
    """Cleanup a string in order to make it a valid filename.

    Parameters
    ----------
    filename : str
        The input string.
    max_length : int, optional
        Maximum number of characters to use, by default 100

    Returns
    -------
    str
        Cleaned filename
    """

    if pd.isnull(filename):
        return ""
    if not isinstance(filename, str):
        filename = str(filename)
    return re.sub(r"[ |/(),']", "_", filename)[:max_length]

# pickle data

def write_pickle(
    obj: object,
    pickle_filename: str,
    ):
    """Write `obj` to file using `pickle`

    Parameters
    ----------
    obj : object
        The object to pickle
    pickle_filename : str
        The filename to save the object to.     

    Returns
    -------
    str
        The output filename
    """
    if not pickle_filename.endswith(".pkl"):
        pickle_filename += ".pkl"
    print ("writing pickle to", pickle_filename)
    with open(pickle_filename, "rb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

    return pickle_filename

def load_pickle(
    pickle_filename: str,
    ):
    """Load an object from a file using the pickle module.

    Parameters
    ----------
    pickle_filename : str
        The filename to load from

    Returns
    -------
    object
        The object
    """
    # assert pickle_filename.endswith(".pkl")
    # if not pickle_filename.endswith(".pkl"):
    #     pickle_filename += ".pkl"
    print ("loading pickle from", pickle_filename)
    with open(pickle_filename, "rb") as f:
        obj = pkl.load(f)
    return obj

def write_compressed_pickle(
    obj: object,
    compressed_pickle_filename: str,
    verbose: bool = False,
    ):
    """Write `obj` to file using gzip-compressed `pickle`

    Parameters
    ----------
    obj : object
        The object to pickle
    compressed_pickle_filename : str
        The filename to save the object to.     

    Returns
    -------
    str
        The output filename
    """
    if not compressed_pickle_filename.endswith(".pkl.gz"):
        compressed_pickle_filename += ".pkl.gz"
    if verbose:
        print ("Writing compressed pickle to", compressed_pickle_filename)
    with gzip.open(compressed_pickle_filename, "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
    if verbose:
        print ("Write completed")
    return compressed_pickle_filename

def load_compressed_pickle(
    compressed_pickle_filename: str,
    default: object = {},
    verbose: bool = False,
    ):
    """Load an object from a gzip-compressed file using the pickle module.

    Parameters
    ----------
    compressed_pickle_filename : str
        The filename to load from

    Returns
    -------
    object
        The loaded object
    """
    # assert compressed_pickle_filename.endswith(".pkl.gz")
    if not compressed_pickle_filename.endswith(".pkl.gz"):
        compressed_pickle_filename += ".pkl.gz"
    if verbose:
        print ("Loading compressed pickle from", compressed_pickle_filename)
    try:
        with gzip.open(compressed_pickle_filename, "rb") as f:
            obj = pkl.load(f)
        return obj
    except Exception as e:
        print ("Load compressed pickle exception", e, "returning", default)
        return default

# JSON data

def recursively_convert_json_to_object(
    json_data: object,
    ):
    """Recurse through JSON string/object and convert into full object

    Parameters
    ----------
    json_data : object
        The object to convert.
        May be instance of str, Decimal, dict, or list

    Returns
    -------
    object
        The JSON object described by `json_data`
    """
    if isinstance(json_data, str):
        try:
            json_data_loaded = json.loads(json_data)
            if not isinstance(json_data_loaded, float): #  handle annoying "1E45" style cases
                json_data = recursively_convert_json_to_object(json_data_loaded)
        except json.JSONDecodeError:
            pass
    elif isinstance(json_data, np.int32) or isinstance(json_data, np.int64) or isinstance(json_data, np.uint8):
        return int(json_data)
    elif isinstance(json_data, Decimal) or isinstance(json_data, np.float32) or isinstance(json_data, np.float64):
        return float(json_data)
    elif isinstance(json_data, dict):
        for k, v in json_data.items():
            json_data[k] = recursively_convert_json_to_object(v)
    elif isinstance(json_data, list):
        for i, json_record in enumerate(json_data):
            json_data[i] = recursively_convert_json_to_object(json_record)

    return json_data


def recursively_get_all_json_keys(
    json_data: object,
    ):
    """Recurse through JSON structure and collect all unique keys into a set.

    Parameters
    ----------
    json_data : object
        The object to recurse through.
        May be instance of str, Decimal, dict, or list 

    Returns
    -------
    set
        Set of all keys from all depths of `json_data`
    """

    all_keys = set() 
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
            json_record_keys = recursively_get_all_json_keys(json_data)
            all_keys.update(json_record_keys)
        except json.JSONDecodeError:
            pass
    elif isinstance(json_data, dict):
        for k, v in json_data.items():
            all_keys.add(k)
            json_record_keys = recursively_get_all_json_keys(v)
            all_keys.update(json_record_keys)
    elif isinstance(json_data, list):
        for json_record in json_data:
            all_keys.update(recursively_get_all_json_keys(json_record))
    return all_keys

def convert_nan_to_None_json(
    json_data,
    ):
    if isinstance(json_data, dict):
        return {k:convert_nan_to_None_json(v) for k,v in json_data.items()}
    elif isinstance(json_data, list):
        return [convert_nan_to_None_json(v) for v in json_data]
    elif isinstance(json_data, float) and np.isnan(json_data):
        return None
    return json_data


def write_json(
    data: object, 
    json_filename: str, 
    indent: int = 4,
    clean: bool = True,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    verbose: bool = False,
    ):
    """Write `object` to JSON file.

    Parameters
    ----------
    data : object
        The object to write
    json_filename : str
        The filename of the JSON file to write
    indent : int, optional
        Number of characters to indent, by default 4
    clean : bool, optional
        Recurse through object and convert any JSON strings to full objects before writing, by default True
    encoding : str, optional
        String encoding to use on file, by default "utf-8"
    ensure_ascii : bool, optional
        Ensure ascii character encoding, by default False
    verbose : bool, optional
        Flag to print to console, by default False

    Returns
    -------
    str
        The output filename
    """
    
    if not json_filename.endswith(".json"):
        json_filename += ".json"

    if verbose:
        print ("writing json to", json_filename)

    if isinstance(data, str):
        if verbose:
            print ("Converting string to object")
        try:
            data = json.loads(data)
        except Exception as e:
            print ("COULD NOT CONVERT", e)
            pass

    if clean:
        data = recursively_convert_json_to_object(data)

    with open(json_filename, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    if verbose:
        print ("wrote json to", json_filename)
    return json_filename

def load_json(
    json_filename: str, 
    key_type: object = None,
    default: object = dict(),
    verbose: bool = False, 
    ):
    """Load JSON encoded data from `json_filename`.

    Parameters
    ----------
    json_filename : str
        The file to load from
    key_type : object, optional
        Type of object keys, by default None
    default : object, optional
        Value to return on read error, by default dict()
    verbose : bool, optional
        Flag to print to console, by default False

    Returns
    -------
    object
        The loaded object
    """
    try:
        if verbose:
            print ("loading json from", json_filename)
        # return default if file does not exist
        if not os.path.exists(json_filename):
            if verbose:
                print (json_filename, "does not exist")
                print ("Returning", default)
            return default
        # load the JSON file using the json library
        with open(json_filename, "r") as f:
            obj = json.load(f)
        if key_type is not None and isinstance(obj, dict):
            obj = {
                key_type(key): value
                for key, value in obj.items()
            }
        if verbose:
            print ("loaded data from", json_filename)
    except JSONDecodeError as e:
        print ("Error reading json file", json_filename, ":", e)
        print ("Returning", default)
        obj = default
    return obj

# gzip
def gzip_file(
    input_filename: str,
    output_filename: str = None,
    delete_original_file: bool = False,
    verbose: bool = False,
    ):

    if output_filename is None:
        output_filename = input_filename + ".gz" 

    if verbose:
        print ("Gzipping file", input_filename, "to", output_filename)
    if not output_filename.endswith(".gz"):
        if verbose:
            print (output_filename, "does not end with .gz, adding it")
        output_filename += ".gz"

    if not os.path.exists(input_filename):
        print (input_filename, "does not exist!")
        return None

    with open(input_filename, "rb") as infile, gzip.open(output_filename, "wb") as outfile:
        shutil.copyfileobj(infile, outfile)

    
    if delete_original_file:
        delete_file(input_filename, verbose=verbose)

    return output_filename

def gunzip_file(
    gzip_filename: str,
    output_filename: str = None,
    delete_gzip_file: bool = False,
    verbose: bool = False,
    ):

    if output_filename is None:
        output_filename, _ = os.path.splitext(gzip_filename) # remove .gz 

    if verbose:
        print ("Gunzipping file", gzip_filename, "to", output_filename)
    if not gzip_filename.endswith(".gz"):
        if verbose:
            print (gzip_filename, "does not have a .gz extension, adding it")
        gzip_filename += ".gz"

    if not os.path.exists(gzip_filename):
        print (gzip_filename, "does not exist!")
        return None


    with gzip.open(gzip_filename, "rb") as infile, open(output_filename, "wb") as outfile:
        shutil.copyfileobj(infile, outfile)

    if delete_gzip_file:
        delete_file(gzip_filename, verbose=verbose)

    return output_filename

# smiles
def is_valid_smiles(
    smi : str,
    ):
    """Function to use RDKit to determine the validity of a SMILES string, `smi`

    Parameters
    ----------
    smi : str
        The SMILES string to check

    Returns
    -------
    bool
        Whether or not the SMILES string is valid
    """
    if smi is None:
        return False
    try:
        return Chem.MolFromSmiles(smi) is not None
    except TypeError:
        return False


def parse_user_submitted_compounds(
    submitted_compound_string: str,
    ):
    """Function to take a single string potentially containing multiple SMILES and molecule identifiers and split them up sensibly.
    SMILES can be optionally followed by molecule identifiers.

    Examples
    ----------
    "CCCCC molecule_1 CCHNCC molecule 2"
    "CCCCC CCHNCC"
    "CCCCC CCHNCC molecule_2"

    Parameters
    ----------
    submitted_compound_string : str
        String to parse     

    Returns
    -------
    list
        A pair of lists: 
            1) a list of identified SMILES
            2) a list of molecule identifiers
        The first element in the SMILES list matches with the first identifier in the molecule identifier list, 
        and so on.
    """

    read_molecule_smiles = []
    read_molecule_identifiers = []

    molecule_identifier = None
    found_smiles = False

    unknown_count = 0

    for delimiter in (None, ","):

        for token in submitted_compound_string.split(delimiter):
            is_smiles = is_valid_smiles(token)

            if is_smiles:
                if found_smiles:
                    if molecule_identifier is None:
                        unknown_count += 1
                        molecule_identifier = f"unnamed_molecule_{unknown_count}"
                    read_molecule_identifiers.append(molecule_identifier)
                    molecule_identifier = None
                    found_smiles = False
                read_molecule_smiles.append(token)
                found_smiles = True 
            else:
                if molecule_identifier is None:
                    molecule_identifier = token
                else:
                    molecule_identifier += f" {token}"
                
        if len(read_molecule_identifiers) < len(read_molecule_smiles):
            if molecule_identifier is None:
                unknown_count += 1
                molecule_identifier = f"unnamed_molecule_{unknown_count}"
            read_molecule_identifiers.append(molecule_identifier)

        molecule_identifier = None

        if len(read_molecule_smiles) > 0:
            break # None delimiter was successful

    return read_molecule_smiles, read_molecule_identifiers

def write_smiles(
    smiles, 
    smiles_filename: str,
    smiles_key: str = "smiles",
    molecule_identifier: str = "molecule_id",
    delimiter="\t",
    verbose=False,
    ):
    """Write a list of SMILES to file.

    Parameters
    ----------
    smiles : list
        The list of SMILES to write
    smiles_filename : str
        The filename to write to
    smiles_key : str, optional
        Key to identify SMILES if a list of dicts is passed in, by default "smiles"
    molecule_identifier : str, optional
        Key to identify molecules if a list of dicts is passed in, by default "molecule_id"
    delimiter : str, optional
        Delimiter to use in file, by default "\t"
    verbose : bool, optional
        Flag to print to console, by default False

    Returns
    -------
    str
        The filename of the new SMILES file
    """
    assert isinstance(smiles, list) 
    if verbose:
        print ("Writing", len(smiles), "SMILES to", smiles_filename)
    with open(smiles_filename, "w") as f:
        for smi in smiles:
            if isinstance(smi, tuple):
                # list of (molecule_id, smiles) tuples
                molecule_id, molecule_smiles = smi
            elif isinstance(smi, dict):
                # use molecule_identifer
                molecule_id = smi[molecule_identifier]
                molecule_smiles = smi[smiles_key]
            f.write(f"{molecule_smiles}{delimiter}{molecule_id}\n")
    return smiles_filename

def read_smiles(
    smiles_filename: str, 
    remove_invalid_molecules: bool = True, 
    clean_molecules_with_rdkit: bool = False,
    return_series: bool = False, 
    return_list: bool = False,
    assume_clean_input: bool = False,
    molecule_identifier_key: str = "molecule_id",
    smiles_key: str = "smiles",
    delimiter: str = "\\s+",
    verbose: bool =False,
    ):
    """Read molecule SMILES from file

    Parameters
    ----------
    smiles_filename : str
        The name of the file to read from
    remove_invalid_molecules : bool, optional
        Flag to use RDKit to eliminate invalid molecules, by default True
    clean_molecules_with_rdkit : bool, optional
        Flag to replace SMILES with RDKit-computed SMILES, by default False
    return_series : bool, optional
        Flag to return as an instance of Pandas Series, by default False
    return_list : bool, optional
        Flag to return as a list of dicts, by default False
    assume_clean_input : bool, optional
        Flag to assume input is well-formed with a delimiter, by default False
    molecule_identifier_key : str, optional
        Key for molecule identifiers in returned object, by default "molecule_id"
    smiles_key : str, optional
        Key for SMILES strings in returned object, by default "smiles"
    delimiter : str, optional
        Delimiter to use to separate SMILES from molecule IDs in the input file, by default "\t"
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    object
        May be one of pd.Series, pd.DataFrame or a list of dicts
    """
    if verbose:
        print ("reading SMILES from file", smiles_filename)
    if not os.path.exists(smiles_filename):
        print ("Attempting to read SMILES from a file that does not exist:", smiles_filename)
        return []

    if smiles_filename.endswith(".csv"):
        # read in as csv
        if verbose:
            print ("SMILES file is a csv file")
        smiles_df = pd.read_csv(smiles_filename, index_col=0)


    elif assume_clean_input:
        if verbose:
            print ("Assuming smiles file is well formatted using delimiter", delimiter)
        smiles_df = pd.read_csv(
            smiles_filename, 
            names=[smiles_key, molecule_identifier_key],
            sep=delimiter, 
            header=None)

    else:
        if verbose:
            print ("Assumining input is not well formatted")
        with open(smiles_filename, "r") as f:
            text = f.read()
        molecule_smiles, molecules_identifiers = parse_user_submitted_compounds(text)
        # construct DataFrame from list of molecule names and smiles
        smiles_df = pd.DataFrame(
            {
                smiles_key: molecule_smiles, 
                molecule_identifier_key: molecules_identifiers
            },
        )
        
    smiles_df = smiles_df.loc[~pd.isnull(smiles_df[smiles_key])]
    if verbose:
        print ("read", smiles_df.shape[0], "SMILES from file", smiles_filename)
    if remove_invalid_molecules:
        if verbose:
            print ("removing invalid smiles")
        valid_idx = smiles_df[smiles_key].map(is_valid_smiles)
        smiles_df = smiles_df.loc[valid_idx]
    if clean_molecules_with_rdkit:
        if verbose:
            print ("Cleaning molecules using RDKit")
        smiles_df[smiles_key] = smiles_df[smiles_key].map(
            lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        )
    if verbose:
        print (smiles_df.shape[0], "SMILES remaining after filtering")
    
    # set molecule identifier as index of DataFrame
    smiles_df = smiles_df.set_index(molecule_identifier_key, drop=True)
    if not return_list and not return_series:
        return smiles_df # return as DataFrame
    smiles_df = smiles_df[smiles_key]
    if return_series:
        return smiles_df
    # return as list of dicts
    return [
        {molecule_identifier_key: mol, smiles_key: smi}
        for mol, smi in smiles_df.items()
    ]

def load_sparse_labels(
    labels_filename: str,
    ):
    """Loads a matrix saved in npz format.

    Parameters
    ----------
    labels_filename : str
        The filename of the matrix to load

    Returns
    -------
    scipy.sparse.csr_matrix
        The matrix describing the labels.
    """
    print ("loading sparse labels from", labels_filename)
    Y = sp.load_npz(labels_filename)
    print ("labels shape is", Y.shape)
    return Y # sparse format

# def standardise_smi(smi, return_smiles=False):
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         if return_smiles:
#             return smi 
#         else:
#             return mol
#     try:
#         from standardiser import standardise
#         mol = standardise.run(mol)
#     except standardise.StandardiseException as e:
#         pass
#     if return_smiles:
#         return Chem.MolToSmiles(mol)
#     else:
#         return mol

# def embed_2D_mol_in_3D(smi):
#     assert smi is not None
#     mol = Chem.MolFromSmiles(smi)
#     mol_with_H = Chem.AddHs(mol)

#     try:
#         AllChem.EmbedMolecule(mol_with_H, useRandomCoords=False)
#         AllChem.MMFFOptimizeMolecule(mol_with_H)
#     except ValueError:
#         AllChem.EmbedMolecule(mol_with_H,useRandomCoords=True)
#         AllChem.MMFFOptimizeMolecule(mol_with_H)


#     embedded_mol = Chem.RemoveHs(mol_with_H)
#     return embedded_mol

# def smiles_to_sdf(
#     smiles_filename, 
#     sdf_filename,
#     filter_valid=False,
#     standardise=True,
#     embed=False):
#     print ("converting smiles from", smiles_filename, 
#         "to SDF file", sdf_filename)
#     smiles_df = read_smiles(smiles_filename, remove_invalid_molecules=filter_valid)

#     print ("num smiles:", smiles_df.shape[0])

#     AddMoleculeColumnToFrame(smiles_df, 'SMILES', 'Molecule')
#     molColName = "Molecule"

#     if standardise:
#         print ("standardising SMILES")
#         smiles_df["MoleculeStandard"] = smiles_df["SMILES"].map(standardise_smi, na_action="ignore")
#         smiles_df["SMILESStandard"] = smiles_df["MoleculeStandard"].map(Chem.MolToSmiles, na_action="ignore")
#         molColName = "MoleculeStandard"

#     if embed:
#         print ("embedding SMILES into 3D")
#         smiles_df["MoleculeEmbedded"] = smiles_df["SMILES"].map(embed_2D_mol_in_3D, na_action="ignore")
#         molColName = "MoleculeEmbedded"

#     smiles_df = smiles_df.loc[~pd.isnull(smiles_df[molColName])] # drop missing values
#     print ("num SMILES remaining:", smiles_df.shape[0])
#     WriteSDF(smiles_df, sdf_filename, molColName=molColName,
#         idName="RowID", properties=list(smiles_df.columns))

def copy_file(
    input_filename: str, 
    output_filename: str,
    verbose: bool = False,
    ):
    """Copy file utility function.

    Parameters
    ----------
    input_filename : str
        Filename to copy
    output_filename : str
        Filename to copy to
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    str
        Output filename
    """

    if os.path.abspath(input_filename) != os.path.abspath(output_filename):
        if verbose:
            print ("Copying", input_filename, "to", output_filename)
        output_directory = os.path.dirname(output_filename)
        if output_directory != "":
            os.makedirs(output_directory, exist_ok=True)
        if os.path.exists(input_filename):
            shutil.copyfile(input_filename, output_filename)    
    
    return output_filename

def move_file(
    input_filename: str, 
    output_filename: str,
    verbose: bool = False,
    ):
    """Move file utility function.

    Parameters
    ----------
    input_filename : str
        Filename to move
    output_filename : str
        Filename to move to
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    str
        Output filename
    """

    if os.path.abspath(input_filename) != os.path.abspath(output_filename):
        if verbose:
            print ("Moving", input_filename, "to", output_filename)
        output_directory = os.path.dirname(output_filename)
        if output_directory != "":
            os.makedirs(output_directory, exist_ok=True)
        if os.path.exists(input_filename):
            shutil.move(input_filename, output_filename)    
    
    return output_filename

def delete_file(
    filename : str,
    verbose: bool = False,
    ):
    """Safely delete a file by checking that it exists

    Parameters
    ----------
    filename : str
        File to delete
    verbose : bool, optional
        Flag to print updates to console, by default False
    """
    if os.path.exists(filename):
        if verbose:
            print ("Deleting file", filename)
        try:
            os.remove(filename)
        except Exception as e:
            print ("Error deleting file", filename, ":", e)

def copy_directory(
    directory_to_copy: str,
    copy_to: str,
    verbose: bool = False,
    ):

    if os.path.isdir(copy_to):
        directory_to_copy_basename = os.path.basename(directory_to_copy)
        copy_to = os.path.join(copy_to, directory_to_copy_basename)

    if not os.path.exists(copy_to): 
        if verbose:
            print ("Copying directory", directory_to_copy, "to", copy_to)
        shutil.copytree(directory_to_copy, copy_to)   

    return copy_to 
    

def delete_directory(
    directory_name : str,
    verbose: bool = False,
    ):
    """Safely delete a directory by checking that it exists

    Parameters
    ----------
    directory_name : str
        Directory to delete
    verbose : bool, optional
        Flag to print updates to console, by default False
    """
    if os.path.isdir(directory_name):
        if verbose:
            print ("Deleting directory", directory_name)
        try:
            shutil.rmtree(directory_name)
        except Exception as e:
            print ("Error deleting directory", directory_name, ":", e)
            print ("Trying to delete with sudo")
            cmd = f"echo {SUDO_PASSWORD} | sudo -S rm -r {directory_name}"
            try:
                execute_system_command(cmd, verbose=verbose)
                print ("Delete with sudo successful")
            except Exception as e:
                print ("sudo delete exception", e)


def download_from_client(
    input_filehandle: object, 
    output_dir: str,
    verbose: bool = False,
    ):
    """Download a file received from a client using file hanldle `input_filehandle`

    Parameters
    ----------
    input_filehandle : object
        Handle on file object
    output_dir : str
        Directory to write to
    verbose : bool, optional
        Flag to print updates to console, by default False

    Returns
    -------
    str
        Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    assert hasattr(input_filehandle, "name")
    input_filename = input_filehandle.name
    input_filename = sanitise_filename(input_filename)
    if verbose:
        print ("input file", input_filename, "has been received from client -- downloading")

    # write compounds to local directory of server
    local_filename = os.path.join(
        output_dir,
        input_filename)
    if verbose:
        print ("downloading file to", local_filename)
    with open(local_filename, "wb+") as out_file:
        if verbose:
            print ("opened file", local_filename)
        for i, chunk in enumerate(input_filehandle.chunks(), start=1):
            out_file.write(chunk)
            if verbose:
                print ("wrote chunk", i)
    if verbose:
        print ("Download file from client complete")
    return local_filename

def clean_up_molecule_names(
    molecule_names: list,
    ):
    """Ensure all submitted names are valid filenames and unique

    Parameters
    ----------
    molecule_names : list
        List of molecule names

    Returns
    -------
    list
        Cleaned up list of unique molecule names
    """

    n_molecules = len(molecule_names)
    
    unknown_count = 0

    # sanitise names and identify missing names
    for i in range(n_molecules):
        name = molecule_names[i]
        name_sanitised = sanitise_filename(name)
        if name_sanitised == "" or name_sanitised == "None":
            name_sanitised = f"unknown_molecule_{unknown_count+1}"
            unknown_count += 1
        if name_sanitised != name:
            molecule_names[i] = name_sanitised

    # handle duplicate names (add wart to duplicate names)
    name_counts = dict()
    for i in range(n_molecules):
        name = molecule_names[i]
        if name not in name_counts:
            name_counts[name] = 0
        name_counts[name] += 1
        if name_counts[name] > 1: # if count > 1, add wart to name
            molecule_names[i] = f"{molecule_names[i]}_{name_counts[name]}"

    assert len(set(molecule_names)) == n_molecules

    return molecule_names

# def get_compound_names(filename):
#     print ("DETERMINING COMPOUND NAMES IN SUBMITTED FILE", filename)
#     if filename.endswith(".sdf"):
#         sdf_id_col = "ID"
#         sdf = LoadSDF(filename, idName=sdf_id_col)
#         names = list(sdf[sdf_id_col].values)
#         changed, names = clean_up_molecule_names(names)
#         if changed:
#             sdf[sdf_id_col] = names
#             print ("names have changed, writing SDF to file", filename)
#             WriteSDF(sdf, out=filename, idName=sdf_id_col)
#     else: # assume smiles file
#         print ("FILE IS IN SMILES FORMAT")
#         smiles = read_smiles(filename, 
#             remove_invalid_molecules=True, 
#             return_series=True)
#         names = list(smiles.index)
#         _, names = clean_up_molecule_names(names)
            
#         print ("names have changed, writing smiles to file", filename)
#         write_smiles([(name, smi) 
#             for name, smi in zip(names, smiles.values)], 
#         filename)

#     print ("COMPLETED DETERMINING COMPOUND NAMES IN SUBMITTED FILE", filename)
#     return names

# def convert_file(
#     input_filename: str, 
#     desired_format: str,
#     ):

#     desired_format = desired_format.lower()

#     smiles_input_types = {".smi", ".txt", ".tsv", ".sml", ".smiles", ".msi"}
#     sdf_input_types = {".sdf", }

#     valid_input_file_types = {*smiles_input_types, *sdf_input_types}

#     input_filename, input_file_type = os.path.splitext(input_filename)
#     assert input_file_type in valid_input_file_types



#     # convert if necessary
#     if input_file_type != desired_format:
        
#         print ("CONVERSION REQUIRED")

#         if desired_format == ".smi":
#             print ("COVERTING TO SMILES FORMAT")
#             if input_file_type == ".sdf":
#                 # convert SDF to smiles
#                 print ("converting SDF to SMILES")
#                 print ("SDF filename:", input_filename)
#                 sdf_df = LoadSDF(input_filename, smilesName="SMILES")
#                 # write smiles
#                 smiles = [(row["ID"], row["SMILES"])
#                     for _, row in sdf_df.iterrows()]
#                 # write smiles to temp_file
#                 output_filename = input_filename + desired_format
#                 write_smiles(smiles, output_filename)
           
#             elif input_file_type in valid_input_file_types - {".sdf"}:
#                 # rename .txt smiles format to .smi
#                 print ("INPUT FILE TYPE:", input_file_type)
#                 # os.rename(input_file, input_file_name + desired_format)
#                 output_filename = input_filename + desired_format
#                 print ("COPYING FROM", input_filename, "TO", output_filename)
#                 shutil.copyfile(input_filename, output_filename)
#             else:
#                 raise NotImplementedError

#         elif desired_format == ".sdf":
#             if input_file_type in {".smi", ".sml", ".txt"}:
#                 # convert from SMILES to SDF
#                 print ("converting SMILES to SDF")
#                 smiles_filename = input_filename
#                 print ("SMILES filename:", smiles_filename)
#                 output_filename = input_filename + desired_format
#                 smiles_to_SDF_3D_rdkit(
#                     smiles_filename, 
#                     output_filename,
#                     standardise=False, 
#                     embed=False)
#             else:
#                 raise NotImplementedError
        
#         else:
#             raise NotImplementedError # conversion not yet implemented
#     else:
#         print ("NO CONVERSION REQUIRED")
#         output_filename = input_filename
    
#     # assert output_filename.endswith(desired_format)

#     # # copy to output directory
#     # output_filename = os.path.join(output_dir, os.path.basename(input_file))
#     # if output_filename != input_file:
#     #     shutil.copyfile(input_file, output_filename)

#     return output_filename

def read_xml_file_and_covert_to_dict(
    xml_filename: str,
    verbose: bool = False,
    ):
    """Use `xmltodict` library to parse xml file as dictionary.

    Parameters
    ----------
    xml_filename : str
        Path of XML file to read
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    dict
        The data in the XML file, in dictionary form
    """
    if verbose:
        print ("Reading xml file", xml_filename)
    with open(xml_filename, "r",) as f:
        xml_data = parse(f.read())
    return xml_data

if __name__ == "__main__":
    
    
    smiles = read_smiles(
        smiles_filename="checkme.smi",
        return_list=True,
        smiles_key="smiles",
        molecule_identifier_key="molecule_id",
    )

    print (smiles)

    # print (load_json("tnjogr"))

    # print(gunzip_file(
    #     gzip_filename="checkme.poc.gz",
    #     delete_gzip_file=False,
    # ))

    # gzip_filename = gzip_file(
    #     input_filename="checkme.poc",
    #     output_filename="thisisatest.poc.gz",
    #     verbose=True,
    # )

    # gunzip_file(
    #     gzip_filename="/media/david/Elements//pocket_align_scores/POC_6A/Q16654_2E0A_A_ANP_501_6.poc.gz",
    #     output_filename="checkme.txt",
    #     verbose=True
    # )