if __name__ == "__main__":
    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import re
import time

from urllib.parse import urlencode

import os

from utils.io.io_utils import write_json, load_json

from utils.request_utils import make_http_request

# will be empty if file does not exist
# bioactivity prediction targets only 
# ALL_UNIPROT_TARGETS_FILENAME = "data/databases/uniprot/all_targets_bioactivity_prediction.json"
# load ~50k uniprot targets to memory?
ALL_UNIPROT_TARGETS_FILENAME = "data/databases/uniprot/all_targets_docking.json"
ALL_UNIPROT_TARGETS = load_json(ALL_UNIPROT_TARGETS_FILENAME, verbose=False)

UNIPROT_TARGET_FIELD_ORDER = (
    "gene",
    "gene_secondary",
    "uniprot_id",
    "accession",
    "protein",
    "protein_secondary",
    "organism_scientific",
)

def uniprot_api_query(
    queries: list,   
    query_term: str,
    human_only: bool = False,
    reviewed: bool = False,
    include_pdb: bool = False,
    include_sequence: bool = True,
    chunk_size: int = 10,
    verbose: bool = False,
    ):

    if chunk_size > 10:
        chunk_size = 10 # seems to work best

    if isinstance(queries, str):
        queries = [queries]
    if isinstance(queries, set):
        queries = sorted(queries)

    num_queries = len(queries)

    n_chunks = num_queries // chunk_size + 1

    root_url = "https://rest.uniprot.org/uniprotkb/search"

    # updated API keys can be found here: https://www.uniprot.org/help/return_fields

    columns = ["id", "accession",  "protein_name", "gene_names", "organism_name", ]
    if include_pdb:
        columns.append("xref_pdb")
    if include_sequence:
        columns.append("sequence")
    columns_str = ",".join(columns)

    if verbose:
        print ("Querying Uniprot with the following parameters:")
        print (query_term, queries)

    # iterate over chunks and make separate requests
    all_records_from_uniprot = []

    for chunk_num in range(n_chunks):
        query_chunk = queries[chunk_num * chunk_size: (chunk_num+1) * chunk_size]
        if len(query_chunk) == 0:
            continue
        
        all_query_parameters = []
        
        # convert to string

        query_chunk =  "+OR+".join((f"{query_term}:{q}" for q in query_chunk))
        all_query_parameters.append(query_chunk)

        if human_only:
            all_query_parameters.append("organism_id:9606")
        if reviewed:
            all_query_parameters.append("reviewed:true")
        if include_pdb:
            all_query_parameters.append("database:pdb")

        all_query_parameters = "+AND+".join(all_query_parameters)

        params = {
            "query": all_query_parameters,
            "format": "tsv",
            "fields": columns_str,
        }

        # new API requires skipping url encoding ":+,"
        params = urlencode(params, safe=":+,()")

        response = make_http_request(root_url, params=params, verbose=verbose, method="GET", sleep_duration=5)
        if response is None:
            continue
        response = response.text 

        # currently implemented for tsv format
        for line in response.split("\n"):
            # skip empty line 
            if line == "":
                continue
            # skip first line 
            if line.startswith("Entry"):
                continue

            line_split = line.split("\t")
            record = {}
            for key_name, entry in zip(
                ("uniprot_identifier", "accession", "protein", "gene", "organism_scientific"),
                line_split,
                ):

                if key_name == "protein":
                    protein_split = entry.split(" (")
                    if len(protein_split) == 0:
                        # raise Exception("protein", entry)
                        record[key_name] = None
                    else:
                        # primary
                        record[key_name] = protein_split[0]
                        record[f"{key_name}_secondary"] = [p[:-1] for p in protein_split[1:]] # remove trailing )
                elif key_name == "gene":
                    gene_split = entry.split()
                    if len(gene_split) == 0:
                        # raise Exception("gene", entry, line_split)
                        record[key_name] = None 
                    else:
                        # primary gene name
                        record[key_name] = gene_split[0]
                        record[f"{key_name}_secondary"] = gene_split[1:]
                else:
                    record[key_name] = entry

            if include_pdb:
                pdb_index = columns.index("xref_pdb")
                record["pdb_ids"] = [
                    pdb_id
                    for pdb_id in  
                    line_split[pdb_index].split(";")
                    if pdb_id != ""
                ]
            if include_sequence:
                sequence_index = columns.index("sequence")
                # record["sequences"] = [
                record["sequence"] = [
                    sequence
                    for sequence in  
                    line_split[sequence_index].split(";") 
                    if sequence != ""
                ][0] # just primary sequence

            all_records_from_uniprot.append(record)

    return all_records_from_uniprot


def uniprot_api_query_accession(
    accessions,
    human_only: bool = False,
    reviewed: bool = False,
    include_pdb: bool = False,
    include_sequence: bool = True,
    chunk_size: int = 10,
    verbose: bool = False,
    ):

    return uniprot_api_query(
        accessions,
        query_term="accession",
        human_only=human_only,
        reviewed=reviewed,
        include_pdb=include_pdb,
        include_sequence=include_sequence,  
        chunk_size=chunk_size,
        verbose=verbose,
    )

def uniprot_api_query_ec_number(
    ec_numbers,
    human_only: bool = False,
    reviewed: bool = False,
    include_pdb: bool = False,
    include_sequence: bool = True,
    chunk_size: int = 10,
    verbose: bool = False,
    ):

    return uniprot_api_query(
        ec_numbers,
        query_term="ec",
        human_only=human_only,
        reviewed=reviewed,
        include_pdb=include_pdb,
        include_sequence=include_sequence,  
        chunk_size=chunk_size,
        verbose=verbose,
    )

def uniprot_api_query_gene_name(
    gene_names,
    human_only: bool = False,
    reviewed: bool = False,
    include_pdb: bool = False,
    include_sequence: bool = True,
    chunk_size: int = 10,
    verbose: bool = False,
    ):

    return uniprot_api_query(
        gene_names,
        query_term="gene",
        human_only=human_only,
        reviewed=reviewed,
        include_pdb=include_pdb,
        include_sequence=include_sequence,  
        chunk_size=chunk_size,
        verbose=verbose,
    )

def uniprot_api_query_protein_name(
    protein_names,
    human_only: bool = False,
    reviewed: bool = False,
    include_pdb: bool = False,
    include_sequence: bool = True,
    chunk_size: int = 10,
    verbose: bool = False,
    ):

    if isinstance(protein_names, str):
        protein_names = [protein_names]

    protein_names = [
        re.sub("[\[\]]", "", protein_name)
        for protein_name in protein_names 
    ]

    return uniprot_api_query(
        protein_names,
        query_term="protein_name",
        human_only=human_only,
        reviewed=reviewed,
        include_pdb=include_pdb,
        include_sequence=include_sequence,  
        chunk_size=chunk_size,
        verbose=verbose,
    )

# def query_uniprot_using_api(
#     accessions: list = None,    
#     gene_names: list = None,    
#     protein_names: list = None,    
#     ec_numbers: list = None,
#     human_only: bool = False,
#     reviewed: bool = False,
#     include_pdb: bool = False,
#     include_sequence: bool = True,
#     chunk_size: int = 10,
#     verbose: bool = False,
#     ):

#     if isinstance(accessions, str):
#         accessions = [accessions]

#     if isinstance(gene_names, str):
#         gene_names = [gene_names]

#     if isinstance(protein_names, str):
#         protein_names = [protein_names]
    
#     if isinstance(ec_numbers, str):
#         ec_numbers = [ec_numbers]

#     if accessions is not None:
#         num_accessions_in_query = len(accessions)
#     else:
#         num_accessions_in_query = 0
#     if gene_names is not None:
#         num_gene_names_in_query = len(gene_names)
#     else:
#         num_gene_names_in_query = 0
#     if protein_names is not None:
#         num_protein_names_in_query = len(protein_names)
#     else:
#         num_protein_names_in_query = 0

#     # if any query parameters contain more than chunk_size elements, then chunk the query
#     if num_accessions_in_query > chunk_size:
#         # chunk accessions 
#         n_chunks = num_accessions_in_query // chunk_size + 1

#         accession_chunks = [
#             accessions[chunk_num*chunk_size:(chunk_num+1)*chunk_size]
#             for chunk_num in range(n_chunks)
#             if len(accessions[chunk_num*chunk_size:(chunk_num+1)*chunk_size]) > 0
#         ]
#         # ignore any supplied gene and protein names
#         gene_name_chunks = [None] * len(accession_chunks)
#         protein_name_chunks = [None] * len(accession_chunks)
#     elif num_gene_names_in_query > chunk_size:
#         # chunk gene names 
#         n_chunks = num_gene_names_in_query // chunk_size + 1

#         gene_name_chunks = [
#             gene_names[chunk_num*chunk_size:(chunk_num+1)*chunk_size]
#             for chunk_num in range(n_chunks)
#             if len(gene_names[chunk_num*chunk_size:(chunk_num+1)*chunk_size]) > 0
#         ]
#         # ignore any supplied gene and protein names
#         accession_chunks = [None] * len(gene_name_chunks)
#         protein_name_chunks = [None] * len(gene_name_chunks)
#     elif num_protein_names_in_query > chunk_size:
#         # chunk protein names 
#         n_chunks = num_protein_names_in_query // chunk_size + 1

#         protein_name_chunks = [
#             protein_names[chunk_num*chunk_size:(chunk_num+1)*chunk_size]
#             for chunk_num in range(n_chunks)
#             if len(protein_names[chunk_num*chunk_size:(chunk_num+1)*chunk_size]) > 0
#         ]
#         # ignore any supplied gene and protein names
#         accession_chunks = [None] * len(protein_name_chunks)
#         gene_name_chunks = [None] * len(protein_name_chunks)
#     else:
#         # use all supplied query parameters
#         accession_chunks = [accessions]
#         gene_name_chunks = [gene_names]
#         protein_name_chunks = [protein_names]

#     root_url = "https://rest.uniprot.org/uniprotkb/search"

#     # updated API keys can be found here: https://www.uniprot.org/help/return_fields

#     columns = ["id", "accession",  "protein_name", "gene_names", "organism_name", ]
#     if include_pdb:
#         columns.append("xref_pdb")
#     if include_sequence:
#         columns.append("sequence")
#     columns_str = ",".join(columns)

#     if verbose:
#         print ("Querying Uniprot with the following parameters:")
#         print ("Accessions", accessions)
#         print ("Gene names", gene_names)
#         print ("Protein names", protein_names)

#     # iterate over chunks and make separate requests
#     all_records_from_uniprot = []

#     for chunk_accessions, chunk_gene_names, chunk_protein_names in zip(
#         accession_chunks, gene_name_chunks, protein_name_chunks,
#     ):

#         chunk_query = []
#         if chunk_accessions is not None:
#             chunk_query.append( "+OR+".join((f"accession:{accession}" for accession in chunk_accessions)) )
#         if chunk_gene_names is not None:
#             chunk_query.append( "+OR+".join((f"gene:{gene_name}" for gene_name in chunk_gene_names)) )
#         if chunk_protein_names is not None:
#             chunk_protein_names = [ 
#                 re.sub("[\[\]]", "", protein_name)
#                 for protein_name in chunk_protein_names
#             ]
#             chunk_query.append( "+OR+".join((f"protein_name:{protein_name}" for protein_name in chunk_protein_names)) )
        
#         if len(chunk_query) == 0:
#             if verbose:
#                 print ("No query parameters!")
#             # return []
#             continue
        
#         if human_only:
#             chunk_query.append("organism_id:9606")
#         if reviewed:
#             chunk_query.append("reviewed:true")
#         if include_pdb:
#             chunk_query.append("database:pdb")

#         chunk_query = "+AND+".join(chunk_query)

#         params = {
#             "query": chunk_query,
#             "format": "tsv",
#             "fields": columns_str,
#         }

#         # new API requires skipping url encoding ":+,"
#         params = urlencode(params, safe=":+,()")

#         response = make_http_request(root_url, params=params, verbose=verbose, method="GET", sleep_duration=5)
#         if response is None:
#             continue
#         response = response.text 

#         # currently implemented for tsv format
#         for line in response.split("\n"):
#             # skip empty line 
#             if line == "":
#                 continue
#             # skip first line 
#             if line.startswith("Entry"):
#                 continue

#             line_split = line.split("\t")
#             record = {}
#             for key_name, entry in zip(
#                 ("uniprot_identifier", "accession", "protein", "gene", "organism_scientific"),
#                 line_split,
#                 ):

#                 if key_name == "protein":
#                     protein_split = entry.split(" (")
#                     if len(protein_split) == 0:
#                         # raise Exception("protein", entry)
#                         record[key_name] = None
#                     else:
#                         # primary
#                         record[key_name] = protein_split[0]
#                         record[f"{key_name}_secondary"] = [p[:-1] for p in protein_split[1:]] # remove trailing )
#                 elif key_name == "gene":
#                     gene_split = entry.split()
#                     if len(gene_split) == 0:
#                         # raise Exception("gene", entry, line_split)
#                         record[key_name] = None 
#                     else:
#                         # primary gene name
#                         record[key_name] = gene_split[0]
#                         record[f"{key_name}_secondary"] = gene_split[1:]
#                 else:
#                     record[key_name] = entry

#             if include_pdb:
#                 pdb_index = columns.index("xref_pdb")
#                 record["pdb_ids"] = [
#                     pdb_id
#                     for pdb_id in  
#                     line_split[pdb_index].split(";")
#                     if pdb_id != ""
#                 ]
#             if include_sequence:
#                 sequence_index = columns.index("sequence")
#                 record["sequences"] = [
#                     sequence
#                     for sequence in  
#                     line_split[sequence_index].split(";")
#                     if sequence != ""
#                 ]

#             all_records_from_uniprot.append(record)

#     return all_records_from_uniprot


# updated database mapping API
# available fields can be found here: curl https://rest.uniprot.org/configure/idmapping/fields

# POLLING_INTERVAL = 3
# DB_MAPPING_API_URL = "https://rest.uniprot.org"

# def submit_id_mapping(
#     map_from, 
#     map_to, 
#     ids,
#     verbose: bool = False,):
#     if not isinstance(ids, str):
#         ids = ",".join(ids)
#     response = make_http_request(
#         url=f"{DB_MAPPING_API_URL}/idmapping/run",
#         params={
#             "from": map_from, 
#             "to": map_to, 
#             "ids": ids
#         },
#         method="POST",
#         verbose=verbose)
#     response.raise_for_status()
#     return response.json()["jobId"]

# # def check_id_mapping_results_ready(job_id):
# #     while True:
# #         request = session.get(f"{API_URL}/idmapping/status/{job_id}")
# #         request.raise_for_status()
# #         j = request.json()
# #         if "jobStatus" in j:
# #             if j["jobStatus"] == "RUNNING":
# #                 print(f"Retrying in {POLLING_INTERVAL}s")
# #                 time.sleep(POLLING_INTERVAL)
# #             else:
# #                 raise Exception(request["jobStatus"])
# #         else:
# #             return bool(j["results"] or j["failedIds"])

# def get_id_mapping_results_link(job_id):
#     url = f"{DB_MAPPING_API_URL}/idmapping/details/{job_id}"
#     request = session.get(url)
#     request.raise_for_status()
#     return request.json()["redirectURL"]

# def get_id_mapping_results(
#     job_id,
#     verbose: bool = False,
#     ):
#     if verbose:
#         print ("Obtaining results for Uniprot job ID", job_id)
#     while True:
#         r = make_http_request(
#             url=f"{DB_MAPPING_API_URL}/idmapping/status/{job_id}",
#             params={},
#             method="GET",
#             max_retries=1,
#             sleep_duration=0,
#             verbose=verbose,
#         )
#         if r is None:
#             return {
#                 "results": [],
#                 "failedIds": [],
#             }

#         r.raise_for_status()
#         job = r.json()
#         if "jobStatus" in job:
#             if job["jobStatus"] == "RUNNING":
#                 if verbose:
#                     print("Retrying in", POLLING_INTERVAL, "seconds")
#                 time.sleep(POLLING_INTERVAL)
#             else:
#                 raise Exception(job["jobStatus"])
#         else:
#             return job


import re
import time
import json
import zlib
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry


POLLING_INTERVAL = 3

DATABASE_MAP_API_URL = "https://rest.uniprot.org"

retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def submit_id_mapping(
    map_from, 
    map_to, 
    ids,
    verbose: bool = False,
    ):
    if verbose:
        print ("Mapping", map_from, "to", map_to, "using Uniprot API")
    request = requests.post(
        f"{DATABASE_MAP_API_URL}/idmapping/run",
        data={
            "from": map_from, 
            "to": map_to, 
            "ids": ",".join(ids)
        },
    )
    request.raise_for_status()
    return request.json()["jobId"]


def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{DATABASE_MAP_API_URL}/idmapping/status/{job_id}")
        request.raise_for_status()
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] == "RUNNING":
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):
    url = f"{DATABASE_MAP_API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    request.raise_for_status()
    return request.json()["redirectURL"]


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    return response.text


def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    print(f"Fetched: {n_fetched} / {total}")


def get_id_mapping_results_search(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0].lower() if "format" in query else "json"
    if "size" in query:
        size = int(query["size"][0])
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    request.raise_for_status()
    results = decode_results(request, file_format, compressed)
    if "x-total-results" in request.headers:
        total = int(request.headers["x-total-results"])
        print_progress_batches(0, size, total)
        for i, batch in enumerate(get_batch(request, file_format, compressed)):
            results = combine_batches(results, batch, file_format)
            print_progress_batches(i + 1, size, total)
    return results


def get_id_mapping_results_stream(url):
    if "/stream/" not in url:
        url = url.replace("/results/", "/stream/")
    request = session.get(url)
    request.raise_for_status()
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)


def map_database_ids(
    query, 
    map_from: str = "CHEMBL_ID",
    map_to: str = "ACC", 
    human_only: bool = False,
    return_format: str = "tab",
    max_retries: int = 3,
    sleep_duration: int = 5,
    verbose: bool = True,
    ):
    """Use UniProt HTTP API to map between database identifiers

    Parameters
    ----------
    query : list/set/str
        List of IDs to map
    map_from : str, optional
        ID type to map from, by default "CHEMBL_ID"
    map_to : str, optional
        ID type to map to, by default "ACC"
    human_only : bool, optional
        Flag to retreive only human records, by default False
    return_format : str, optional
        Response return format, currenly only "tab" is implemented, by default "tab"
    max_retries : int, optional
        Maximum number of HTTP request attempts, by default 3
    sleep_duration : int, optional
        Time in seconds between request attempts, by default 5
    verbose : bool, optional
        Flag to print updates to the console, by default True

    Returns
    -------
    dict
        A mapping from supplied ID(s) to a list of matched IDs

    Raises
    ------
    NotImplementedError
        Thrown if `return_format` is not "tab"
    """

    '''
    ALL POSSIBLE MAPS:
    https://www.uniprot.org/help/api_idmapping


    curl https://rest.uniprot.org/configure/idmapping/fields
    '''

    if verbose:
        print ("Mapping query", query, "from", map_from, "to", map_to, "using UNIPROT DB")
        print ("Maximum number of retries:", max_retries)
        print ("Sleep duration:", sleep_duration)


    if query is None:
        if verbose:
            print ("Query is None")
        return {}


    if isinstance(query, str):
        query = [query]


    # updated to new API
    job_id = submit_id_mapping(
        map_from=map_from,
        map_to=map_to,
        ids=query,
        verbose=verbose,
    )

    if verbose:
        print ("Obtained job ID", job_id)

    # mapping_results = get_id_mapping_results(
    #     job_id,
    #     verbose=verbose
    # )

    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        mapping_results = get_id_mapping_results_search(link)
        # Equivalently using the stream endpoint which is more demanding
        # on the API and so is less stable:
        # mapping_results = get_id_mapping_results_stream(link)

    return_dict = {}
    if "results" in mapping_results:
        for record in mapping_results["results"]:
            original_id = record["from"] 
            if original_id not in return_dict:
                return_dict[original_id] = []
            new_id = record["to"] 
            return_dict[original_id].append(new_id)
    return return_dict

def map_uniprot_accession_to_entrez_gene_id(
    accessions: list, 
    human_only: bool = False,
    ):
    """Helper function to map from UniProt accession ID to Entrez Gene ID

    Parameters
    ----------
    accessions : list
        List of accessions
    human_only : bool, optional
        Flag to retreive only human records, by default False

    Returns
    -------
    dict
        A mapping from supplied ID(s) to a list of matched IDs
    """
    return map_database_ids(
        query=accessions,
        map_from="UniProtKB_AC-ID",
        map_to="GeneID",
        human_only=human_only,
    )

def map_uniprot_accession_to_chembl_id(
    accessions: list, 
    human_only: bool = False,
    ):
    """Helper function to map from UniProt accession ID to ChEMBL ID

    Parameters
    ----------
    accessions : list
        List of accessions
    human_only : bool, optional
        Flag to retreive only human records, by default False

    Returns
    -------
    dict
        A mapping from supplied ID(s) to a list of matched IDs
    """
    return map_database_ids(
        query=accessions,
        map_from="UniProtKB_AC-ID",
        map_to="ChEMBL",
        human_only=human_only,
    )

def map_uniprot_accession_to_pdb_id(
    accessions: list, 
    human_only: bool = False,
    ):
    """Helper function to map from UniProt accession ID to PDB ID

    Parameters
    ----------
    accessions : list
        List of accessions
    human_only : bool, optional
        Flag to retreive only human records, by default False

    Returns
    -------
    dict
        A mapping from supplied ID(s) to a list of matched IDs
    """
    return map_database_ids(
        query=accessions,
        map_from="UniProtKB_AC-ID",
        map_to="PDB",
        human_only=human_only,
    )

def map_chembl_target_id_to_uniprot_accession(
    target_chembl_ids: list,
    human_only: bool = False,
    ):
    """Helper function to map from CHEMBL target ID to UniProt accession ID

    Parameters
    ----------
    target_chembl_ids : list
        List of target CHEMBL IDs
    human_only : bool, optional
        Flag to retreive only human records, by default False

    Returns
    -------
    dict
        A mapping from supplied ID(s) to a list of matched IDs
    """
    full_uniprot_data = map_database_ids(
        query=target_chembl_ids,
        map_from="ChEMBL",
        # map_to="UniProtKB_AC-ID",
        map_to="UniProtKB",
        human_only=human_only,
    )

    # select only primaryAccession
    return {
        chembl_id: [ 
            accession_record["primaryAccession"]
            for accession_record in accession_records   
            if "primaryAccession" in accession_record
        ]
        for chembl_id, accession_records in full_uniprot_data.items()
    }


def map_PDB_ID_to_uniprot_accession(
    pdb_ids: list,
    human_only: bool = False,
    ):
    """Helper function to map from PDB ID to UniProt accession ID

    Parameters
    ----------
    pdb_ids : list
        List of PDB IDs
    human_only : bool, optional
        Flag to retreive only human records, by default False

    Returns
    -------
    dict
        A mapping from supplied ID(s) to a list of matched IDs
    """
    full_uniprot_data = map_database_ids(
        query=pdb_ids,
        map_from="PDB",
        map_to="UniProtKB",
        human_only=human_only,
    )

    # select only primaryAccession
    return {
        pdb_id: [ 
            accession_record["primaryAccession"]
            for accession_record in accession_records   
            if "primaryAccession" in accession_record
        ]
        for pdb_id, accession_records in full_uniprot_data.items()
    }


# def map_EC_number_to_uniprot_accession(
#     ec_numbers: list,
#     human_only: bool = False,
#     ):
#     """Helper function to map from PDB ID to UniProt accession ID

#     Parameters
#     ----------
#     ec_numbers : list
#         List of EC numbers
#     human_only : bool, optional
#         Flag to retreive only human records, by default False

#     Returns
#     -------
#     dict
#         A mapping from supplied ID(s) to a list of matched IDs
#     """
#     full_uniprot_data = map_database_ids(
#         query=ec_numbers,
#         map_from="ec",
#         map_to="UniProtKB",
#         human_only=human_only,
#     )

#     # select only primaryAccession
#     return {
#         pdb_id: [ 
#             accession_record["primaryAccession"]
#             for accession_record in accession_records   
#             if "primaryAccession" in accession_record
#         ]
#         for pdb_id, accession_records in full_uniprot_data.items()
#     }

if __name__ == "__main__":


    records = uniprot_api_query_accession(
        "P16048",
        reviewed=False,
        include_pdb=False,
    )

    for r in records:
        print (r)
    print (len(records))

    # build chembl target hierarchy

    # import pandas as pd

    # from utils.io.io_utils import load_compressed_pickle

    # protein_type_to_family = load_compressed_pickle("data/databases/uniprot/protein_type_to_family.pkl.gz")

    # protein_type_to_family_df = pd.DataFrame([
    #     {
    #         "protein_type": protein_type,
    #         "protein_family": protein_family
    #     }
    #     for protein_type, protein_families in protein_type_to_family.items()
    #     for protein_family in protein_families
    # ])


    # protein_family_to_accession = load_compressed_pickle("data/databases/uniprot/family_to_accession.pkl.gz")

    # protein_family_to_accession_df = pd.DataFrame([ 
    #     {
    #         "protein_family": protein_family,
    #         **protein_data
    #     }
    #     for protein_family, protein_datas in protein_family_to_accession.items()
    #     for protein_data in protein_datas
    # ])


    # all_accessions = sorted(set(protein_family_to_accession_df["accession"]))



    # accession_to_chembl_target_filename = "accession_to_chembl_target.json"

    # accession_to_chembl_target = load_json(accession_to_chembl_target_filename)

    # all_accessions = [ 
    #     accession 
    #     for accession in all_accessions
    #     if accession not in accession_to_chembl_target
    # ]

    # num_accessions = len(all_accessions)
    # chunk_size = 100

    # n_chunks = num_accessions // chunk_size + 1
    # for chunk_num in range(n_chunks):
    #     chunk_accessions = all_accessions[chunk_num * chunk_size : (chunk_num+1)*chunk_size]
    #     if len(chunk_accessions) == 0:
    #         continue


    #     chunk_accession_to_chembl_target = map_uniprot_accession_to_chembl_id(
    #         chunk_accessions,
    #         human_only=False,
    #     )

    #     accession_to_chembl_target.update(chunk_accession_to_chembl_target)

    # print (len(all_accessions), len(accession_to_chembl_target))

    # write_json(accession_to_chembl_target, accession_to_chembl_target_filename)

    # accession_to_chembl_target_df = pd.DataFrame([
    #     {
    #         "accession": accession,
    #         "chembl_target_id": chembl_target[0]
    #     }
    #     for accession, chembl_target in accession_to_chembl_target.items()
    # ])

    
    # protein_type_to_family_df.to_csv("protein_type_to_family.csv")
    # protein_family_to_accession_df.to_csv("protein_family_to_accession.csv")
    # accession_to_chembl_target_df.to_csv("accession_to_chembl_target.csv")

    # all_data_df = protein_type_to_family_df\
    # .merge(protein_family_to_accession_df, on='protein_family', how='inner', )\
    # .merge(accession_to_chembl_target_df, on="accession", how="inner")

    # all_data_df.to_csv("all_grouped_accessions.csv")

    # homo_sapiens_df = all_data_df.loc[all_data_df["organism_scientific"]=="Homo sapiens"]
    # homo_sapiens_df.to_csv("all_grouped_homo_sapiens_accessions.csv")
