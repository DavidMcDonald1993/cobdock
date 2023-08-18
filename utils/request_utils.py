
if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import urllib.parse as urlparse

import os
import json 

import requests
from time import sleep

from utils.io.io_utils import write_json

def prepare_data_for_ajax(
    data: dict,
    ):
    if not isinstance(data, dict):
        return data
    return {
        k: json.dumps(v)
        for k, v in data.items()
    }

def get_request_handle(
    request,
    copy: bool = False,
    as_dict: bool = False,
    ):
    """Obtain the handle of a request
    Parameters
    ----------
    request : HttpRequest
        An instance of HttpRequest
    copy : bool,
        Flag which if positive will return a copy of the request handle dict, by default False
    Returns
    -------
    object
        Handle on GET/POST arguments
    Raises
    ------
    NotImplementedError
        Thrown if request type is not in {"GET", "POST"}
    """
    request_method = request.method

    if request_method == "GET":
        request_handle = request.GET
    elif request_method == "POST":
        request_handle = request.POST
    else:
        raise NotImplementedError

    if copy:
        return request_handle.copy()
    if as_dict:
        return {
            k: v
            for k, v in request_handle.items()
        }
    return request_handle

def handle_json_request(
    request, 
    param_name: str, 
    default: object = None,
    ):
    """Parse a JSON encoded HTTP request value

    Parameters
    ----------
    request : HttpRequest
        The request
    param_name : str
        The name of the argument
    default : object, optional
        Default value, by default None

    Returns
    -------
    object
        The JSON-parsed value or None
    """
    r = get_request_handle(request)
    if param_name in r.keys():
        try:
            return json.loads(r[param_name])
        except json.JSONDecodeError:
            return None # value is not JSON encoded
    return default

def handle_file_request(
    request, 
    param_name: str,
    ):
    """Obtain a file value from a HTTP request

    Parameters
    ----------
    request : HttpRequest
        The request
    param_name : str
        The name of the HTTP argument

    Returns
    -------
    File
        The file or None if `param_name` is not an argument
    """
    if param_name not in request.FILES:
        return None 
    return request.FILES[param_name]

def get_http_parameter_value(
    request, 
    param_name: str, 
    default: object = None, 
    param_type: object = str,
    handle_json: bool = True,
    ):
    """Return the value for a given HTTP argument/parameter

    Parameters
    ----------
    request : HttpRequest
        The request
    param_name : str
        The name of the HTTP argument
    default : object, optional
        A default value, by default None
    param_type : object, optional
        The type of the parameter value, by default str
    handle_json : bool, optional
        Flag to accept a JSON encoded value, by default True

    Returns
    -------
    object
        An object corresponding to the value of the HTTPRequest argument
    """

    if handle_json:
        json_request = handle_json_request(request, param_name, default=default)
        if json_request is not None:
            return json_request
            
    r = get_request_handle(request)
    
    if param_name in r.keys():
        value = r[param_name]
        if value == "None":
            return None # handle explicit None argument for max_actives 
        if value == "":
            return default
        try:
            return param_type(value)
        except:
            pass
    return default

def handle_checkbox_request(
    request, 
    param_name: str, 
    default: object = False):
    """Parse value from a checkbox in a form.
    Returns `default` if the argument is missing / was not checked

    Parameters
    ----------
    request : HttpRequest
        The request
    param_name : str
        The name of the argument
    default : object, optional
        Default value, by default False

    Returns
    -------
    bool
        True if the argument is in the request else `default`
    """

    r = get_request_handle(request)

    if param_name in r.keys():
        return True
    return default

def handle_select_multiple_request(
    request, 
    param_name: str, 
    default: object = None):
    """Parse value from multiple-select in a form.

    Parameters
    ----------
    request : HttpRequest
        The request
    param_name : str
        The name of the argument
    default : object, optional
        Default value, by default None

    Returns
    -------
    list
        List of values for argument name `param_name`
    """
    json_request = handle_json_request(request, param_name, default=default)
    if json_request is not None:
        return json_request
    r = get_request_handle(request)
    for param_name in (param_name, f"{param_name}[]"):
        if param_name not in r.keys():
            continue
        args = [urlparse.unquote(a) for a in r.getlist(param_name)
            if a != ""]
        if len(args) == 0:
            args = default 
        return args
    return default

def make_get_request(
    url: str, 
    params: dict, 
    headers: dict = {},
    stream: bool = False,
    max_retries: int = 5,
    sleep_duration: int = 100,
    allowed_status_codes: set = {200,},
    verbose: bool = False,
    ):
    """A simple function to make a GET request, with retrying if an error is encountered.

    Parameters
    ----------
    url : str
        The URL to make the request to
    params : dict
        A dictionary of HTTP arguments
    headers : dict, optional
        HTTP headers, by default {}
    stream : bool, optional
        Flag to stream response, required for file response, by default False
    max_retries : int, optional
        Maximum number of retries before failure, by default 5
    sleep_duration : int, optional
        Number of seconds to sleep between retries, by default 100
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    HTTPResponse
        The response received, or None if a failure occured

    """
    for retry_num in range(max_retries):
        try:
            if retry_num > 0:
                sleep(sleep_duration)
            if verbose:
                print ("Making GET request to URL", url, "and params", list(params))
            response = requests.get(url, params=params, stream=stream, headers=headers)
            status_code = response.status_code
            if verbose:
                print ("Obtained response with status", status_code)
            if status_code not in allowed_status_codes:
                raise Exception("Bad status code:", status_code)
            return response
        except Exception as e:
            if verbose:
                print ("Retry", retry_num+1, "to URL", url, "failed with exception", e, "sleeping for", sleep_duration, "seconds.")

    return None # Fail

def make_post_request(
    url: str, 
    params: dict, 
    headers: dict = {},
    filelist = None, 
    stream: bool = False,
    max_retries: int = 5,
    sleep_duration: int = 100,
    verbose: bool = False,
    ):
    """_summary_

    Parameters
    ----------
    url : str
        The URL to make the request to
    params : dict
        A dictionary of HTTP arguments
    filelist : dict, optional
        Dictionary mapping argument name to filepath, may also be a list of filepaths 
        in which case, the basename of the file will be used as the argument name, by default None
    stream : bool, optional
        Flag to stream response, required for file response, by default False
    max_retries : int, optional
        Maximum number of retries before failure, by default 5
    sleep_duration : int, optional
        Number of seconds to sleep between retries, by default 100
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    HTTPResponse
        The response received, or None if a failure occured
    """
    
    if filelist is not None:
        if not isinstance(filelist, dict):
            # use file basename as http parameter name
            if verbose:
                print ("Filelist is a list, using file basenames as HTTP argument keys")
            filelist = { 
                os.path.basename(filepath): filepath 
                for filepath in filelist
            }
        
        # fail state
        if not isinstance(filelist, dict):
            print ("Filelist is not a dictionary", filelist)
            return None
        
    successful = False
    files_to_send = {} 

    for retry_num in range(max_retries):
        if retry_num > 0:
            sleep(sleep_duration)
        # build files_to_send dictionary
        if filelist is not None:
            for file_parameter_name, file_path in filelist.items():
                if verbose:
                    print ("Adding file", file_path, "as argument", file_parameter_name, "to request")
                if not os.path.exists(file_path):
                    print (file_path, "does not exist, skipping")
                    continue
                files_to_send[file_parameter_name] = open(file_path, "rb")

        try:
            if verbose:
                print ("Making POST request to URL", url)
            http_response = requests.post(url, data=params, files=files_to_send, stream=stream, headers=headers)
            successful = True
            break
        except Exception as e:
            if verbose:
                print ("Retry", retry_num+1, "to URL", url, "failed with exception", e, "sleeping for", sleep_duration, "seconds.")
            # close all handles and re-initialise dictionary
            for file_parameter_name in files_to_send:
                files_to_send[file_parameter_name].close()
            files_to_send = {} 

    # close all file handles
    for file_parameter_name in files_to_send:
        files_to_send[file_parameter_name].close()
    if not successful:
        return None # Fail
    return http_response 


def make_http_request(
    url: str, 
    params: dict, 
    filelist = None, 
    headers: dict = {},
    method: str = "GET", 
    stream: bool = False,
    max_retries: int = 5,
    sleep_duration: int = 100,
    verbose: bool = False,
    ):
    """Wrapper function to handle making requests with retries enabled.
    Currently only GET and POST requests are implemented.

    Parameters
    ----------
    url : str
        The URL to make the request to
    params : dict
        A dictionary of HTTP arguments
    filelist : dict, optional
        Dictionary mapping argument name to filepath, may also be a list of filepaths 
        in which case, the basename of the file will be used as the argument name, by default None
    headers : dict, optional
        HTTP headers, by default {}
    stream : bool, optional
        Flag to stream response, required for file response, by default False
    max_retries : int, optional
        Maximum number of retries before failure, by default 5
    sleep_duration : int, optional
        Number of seconds to sleep between retries, by default 100
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    HTTPResponse
        The response received, or None if a failure occured

    Raises
    ------
    NotImplementedError
        Thrown if a method other than "GET" or "POST" is given
    """

    method = method.upper()
    if verbose:
        print ("Making", method, "request to URL", url, "with params", list(params))
    if isinstance(params, dict):
        # json encode any lists or dicts?
        for param_name in params:
            param_value = params[param_name]
            if isinstance(param_value, list) or isinstance(param_value, dict):
                params[param_name] = json.dumps(param_value)
    # GET
    if method in {"get", "GET"}:
        return make_get_request(
            url=url,
            params=params,
            headers=headers,
            stream=stream,
            max_retries=max_retries,
            sleep_duration=sleep_duration,
            verbose=verbose,
        )
    # POST                    
    elif method in {"post", "POST"}:
        return make_post_request(
            url=url,
            params=params,
            headers=headers,
            filelist=filelist,
            stream=stream,
            max_retries=max_retries,
            sleep_duration=sleep_duration,
            verbose=verbose,
        )
    else:
        # HTTP request method not yet implemented
        raise NotImplementedError


if __name__ == "__main__":

    # url = "https://alphafold.ebi.ac.uk/files/AF-P09874-F1-model_v3.pdb"

    # response = make_http_request(
    #     url=url,
    #     params={},
    #     stream=False,
    #     verbose=True,
    # )

    # with open("AF-P09874-F1-model_v3.pdb", "w") as f:
    #     f.write(response.text)


    # params = load_json("aiengine-enrichment-analysis-input.json")

    # response = make_http_request(
    #     # url="https://app.npaiengine.com/api/uniprot-to-pdb",
    #     # url="http://localhost:8000/api/uniprot-to-pdb",
    #     # url="https://app.npaiengine.com/api/get-uniprot",
    #     # url="http://localhost:8000/api/get-uniprot",
    #     # url="http://localhost:8080/bioactivity-predict/",
    #     # url="http://localhost:8000/molecule/properties",
    #     # url="http://localhost:8000/enrichment/all",
    #     # url="http://localhost:8000/ai-docking/",
    #     # url="http://localhost:8080/ai-docking/",
    #     url="http://localhost:8000/determine-counterfactuals/",
    #     # params=params,
    #     # params={ # ai docking
    #     #      "ligands_to_targets": {
    #     #         "aspirin": {
    #     #             "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    #     #             "targets": ["1htp"]
    #     #         },
    #     #         "ibuprofen": {
    #     #             "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    #     #             "targets": ["1htp"]
    #     #         }
    #     #     },
    #     #     # "screening_modes": "virtual_screening",
    #     #     "screening_modes": "reverse_docking",
    #     #     # "screening_modes": ["virtual_screening", "reverse_docking"],
    #     #     "response_type": "json",
    #     # },
    #     # params={
    #     #     "supplied_mols": [
    #     #     # "molecule-smiles": [
    #     #          {
    #     #             "molecule_id": "aspirin",
    #     #             "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    #     #         },
    #     #         {
    #     #             "molecule_id": "ibuprofen",
    #     #             "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    #     #         }
    #     #     ],
    #     #     "max_target_rank": 10,
    #     #     # "target_types": ["SINGLE PROTEIN", "OTHER TARGET TYPES"],
    #     #     "organisms": ["Selected animals"],
    #     #     # "return_full_target_data": "False",
    #     #     # "accessions": [
    #     #     #     "P00533", 
    #     #     #     "P09874",
    #     #     # ],
    #     #     # "gene-names": "parp1",
    #     #     # "human-only": "True",
    #     #     # "include-pdb": "True",
    #     # },
    #     params={
    #         "molecule_smiles": [
    #             {
    #                 "molecule_id": "AR",
    #                 "smiles": "N#Cc1ncc(cc1C(F)(F)F)N2C(=O)C3(CCC3)N(C2=S)c4ccc(nc4)OC5CCNCC5",
    #             },
    #             {
    #                 "molecule_id": "aspirin",
    #                 "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    #             },
    #             {
    #                 "molecule_id": "ibuprofen",
    #                 "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    #             },
    #             {
    #                 "molecule_id": "paracetamol",
    #                 "smiles": "CC(=O)Nc1ccc(O)cc1",
    #             },
    #         ],
    #         "num_samples": 500,
    #     },
    #     method="POST",
    #     stream=True,
    #     verbose=True,
    # )

    # # aiengine_archive_filename = os.path.join(".", "aiengine_ai_blind_docking_response.zip")
    # # aiengine_archive_filename = os.path.join(".", "determine_counterfactuals_response.tar.gz")
    # # with open(aiengine_archive_filename, 'wb') as f:
    # #     f.write(response.raw.read())

    # json_response = json.loads(response.text)

    # from utils.uniprot_utils import ALL_UNIPROT_TARGETS, UNIPROT_TARGET_FIELD_ORDER

    # cleaned_response = {}
    # for molecule_id, molecule_accessions in json_response.items():
        
        
    #     molecule_records = []
    #     for accession in molecule_accessions:
    #         if accession in ALL_UNIPROT_TARGETS:
                
    #             accession_record = {
    #                 k: (
    #                     ALL_UNIPROT_TARGETS[accession][k]
    #                     if k in ALL_UNIPROT_TARGETS[accession]
    #                     else None
    #                 )
    #                 for k in UNIPROT_TARGET_FIELD_ORDER
    #             }
    #         else:
    #             accession_record = {
    #                 k: accession if k == "accession" else None 
    #                 for k in UNIPROT_TARGET_FIELD_ORDER
    #             }

    #         all_examples = molecule_accessions[accession]["all_examples"]
    #         selected_cfs = molecule_accessions[accession]["selected_cfs"]
    #         selected_fs = molecule_accessions[accession]["selected_fs"]

    #         predicted_score = all_examples[0]["p"]
    #         accession_record["predicted_score"] = predicted_score

    #         prediction = all_examples[0]["yhat"]
    #         accession_record["prediction"] = prediction

    #         accession_record["selected_counterfactuals"] = selected_cfs
    #         accession_record["selected_factuals"] = selected_fs

    #         molecule_records.append(accession_record)

    #     # sort by predicted_score
    #     molecule_records = sorted(molecule_records,
    #         key=lambda record: record["predicted_score"],
    #         reverse=True)

    #     cleaned_response[molecule_id] = molecule_records

    # try:
    #     write_json(json_response, "checkme.json")
    #     write_json(cleaned_response, "checkme2.json")
    # except Exception as e:
    #     print (response.text)
    #     print (e)


    # response = make_get_request(
    #     url="http://localhost:8000/natural_products/families",
    #     params={
    #         "accessions": json.dumps(["P09874", ]),
    #     },
    #     max_retries=1,
    #     verbose=True,
    # )

    # raise Exception(response.text)

    molecule_smiles  = [ 
        {
            "molecule_id": "aspirin",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        },
        {
            "molecule_id": "baicalein",
            "smiles": "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O",
        },
        {
            "molecule_id": "Olaparib",
            "smiles": "O=C1C2=CC=CC=C2C(CC3=CC(C(N4CCN(CC4)C(C5CC5)=O)=O)=C(C=C3)F)=NN1",
        }
    ]

    pdb_targets = ["1htp"]

    gene_names  = ['AKR1C2', 'AKR1C3', 'CA1', 'CA12', 'CA14', 'CA2', 'CA7', 'CA9', 'GP6', 'NAPRT', 'P2RY12', 'POLK', 'PTGS1', 'PTGS2', 'TBXAS1']

    response = make_http_request(
        # url="http://47.102.129.50:80/enrichment/genes",
        # url="http://localhost:8080/enrichment/cancers",
        # url="http://47.102.129.50:80/natural_products/gene_ontology_enrichment",
        # url="http://localhost:8080/natural_products/gene_ontology_enrichment",
        # url="http://localhost:8000/hit-optimisation",
        # url="http://localhost:8080/target-text-mining",
        # url="http://localhost:8080/bioavailability-predict/",
        # url="http://localhost:8080/natural_products/api/compounds/all_species",
        # url="http://localhost:8080/drug-synergy-prediction/",
        # url="http://47.102.129.50:80/natural_products/superkingdoms",
        # url="http://localhost:8000/natural_products/superkingdoms",
        # url="http://localhost:8000/natural_products/chembl_targets",
        # url="http://localhost:8000/natural_products/search",
        # url="http://localhost:8000/natural_products/single_protein_targets",
        # url="http://localhost:8000/natural_products/chembl_targets",
        # url="http://localhost:8000/natural_products/pathways",
        # url="http://localhost:8000/natural_products/drugs",
        # url="http://localhost:8000/natural_products/diseases",
        # url="http://localhost:8000/natural_products/species",
        # url="http://localhost:8000/natural_products/organisms",
        # url="http://localhost:8000/natural_products/search",
        # url="http://localhost:8000/natural_products/species_names",
        # url="http://localhost:8000/natural_products/scaffolds",
        # url="http://localhost:8000/natural_products/cluster",
        # url="http://localhost:8080/natural_products/np_pathways",
        # url="http://localhost:8000/natural_products/single_protein_targets/all",
        # url="http://localhost:8000/natural_products/single_protein_targets/info",
        # url="http://localhost:8000/natural_products/single_protein_targets/screen",
        # url="http://localhost:8000/natural_products/single_protein_targets/pathways",
        # url="http://localhost:8000/natural_products/single_protein_targets/drugs",
        # url="http://localhost:8000/natural_products/single_protein_targets/diseases",
        # url="http://localhost:8000/natural_products/diseases/all",
        # url="http://localhost:8080/natural_products/diseases/screen",
        # url="http://localhost:8080/natural_products/diseases/screen_species",
        url="http://localhost:8080/natural_products/diseases/screen_food",
        # url="http://localhost:8000/natural_products/diseases/drugs",
        # url="http://localhost:8000/natural_products/diseases/pathways",
        # url="http://localhost:8000/natural_products/drugs/all",
        # url="http://localhost:8000/natural_products/drugs/classes",
        # url="http://localhost:8000/natural_products/drugs/screen",
        # url="http://localhost:8000/natural_products/drugs/single_protein_targets",
        # url="http://localhost:8000/natural_products/drugs/diseases",
        # url="http://localhost:8000/natural_products/pathways/all",
        # url="http://localhost:8000/natural_products/pathways/single_protein_targets",
        # url="http://localhost:8000/natural_products/pathways/drugs",
        # url="http://localhost:8000/natural_products/pathways/diseases",
        # url="http://localhost:8000/natural_products/reactions/all",
        # url="http://localhost:8000/natural_products/reactions/screen",
        # url="http://localhost:8080/natural_products/herb_analysis",
        # url="http://localhost:8080/molecule/properties",
        # url="http://47.102.129.50:80/natural_products/herb_analysis",
        # url="http://47.102.129.50:80/np-classify/",
        # url="http://localhost:8000/natural_products/pathway_enrichment",
        # url="http://localhost:8000/natural_products/cancer_enrichment",
        # url="http://localhost:8000/natural_products/reactions/all",
        # url="http://localhost:8000/natural_products/pathway_enrichment",
        # url="http://localhost:8000/toxicity-predict/",
        # url="http://localhost:5000/prediction/",
        params={
            # "small_molecule_ids": [14060, 20239],
            # "log_p_lt": 3,
            # "log_p_gt": 2.9,
            # "molecular_weight_lt": 300,
            # "molecular_weight_gt": 295,
            # "gene-list": gene_names,
            # # "genus": "Homo",
            # "genus": "Aconitum",
            # "species": ["Aconitum carmichaelii", "Aconitum napellus", "Isodon serra"],
            # "small_molecule_filter": "lipinski_filter",
            "disease_names": ["Heart failure", "Neoplasms"],
            # "return_format": "graph",
            # "return_format": "cytoscape",
            # "return_format": "table",
            # "chinese_species_name": "乌头",
            # "molecule_ids": ["aspirin"],
            # "drug_pairs": [
            #     [
            #         {
            #             "molecule_id": "aspirin",
            #             "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            #         },
            #         {
            #             "molecule_id": "ibuprofen",
            #             "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            #         }
            #     ],
            # ],
            # "text": [
            #     "Withaferin-A is a withanolide, predominantly present in Ashwagandha (Withania somnifera). It has been shown to possess anticancer activity in a variety of human cancer cells in vitro and in vivo. Molecular mechanism of such cytotoxicity has not yet been completely understood. Withaferin-A and Withanone were earlier shown to activate p53 tumor suppressor and oxidative stress pathways in cancer cells. 2,3-dihydro-3beta-methoxy analogue of Withaferin-A (3betamWi-A) was shown to lack cytotoxicity and well tolerated at higher concentrations. It, on the other hand, protected normal cells against oxidative, chemical and UV stresses through induction of anti-stress and pro-survival signaling. We, in the present study, investigated the effect of Wi-A and 3betamWi-A on cell migration and metastasis signaling. Whereas Wi-A binds to vimentin and heterogeneous nuclear ribonucleoprotein K (hnRNP-K) with high efficacy and downregulates its effector proteins, MMPs and VEGF, involved in cancer cell metastasis, 3betamWi-A was ineffective. Consistently, Wi-A, and not 3betamWi-A, caused reduction in cytoskeleton proteins (Vimentin, N-Cadherin) and active protease (u-PA) that are essential for three key steps of cancer cell metastasis (EMT, increase in cell migration and invasion).",
            #     "Withaferin-A is a withanolide, predominantly present in Ashwagandha (Withania somnifera). It has been shown to possess anticancer activity in a variety of human cancer cells in vitro and in vivo. Molecular mechanism of such cytotoxicity has not yet been completely understood. Withaferin-A and Withanone were earlier shown to activate p53 tumor suppressor and oxidative stress pathways in cancer cells. 2,3-dihydro-3beta-methoxy analogue of Withaferin-A (3betamWi-A) was shown to lack cytotoxicity and well tolerated at higher concentrations. It, on the other hand, protected normal cells against oxidative, chemical and UV stresses through induction of anti-stress and pro-survival signaling. We, in the present study, investigated the effect of Wi-A and 3betamWi-A on cell migration and metastasis signaling. Whereas Wi-A binds to vimentin and heterogeneous nuclear ribonucleoprotein K (hnRNP-K) with high efficacy and downregulates its effector proteins, MMPs and VEGF, involved in cancer cell metastasis, 3betamWi-A was ineffective. Consistently, Wi-A, and not 3betamWi-A, caused reduction in cytoskeleton proteins (Vimentin, N-Cadherin) and active protease (u-PA) that are essential for three key steps of cancer cell metastasis (EMT, increase in cell migration and invasion).",
            #     "Withaferin-A is a withanolide, predominantly present in Ashwagandha (Withania somnifera). It has been shown to possess anticancer activity in a variety of human cancer cells in vitro and in vivo. Molecular mechanism of such cytotoxicity has not yet been completely understood. Withaferin-A and Withanone were earlier shown to activate p53 tumor suppressor and oxidative stress pathways in cancer cells. 2,3-dihydro-3beta-methoxy analogue of Withaferin-A (3betamWi-A) was shown to lack cytotoxicity and well tolerated at higher concentrations. It, on the other hand, protected normal cells against oxidative, chemical and UV stresses through induction of anti-stress and pro-survival signaling. We, in the present study, investigated the effect of Wi-A and 3betamWi-A on cell migration and metastasis signaling. Whereas Wi-A binds to vimentin and heterogeneous nuclear ribonucleoprotein K (hnRNP-K) with high efficacy and downregulates its effector proteins, MMPs and VEGF, involved in cancer cell metastasis, 3betamWi-A was ineffective. Consistently, Wi-A, and not 3betamWi-A, caused reduction in cytoskeleton proteins (Vimentin, N-Cadherin) and active protease (u-PA) that are essential for three key steps of cancer cell metastasis (EMT, increase in cell migration and invasion).",
            # ],
            # "molecules": molecule_smiles,
            # "chinese_only": True,
            # # "columns": ["small_molecule_id", "smiles"],
            # # "natural_product_ids": ["CNP0114535", ],
            # "diseases": ["Retinoblastoma", ],
            # # "language_code": "zh_Hans",
            # "accession": ["P09874"],
            # "herb_species_names": [
            #     # "Aconitum Carmichaelii", 
            #     "Aconitum Ferox", "Aconitum Taipeicum", "",
            # ],
            # # "herb_chinese_species_names": ["乌头", ],
            # "cluster_small_molecules_in_analysis_network": True,
            # "keep_entire_cluster_predictions_only": True,
            # # "target_types": "OTHER TARGET TYPES",
            # "target_types": ["PROTEIN COMPLEX", "CELL-LINE"],
            # # "pathways": ["G1 Phase"],
            # "drug_ids": ["D0C9XA", ], # legacy handling
            # "pathways": ["Respiratory electron transport",],
            # "reactions": ["β3-agonists bind ADRB3",],
            # "name_like": "gink",
            # "summarise": True,
            # "molecules": json.dumps(molecule_smiles),
            # "pdb_targets": json.dumps(pdb_targets),
            # "population_size": 50,
            # "number_of_generations": 2,
            # "n_proc": 23,
        },
        max_retries=1,
        method="POST",
        verbose=True,
    )

    write_json(json.loads(response.text), "checkme.json", verbose=True)
