if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os, sys

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed


from cobdock.docking.vina.reverse_vina import execute_reverse_docking_vina
from cobdock.docking.galaxydock.reverse_galaxydock import execute_reverse_docking_galaxydock
from cobdock.docking.plants.reverse_plants import execute_reverse_docking_plants
from cobdock.docking.zdock.reverse_zdock import execute_reverse_docking_zdock

from utils.io.io_utils import write_json, load_json

def compute_closest_pockets(
    algorithm_name:str,
    source_data: dict,
    target_data: dict,
    is_molecular_docking_program: bool, # switch for whether or not to consider ligands
    verbose: bool = False,
    ):

    if verbose:
        print ("Computing closest poses/pockets to each voxel for algorithm", algorithm_name)
        print ("Is molecular docking program:", is_molecular_docking_program)

    pocket_minimum_pairwise_distances = {}

    # iterate over ligands 
    for ligand_id, source_data_ligand in source_data.items():
        
        if ligand_id not in pocket_minimum_pairwise_distances:
            pocket_minimum_pairwise_distances[ligand_id] = {}

        if is_molecular_docking_program:
            
            if ligand_id not in target_data:
                assert False, ("MISSING LIGAND", ligand_id)
                continue # TODO
            
            target_data_ligand = target_data[ligand_id]
        else:
            target_data_ligand = target_data

        # iterate over targets
        for accession, source_data_ligand_accession in source_data_ligand.items():

            if accession not in target_data_ligand:
                assert False, ("MISSING ACCESSION", accession)
                continue # TODO
            
            target_data_ligand_accession = target_data_ligand[accession]

            if accession not in pocket_minimum_pairwise_distances[ligand_id]:
                pocket_minimum_pairwise_distances[ligand_id][accession] = {}
                
            # iterate over crystal structures
            for pdb_id, source_data_ligand_accession_pdb_id in source_data_ligand_accession.items():
                
                if pdb_id not in target_data_ligand_accession:
                    assert False, ("MISSING PDBID", pdb_id)
                    continue # TODO
                    
                target_data_ligand_accession_pdb_id = target_data_ligand_accession[pdb_id]
                    
                if pdb_id not in pocket_minimum_pairwise_distances[ligand_id][accession]:
                    pocket_minimum_pairwise_distances[ligand_id][accession][pdb_id] = {}
                
                sorted_source_pose_ids = sorted(source_data_ligand_accession_pdb_id)
                sorted_target_pose_ids = sorted(target_data_ligand_accession_pdb_id)

                if len(sorted_target_pose_ids) == 0:
                    # no poses found for ligand, target pair 
                    raise Exception("NO POSES!", ligand_id, accession, pdb_id) # TODO

                # handle missing target data
                if len(sorted_target_pose_ids) == 1 and (sorted_target_pose_ids[0] == "null" or sorted_target_pose_ids[0] is None): 
                    selected_target_pose_id = sorted_target_pose_ids[0]

                    closest_pocket_data = {
                        "min_pose_id": selected_target_pose_id,
                        "distance": np.nan,
                        **target_data_ligand_accession_pdb_id[selected_target_pose_id] # return all data
                    }

                    # assign to all source_poses
                    for source_pose_id in sorted_source_pose_ids:

                        pocket_minimum_pairwise_distances[ligand_id][accession][pdb_id][source_pose_id] = closest_pocket_data

                else: # target poses were found
                
                    source_poses_as_array = np.array([
                        [
                            source_data_ligand_accession_pdb_id[source_pose_id][key]
                            for key in ("center_x", "center_y", "center_z")
                        ]
                        for source_pose_id in sorted_source_pose_ids
                    ])
                    
                    target_poses_as_array = np.array([
                        [
                            target_data_ligand_accession_pdb_id[target_pose_id][key]
                            for key in ("center_x", "center_y", "center_z")
                        ]
                        for target_pose_id in sorted_target_pose_ids
                    ])
                    
                    # shape is n_source_poses, n_target_poses
                    dists = np.array([
                        [
                            (
                                np.linalg.norm(source_pose - target_pose)
                                # if not (source_pose==None).any() and not (target_pose==None).any()
                                if not np.isnan(source_pose).any() and not np.isnan(target_pose).any()
                                else np.nan # missing value
                            )
                            for target_pose in target_poses_as_array
                        ]
                        for source_pose in source_poses_as_array
                    ])
                    assert dists.shape[0] == source_poses_as_array.shape[0]
                    assert dists.shape[1] == target_poses_as_array.shape[0]

                    min_dist_ids = dists.argmin(axis=1, )
                    
                    for i, (source_pose_id, min_dist_id) in enumerate(zip(sorted_source_pose_ids, min_dist_ids)):
                        min_dist = dists[i, min_dist_id]
                        min_pose_id = sorted_target_pose_ids[min_dist_id]
                        
                        closest_pocket_data = {
                            "min_pose_id": min_pose_id,
                            "distance": min_dist,
                            **target_data_ligand_accession_pdb_id[min_pose_id] # return all data
                        }

                        pocket_minimum_pairwise_distances[ligand_id][accession][pdb_id][source_pose_id] = closest_pocket_data
                    
    return pocket_minimum_pairwise_distances

def assign_pose_pocket_to_closest_voxel(
    all_ligand_target_voxels: dict,
    computed_poses: dict,
    attribute_name: str,
    is_molecule_docking_program: bool = True,
    verbose: bool = True,
    ):

    count_attribute_name = f"number_{attribute_name}"

    attribute_name = "sampled_pose_" + attribute_name
    count_attribute_name = "sampled_pose_" + count_attribute_name

    # return this 
    voxel_pose_pockets = {}

    if verbose:
        print ("Assigning all", attribute_name, "points to their closest voxel")
        print ("Attribute name of count is", count_attribute_name)
        print ("Is molecular docking program:", is_molecule_docking_program)

    # set counts to 0
    for ligand_id in all_ligand_target_voxels:

        if ligand_id not in voxel_pose_pockets:
            voxel_pose_pockets[ligand_id] = {}

        # iterate over targets
        for accession in all_ligand_target_voxels[ligand_id]:

            if accession not in voxel_pose_pockets[ligand_id]:
                voxel_pose_pockets[ligand_id][accession] = {}

            # iterate over crystal structures 
            for pdb_id in all_ligand_target_voxels[ligand_id][accession]:

                if pdb_id not in voxel_pose_pockets[ligand_id][accession]:
                    voxel_pose_pockets[ligand_id][accession][pdb_id] = {}

                # iterate over voxels 
                for voxel_id in all_ligand_target_voxels[ligand_id][accession][pdb_id]:
                    # initialise list of pose ids and count
                    voxel_pose_pockets[ligand_id][accession][pdb_id][voxel_id] = {
                        attribute_name: [],
                        count_attribute_name: 0,
                    }

    # begin collation
    # iterate over ligands
    for ligand_id, all_voxels_for_ligand in all_ligand_target_voxels.items():

        if is_molecule_docking_program:
            
            if ligand_id not in computed_poses:
                raise Exception("Missing ligand:", ligand_id)
                continue # TODO
            
            computed_poses_ligand = computed_poses[ligand_id]
        else:
            computed_poses_ligand = computed_poses

        # iterate over targets 
        for accession, all_voxels_ligand_accession in all_voxels_for_ligand.items():

            if accession not in computed_poses_ligand:
                print ("missing target", accession)
                raise Exception
                continue

            computed_poses_ligand_accession = computed_poses_ligand[accession]
            
            # iterate over crystal structures 
            for pdb_id, all_voxels_ligand_accession_pdb_id in all_voxels_ligand_accession.items():

                if pdb_id not in computed_poses_ligand_accession:
                    print ("missing crystal structure", pdb_id)
                    raise Exception
                    continue

                computed_poses_ligand_accession_pdb_id = computed_poses_ligand_accession[pdb_id]

                sorted_voxel_ids = sorted(all_voxels_ligand_accession_pdb_id)
                sorted_computed_pose_ids = sorted(computed_poses_ligand_accession_pdb_id)

                num_voxels = len(sorted_voxel_ids)
                num_poses = len(sorted_computed_pose_ids)

                if num_voxels == 0:
                    # no poses found for ligand, target pair 
                    raise Exception("NO POSES!", ligand_id, accession, pdb_id) # TODO

                if None in sorted_voxel_ids:
                    raise Exception(sorted_voxel_ids)
                if None in sorted_computed_pose_ids or "null" in sorted_computed_pose_ids:
                    # skip assigning computed poses for fail
                    continue
                
                # build array of voxel locations 
                voxels_as_array = np.array([
                    [
                        all_voxels_ligand_accession_pdb_id[sorted_voxel_id][key]
                        for key in ("center_x", "center_y", "center_z")
                    ]
                    for sorted_voxel_id in sorted_voxel_ids
                ])
                
                # build array of pose/pocket locations 
                computed_pose_locations_as_array = np.array([
                    [
                        computed_poses_ligand_accession_pdb_id[sorted_pose_id][key]
                        for key in ("center_x", "center_y", "center_z")
                    ]
                    for sorted_pose_id in sorted_computed_pose_ids
                ])

                # compute distances between all poses/pockets and all voxels
                # shape = (n_voxels, n_poses/n_pockets)
                dists = np.array([
                    [
                        (
                            np.linalg.norm(voxel - computed_pose_location)
                            # if not (sampled_point==None).any() and not (computed_pose_location==None).any()
                            if not np.isnan(voxel).any() and not np.isnan(computed_pose_location).any()
                            else np.nan
                        )
                        for computed_pose_location in computed_pose_locations_as_array
                    ]
                    for voxel in voxels_as_array
                ])
                assert dists.shape[0] == num_voxels
                assert dists.shape[1] == num_poses
                
                # minimum distance for each pocket/pose
                min_dist_ids = dists.argmin(axis=0, )
                assert min_dist_ids.shape[0] == num_poses

                # iterate over pairs of pose_pocket_index, closest_voxel_index
                for i, min_dist_id in enumerate(min_dist_ids):
                    # map closest voxel index to voxel id
                    min_dist_voxel_id = sorted_voxel_ids[min_dist_id]
                    # get id of pose
                    computed_pose_id = sorted_computed_pose_ids[i] 
                    # assert computed_pose_id not in all_voxels_ligand_accession_pdb_id[min_dist_voxel_id][attribute_name], (computed_pose_id, ligand_id)
                    voxel_pose_pockets[ligand_id][accession][pdb_id][min_dist_voxel_id][attribute_name].append(computed_pose_id)
                    # update count for min_dist_id_sampled_point_name
                    voxel_pose_pockets[ligand_id][accession][pdb_id][min_dist_voxel_id][count_attribute_name] += 1

    return voxel_pose_pockets

def extract_target_data_from_ligands_to_targets_data_structure(
    ligands_to_targets: dict,
    data_filename: str,
    data_key: str,
    verbose: bool = True,
    ):

    # extract data about unique from ligands_to_targets dictionary
    # the data is keyed using data_key

    if verbose:
        print ("Extracting data from prepared targets dictionary")
        print ("Extracting data key", data_key)
        print ("Extracting to file", data_filename)
    
    target_data = {}

    for ligand in ligands_to_targets:
        accessions_for_ligand =  ligands_to_targets[ligand]["prepared_targets"]
        for accession in accessions_for_ligand:
            
            if accession not in target_data:
                target_data[accession] = {}
            for pdb_id in accessions_for_ligand[accession]:
                if pdb_id in target_data[accession]:
                    continue # skip target if already seen 
                target_data[accession][pdb_id] = { # copy
                    k: v 
                    for k, v in accessions_for_ligand[accession][pdb_id][data_key].items()
                }


    # handle missing data
    for accession, accession_pdb_ids in target_data.items():
        for pdb_id, pdb_id_data in accession_pdb_ids.items():
            if len(pdb_id_data) == 0:
                pdb_id_data[None] = {
                    f"center_{k}": np.nan 
                    for k in ("x", "y", "z")
                }

    # write to file
    write_json(target_data, data_filename, verbose=verbose)

    return target_data

def collate_all_data(
    ligands_to_targets: dict,
    output_dir: str,

    commercial_use_only: bool,

    all_target_natural_ligands_filename: str,
    fpocket_data_filename: str,
    p2rank_data_filename: str,
    vina_data_filename: str,
    galaxydock_data_filename: str,
    plants_data_filename: str,
    zdock_data_filename: str,

    maximum_bounding_box_volume: float = 100**3,
    skip_if_complete: bool = True,
    delete_output_directory: bool = False,
    compute_rmsd_with_submitted_ligand: bool = True,
    num_complexes: int = 1,
    verbose: bool = True,
    ):

    if verbose:
        print ("Running docking and collating all data to", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # get voxel locations to serve as base
    all_ligand_target_voxels = {}
    for ligand_id in ligands_to_targets:
        if ligand_id not in all_ligand_target_voxels:
            all_ligand_target_voxels[ligand_id] = {}
        
        accessions_for_ligand = ligands_to_targets[ligand_id]["prepared_targets"]
        for accession in accessions_for_ligand:
            if accession not in all_ligand_target_voxels[ligand_id]:
                all_ligand_target_voxels[ligand_id][accession] = {}
            for pdb_id in accessions_for_ligand[accession]:
                if pdb_id not in all_ligand_target_voxels[ligand_id][accession]:
                    all_ligand_target_voxels[ligand_id][accession][pdb_id] = {}
                for voxel_location_id, sampled_location in enumerate(accessions_for_ligand[accession][pdb_id]["voxel_locations"]):
                    all_ligand_target_voxels[ligand_id][accession][pdb_id][voxel_location_id + 1] = sampled_location

    # Begin docking and pose/pocket collation

    # extract all natural ligands 
    all_target_natural_ligands_data = extract_target_data_from_ligands_to_targets_data_structure(
        ligands_to_targets=ligands_to_targets,
        data_filename=all_target_natural_ligands_filename,
        data_key="natural_ligands",
        verbose=verbose,
    )
    
    # fpocket 
    fpocket_data = extract_target_data_from_ligands_to_targets_data_structure(
        ligands_to_targets=ligands_to_targets,
        data_filename=fpocket_data_filename,
        data_key="fpocket_old",
        verbose=verbose,
    )

    # p2rank
    p2rank_data = extract_target_data_from_ligands_to_targets_data_structure(
        ligands_to_targets=ligands_to_targets,
        data_filename=p2rank_data_filename,
        data_key="p2rank",
        verbose=verbose,
    )

    # begin docking execution and data collation into dictionaries here
    vina_data = execute_reverse_docking_vina(
        ligands_to_targets=ligands_to_targets,
        output_dir=output_dir,
        collated_data_filename=vina_data_filename,
        maximum_bounding_box_volume=maximum_bounding_box_volume,
        skip_if_complete=skip_if_complete,
        delete_output_directory=delete_output_directory,
        compute_rmsd_with_submitted_ligand=compute_rmsd_with_submitted_ligand,
        num_complexes=num_complexes,
        verbose=verbose,
    )

    data_to_collate = [
        (fpocket_data, "fpocket", ),
        (p2rank_data, "p2rank", ),
        (all_target_natural_ligands_data, "natural_ligand", ),
        (vina_data, "vina", ),
        # (galaxydock_data, "galaxydock",),
        # (plants_data, "plants", ),
        # (zdock_data, "zdock", ),
    ]

    if not commercial_use_only:

        galaxydock_data = execute_reverse_docking_galaxydock(
            ligands_to_targets=ligands_to_targets,
            output_dir=output_dir,
            collated_data_filename=galaxydock_data_filename,
            maximum_bounding_box_volume=maximum_bounding_box_volume,
            skip_if_complete=skip_if_complete,
            delete_output_directory=delete_output_directory,
            compute_rmsd_with_submitted_ligand=compute_rmsd_with_submitted_ligand,
            num_complexes=num_complexes,
            verbose=verbose,
        )

        plants_data = execute_reverse_docking_plants(
            ligands_to_targets=ligands_to_targets,
            output_dir=output_dir,
            collated_data_filename=plants_data_filename,
            maximum_bounding_box_volume=maximum_bounding_box_volume,
            skip_if_complete=skip_if_complete,
            delete_output_directory=delete_output_directory,
            compute_rmsd_with_submitted_ligand=compute_rmsd_with_submitted_ligand,
            num_complexes=num_complexes,
            verbose=verbose,
        )

        zdock_data = execute_reverse_docking_zdock(
            ligands_to_targets=ligands_to_targets,
            output_dir=output_dir,
            collated_data_filename=zdock_data_filename,
            maximum_bounding_box_volume=maximum_bounding_box_volume,
            skip_if_complete=skip_if_complete,
            delete_output_directory=delete_output_directory,
            compute_rmsd_with_submitted_ligand=compute_rmsd_with_submitted_ligand,
            num_complexes=num_complexes,
            verbose=verbose,
        )

        data_to_collate.extend(
            [
                (galaxydock_data, "galaxydock",),
                (plants_data, "plants", ),
                (zdock_data, "zdock", ),
            ]
        )

  
    #######################################################################
    
    # begin identification of closest pockets/poses for each voxel
    # assign pose/pocket to each over
    # also, count number of poses/pockets in voxel


    closest_poses_pocket_for_each_voxel = {}
    all_poses_pocket_for_each_voxel = {}

    collation_n_proc = 5



    with ProcessPoolExecutor(max_workers=collation_n_proc) as p:

        running_tasks = {}

        for algorithm_pockets_poses, algorithm_name, in data_to_collate:
            is_molecular_docking_program = algorithm_name not in {"fpocket", "p2rank", "natural_ligand"}

            task = p.submit(
                compute_closest_pockets,
                algorithm_name=algorithm_name,
                source_data=all_ligand_target_voxels,
                target_data=algorithm_pockets_poses,
                is_molecular_docking_program=is_molecular_docking_program,
                verbose=verbose,
            )
            
            running_tasks[task] = {
                "algorithm_name": algorithm_name,
                "task_name": "closest",
            }

            # also update voxels with counts of all poses/pockets inside the voxel
            task = p.submit(
                assign_pose_pocket_to_closest_voxel,
                all_ligand_target_voxels=all_ligand_target_voxels,
                computed_poses=algorithm_pockets_poses,
                attribute_name=f"{algorithm_name}_poses_at_location",
                is_molecule_docking_program=is_molecular_docking_program,
                verbose=verbose,
            )

            running_tasks[task] = {
                "algorithm_name": algorithm_name,
                "task_name": "all",
            }
        
        for running_task in as_completed(running_tasks):

            task_data = running_tasks[running_task]
            algorithm_name = task_data["algorithm_name"]
            task_name = task_data["task_name"]
            task_result = running_task.result()

            if task_name == "closest":
                closest_poses_pocket_for_each_voxel[algorithm_name] = task_result
            elif task_name == "all":
                all_poses_pocket_for_each_voxel[algorithm_name] = task_result

            del running_tasks[running_task]


    ####################################################################################################################
    #############                         Write All Results to file                     ############################ 
    ####################################################################################################################

   
    all_aggregated_data_output_dir = os.path.join(output_dir, "pairs")
    os.makedirs(all_aggregated_data_output_dir, exist_ok=True)

    if verbose:
        print ("All data aggregation complete, writing voxel features for all pairs to file")
        print ("Writing to", all_aggregated_data_output_dir)

    # remove any box further away than this from at LEAST ONE identified pose
    # TODO: set threshold when model is trained
    # max_pose_distance = None 
    max_pose_distance = 15

    # iterate over ligands and construct dataframe
    for ligand_id, ligand_voxels in all_ligand_target_voxels.items():

        # iterate over accessions
        for accession, ligand_accession_voxels in ligand_voxels.items():

            # iterate over all targets for accession
            for pdb_id, ligand_accession_pdb_id_voxels in ligand_accession_voxels.items():

                # unique ID for pair
                pair_id = f"{ligand_id}_{pdb_id}"

                all_voxel_data_for_pair_filename = os.path.join(all_aggregated_data_output_dir, f"{pair_id}.parquet")

                # initialise list of data for all voxels for current crystal structure
                all_voxel_data_for_pair = []

                # iterate over all voxels 
                for voxel_id, voxel_data in ligand_accession_pdb_id_voxels.items():

                    # initialise aggrgated data for current voxel
                    all_aggregated_voxel_data = {
                        "ligand_target_voxel_id": f"{pair_id}_{voxel_id}",
                        "ligand_id": ligand_id,
                        "accession": accession,
                        "pdb_id": pdb_id,
                        "voxel_id": voxel_id,
                        # add voxel data (location/size/etc.)
                        **{
                            f"voxel_{k}": v
                            for k, v in voxel_data.items()
                             if not isinstance(v, list) and not isinstance(v, set) and not isinstance(v, dict)
                        },
                    }

                    # iterative over all algorithms and write data
                    for algorithm_name in closest_poses_pocket_for_each_voxel:
                        
                        # closest pose/pocket to voxel
                        closest_pose_pocket_for_current_voxel = closest_poses_pocket_for_each_voxel[algorithm_name][ligand_id][accession][pdb_id][voxel_id]
                        # remove lists/dicts/sets
                        closest_pose_pocket_for_current_voxel = {
                            f"{algorithm_name}_{k}": (
                                np.nan if v in {None, "null"} 
                                else v
                            ) 
                            for k, v in closest_pose_pocket_for_current_voxel.items()
                            if not isinstance(v, list) and not isinstance(v, set) and not isinstance(v, dict)
                            and not k.endswith("filename")
                        }
                        # update all_aggregated_voxel_data
                        all_aggregated_voxel_data.update(closest_pose_pocket_for_current_voxel)


                        # all poses/pockets inside voxel
                        all_poses_pockets_inside_current_voxel = all_poses_pocket_for_each_voxel[algorithm_name][ligand_id][accession][pdb_id][voxel_id]
                        # update all_aggregated_voxel_data
                        all_aggregated_voxel_data.update(all_poses_pockets_inside_current_voxel)

                    # add "selected pocket" for current voxel
                    fpocket_distance = all_aggregated_voxel_data["fpocket_distance"]
                    p2rank_distance = all_aggregated_voxel_data["p2rank_distance"]

                    # handle both programs failing
                    if pd.isnull(fpocket_distance) and pd.isnull(p2rank_distance):
                        # use voxel location 
                        selected_pocket_program = "voxel"
                        selected_pocket_min_pose_id = np.nan
                        selected_pocket_distance = np.nan
                    else:
                        # missing fpocket or p2rank is closer
                        # p2rank_distance is null will not pass this check
                        if pd.isnull(fpocket_distance) or p2rank_distance < fpocket_distance:
                            selected_pocket_program = "p2rank"
                        else:
                            selected_pocket_program = "fpocket"
                        selected_pocket_min_pose_id = int(all_aggregated_voxel_data[f"{selected_pocket_program}_min_pose_id"])
                        selected_pocket_distance = all_aggregated_voxel_data[f"{selected_pocket_program}_distance"]


                    selected_pocket_program_data = {
                        "selected_pocket_program": selected_pocket_program,
                        "selected_pocket_min_pose_id": selected_pocket_min_pose_id,
                        "selected_pocket_distance": selected_pocket_distance,
                        # add location of selected pocket
                        **{
                            f"selected_pocket_{k}": all_aggregated_voxel_data[f"{selected_pocket_program}_{k}"]
                            for k in (
                                "center_x",
                                "center_y",
                                "center_z",
                                "size_x",
                                "size_y",
                                "size_z",
                            )
                        }
                    }
                    # update all_aggregated_voxel_data with data about selected pocket location 
                    all_aggregated_voxel_data.update(selected_pocket_program_data)

                    # skip row if further away than max_pose_distance for ALL programs
                    # accept row if within max_pose_distance from ANY program
                    if max_pose_distance is not None:
                        within_max_pose_distance = False 
                        for algorithm_name in closest_poses_pocket_for_each_voxel:
                            if all_aggregated_voxel_data[f"{algorithm_name}_distance"] <= max_pose_distance:
                                within_max_pose_distance = True 
                                break
                        if not within_max_pose_distance:
                            # do not add pose_data
                            continue # skip adding current voxel data for current pair 

                    # all_data_rows.append(pose_data)
                    all_voxel_data_for_pair.append(all_aggregated_voxel_data)

                # write csv file for pair
                all_voxel_data_for_pair = pd.DataFrame(all_voxel_data_for_pair)

                all_voxel_data_for_pair.set_index("ligand_target_voxel_id", drop=True, inplace=True)
                if verbose:
                    print ("writing pair data to", all_voxel_data_for_pair_filename, ligand_id, accession, pdb_id)
                if all_voxel_data_for_pair_filename.endswith(".parquet"):
                    all_voxel_data_for_pair.to_parquet(all_voxel_data_for_pair_filename)
                else:
                    all_voxel_data_for_pair.to_csv(all_voxel_data_for_pair_filename)

                # add data collated_data_filename to ligands_to_targets
                ligands_to_targets[ligand_id]["prepared_targets"][accession][pdb_id]["collated_data_filename"] = all_voxel_data_for_pair_filename

    return ligands_to_targets



if __name__== '__main__':
    pass