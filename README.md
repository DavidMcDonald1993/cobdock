# Reference Implementation of COBDock algorithm
This readme file documents all of the required steps to run COBDock.

Note that code was implemented and tested on a Linux operating system only.

## How to set up environment
We have provided an Anaconda environment file for easy set up.
If you do not have Anaconda installed, you can get Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
Create the `cobdock-env` environment using the following command:
```bash
conda env create -n cobdock-env -f environment.yml
conda activate cobdock-env
```
Then, install additional packages with pip using the following command:
```bash
pip install pdb-tools scoria
pip install git+https://github.com/thelahunginjeet/kbutil
pip install git+https://github.com/thelahunginjeet/pyrankagg 
```

## Required external programs
Before running COBDock, one will first need to download and prepare all of the molecular docking and pocket identification algorithms.
Before downloading, prepare a `./bin` folder using the following commands:

```bash
mkdir bin
mkdir bin/docking
mkdir bin/location_analysis
```


### Vina 1.1.2
Vina 1.1.2 [1] can be downloaded from [here](https://vina.scripps.edu/downloads/).
Execute this command to create a `vina` subdirectory inside `./bin/docking`:
```bash
mkdir bin/docking/vina
```
and place the vina binary inside the newly created `vina` subdirectory. 

### GalaxyDock3
GalaxyDock3 [2] can be downloaded from [here](https://galaxy.seoklab.org/softwares/galaxydock.html).
Download the archive file and extract the `GalaxyDock3` directory into `./bin/docking`.

### PLANTS1.2
The PLANTS1.2 [3] and SPORES binaries are available [here](http://www.tcd.uni-konstanz.de/plants_download/).
Execute this command to create a `plants` subdirectory inside `./bin/docking`:
```bash
mkdir bin/docking/plants
```
and place both binaries inside the newly created `plants` subdirectory. 

### ZDOCK
The ZDOCK [4] binary is available [here](https://zdock.umassmed.edu/software/).
Also required are the following files: `create_lig`, `create.pl`, `mark_sur`, and `uniCHARMM`.
Execute this command to create a `zdock` subdirectory inside `./bin/docking`:
```bash
mkdir bin/docking/zdock
```
and place the ZDOCK binary and all of the suppliementary files inside the newly created `zdock` subdirectory. 

### fpocket4.0
Fpocket4.0 [5] has is included in the provided Anaconda environment.

### P2Rank 2.3.1
P2Rank 2.3.1 can be downloaded as an archive [here](https://github.com/rdk/p2rank/releases/download/2.3.1/p2rank_2.3.1.tar.gz).
Extract it to the `./bin/location_analysis` subdirectory so that `p2rank_2.3.1` can be found inside `./bin/location_analysis`.
Confirm that the `prank` binary can be found at `./bin/location_analysis/p2rank_2.3.1/prank`.


### MGLTools-1.5.6
We have provided MGLTools-1.5.6 at `./MGLTools-1.5.6`.

## How to run

### Configuring environment variables
Modify the `.env` file to set the maximum number of processes for:

1. Preparing targets 
2. Executing Vina 
3. Executing GalaxyDock
4. Executing PLANTS
5. Executing ZDOCK

We strongly advise setting a low number of processes for PLANTS since it is memory intensive.
This file can also be used to configure timeouts for all of the molecular docking programs (set to "1h" (one hour) by default). 
To set the timeout to 30 minutes, change the timeout to "30m".
To set the timeout to 10 seconds, change the timeout to "10s".


### Running the COBDock pipeline
Run the following command:

```bash
python cobdock/run_cobdock.py
```
to execute COBDock for aspirin and Uniprot target P23219. Crystal structures will be selected and prepared automatically. 
If the run is successful, ``COBDock successfully completed`` will be printed to the console.
Please allow some time for a run to complete as some molecular docking algoritms can take some time. 


### All input arguments explained
The primary function to run COBDock is `execute_cobdock` in the file `./cobdock/run_cobdock.py`.
This function has the following input arguments:

    * ligands_to_targets: a data structure mapping ligand IDs and structures to target lists.  
    * output_dir: directory to execute COBDock in. 
    * map_uniprot_to_pdb: boolean flag to indicate that the targets given in the ligands_to_targets data structure are Uniprot accession IDs. Must be set to `False` if the input targets are given as PDB IDs.
    * number_of_pdb: the number of crystal structures to select for each Uniprot target. Has no effect when map_uniprot_to_pdb is set to `False`.
    * num_top_pockets: the number of top-ranked pockets to dock into.
    * num_poses: the number of poses to generate at each of the top ranked pockets. The total number of poses will be num_top_pockets * num_poses.
    * num_complexes: the number of top poses to convert into complexes by combining with the structure of the target.


### Configuring future runs

Edit the `ligands_to_targets` data structure defined in `./cobdock/run_cobdock.py` to configure all of the pairs that will be input into COBDock.
SMILES strings must be provided for all input ligands. 
Targets must either be all:

1. PDB IDs of crystal structure(s) of interest. Optionally, a chain can be appended with an underscore: `6NNA_B` will select only chain `B` in PDB structure `6NNA`.
2. Uniprot accession IDs of the target(s) of interest. Crystal structures for any given Uniprot targets will be selected automatically. 

There is not requirement for all ligands to be docked into the same set of targets, but all targets given must be either PDB IDs OR Uniprot accession IDs. 
One cannot give a list containing both PDB IDs and Uniprot accession IDs. 
If Uniprot accession IDs are given, be sure to set the argument `map_uniprot_to_pdb` to `True` before running.




### Output structure explained  
The directory `example_output` provides a sample output from a run.
Output directories are structured as follows: 

The directory contains four subdirectories, and it is the following two:

1. pocket_locations
2. docking

that are the most important. 

#### pocket_locations subdirectory
The `pocket_locations` subdirectory contains all of the primary outputs of the run:

1. `pocket_locations/top_XXXX_pockets.json` provides details of the top XXXX (where XXXX is equal to the setting of the input paramter `num_top_pockets`) predicted binding sites for all ligand, target pairs.
The data structure is a nested set of key-value maps, mapping ligand_id to Uniprot accession ID to PDB ID with chain to pocket ID to pocket data. 
The pocket data is comprised of the location and size of the pocket, its machine-learning predicted pocket score, its rank, as well as its original ID for the binding site identification algorithm that predicted it. 

2. `pocket_locations/pose_data.json` provides a summary of all of the poses that were generated after executing docking at all of the pockets defined in `pocket_locations/top_XXXX_pockets.json`.
The data structure is similar to that of `pocket_locations/top_XXXX_pockets.json`, however, instead of pocket data, each pocket ID now maps to pose ID, which maps to pose data.
Pose data provides the location and size of the bounding box of the pose, and its binding energy (given under the key `energy`).

The `pocket_locations` subdirectory also contains a `local_docking` subdirectory that contains PDB files of all of the predicted poses (and complexes).
The poses for a ligand with the ligand ID `LIGAND`, docked into target `ACCESSION` (crystal structure `PDBID`), pocket number `POCKETNUM` can be found at:

```
pocket_locations/local_docking/LIGAND/ACCESSION/PDBID/POCKETNUM/poses
```
and, if `num_complexes` > 0, then any complexes can be found at:
```
pocket_locations/local_docking/LIGAND/ACCESSION/PDBID/POCKETNUM/complexes
```



## Acknowledgements
We sincerely thank the authors of the software that comprises the COBDock pipeline.  

## References

[1] Trott, Oleg, and Arthur J. Olson. "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading." Journal of computational chemistry 31.2 (2010): 455-461.

[2] Yang, Jinsol, Minkyung Baek, and Chaok Seok. "GalaxyDock3: Protein–ligand docking that considers the full ligand conformational flexibility." Journal of Computational Chemistry 40.31 (2019): 2739-2748.

[3] Korb, Oliver, Thomas Stützle, and Thomas E. Exner. "An ant colony optimization approach to flexible protein–ligand docking." Swarm Intelligence 1 (2007): 115-134.

[4] Chen, Rong, Li Li, and Zhiping Weng. "ZDOCK: an initial‐stage protein‐docking algorithm." Proteins: Structure, Function, and Bioinformatics 52.1 (2003): 80-87.

[5] Le Guilloux, Vincent, Peter Schmidtke, and Pierre Tuffery. "Fpocket: an open source platform for ligand pocket detection." BMC bioinformatics 10.1 (2009): 1-11.

[6] Krivák, Radoslav, and David Hoksza. "P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure." Journal of cheminformatics 10 (2018): 1-12.