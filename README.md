# Reference Implementation of COBDock algorithm
This readme file documents all of the required steps to run COBDock.

Note that code was implemented and tested on a Linux operating system only.

## How to set up environment
We have provided an Anaconda environment file for easy set up.
If you do not have Anaconda installed, you can get Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
Create the `cobdock-env` environment using the following command:
```bash
conda create -n cobdock-env -f environment.yml
conda activate cobdock-env
```
Then, install two additional packages from git using the following command:
```bash
pip install git+https://github.com/thelahunginjeet/kbutil
pip install git+https://github.com/thelahunginjeet/pyrankagg 
```

## Required external programs
Before running COBDock, one will first need to download and prepare all of the molecular docking and pocket identification algorithms.
Before downloading, prepare a `bin` folder using the following commands:

```bash
mkdir bin
mkdir bin/docking
mkdir bin/location_analysis
```

<!-- ### MGLTools -->
<!-- Download MGLTools-1.5.6 from [here]():  -->

### Vina 1.1.2
Vina 1.1.2 can be downloaded from [here](https://vina.scripps.edu/downloads/).
Execute this command to create a `vina` subdirectory inside `bin/docking`:
```bash
mkdir bin/docking/vina
```
and place the vina binary inside the newly created `vina` subdirectory. 

### GalaxyDock3
GalaxyDock3 can be downloaded from [here](https://galaxy.seoklab.org/softwares/galaxydock.html).
Download the archive file and extract the `GalaxyDock3` directory into `bin/docking`.

### PLANTS1.2
The PLANTS1.2 and SPORES binaries are available [here](http://www.tcd.uni-konstanz.de/plants_download/).
Execute this command to create a `plants` subdirectory inside `bin/docking`:
```bash
mkdir bin/docking/plants
```
and place both binaries inside the newly created `plants` subdirectory. 

### ZDOCK
The ZDOCK binary is available [here](https://zdock.umassmed.edu/software/).
Execute this command to create a `zdock` subdirectory inside `bin/docking`:
```bash
mkdir bin/docking/zdock
```
and place the ZDOCK binary inside the newly created `zdock` subdirectory. 

### fpocket4.0
Fpocket4.0 has is included in the provided Anaconda environment.

### P2Rank 2.3.1
P2Rank 2.3.1 can be downloaded as an archive [here](https://github.com/rdk/p2rank/releases/download/2.3.1/p2rank_2.3.1.tar.gz).
Extract it to the `bin/location_analysis` subdirectory so that `p2rank_2.3.1` can be found inside `bin/location/analysis`.
Confirm that the `prank` binary can be found at `bin/location_analysis/p2rank_2.3.1/prank`.



## How to run


### Configuring environment variables
Modify the `.env` file to set the maximum number of processes for:

1. Preparing targets 
2. Executing Vina 
3. Executing GalaxyDock
4. Executing PLANTS
5. Executing ZDOCK

We strongly advise setting a low number of processes for PLANTS since it is memory intensive.

### Running the COBDock pipeline
Run the following command:

```bash
python cobdock/run_cobdock.py
```
to execute COBDock for aspirin and Uniprot target P23219. Crystal structures will be selected and prepared automatically. 
If the run is successful, ``COBDock successfully completed`` will be printed to the console.
Please allow some time for a run to complete as some molecular docking algoritms can take some time. 


### Configuring future runs


#### All input arguments explained



### Output structure explained  
The directory `example_output` provides a sample output from a run.


## Acknowledgements
We sincerely thank the authors of the software that comprises the COBDock pipeline.  

## References