# Reference Implementation of COBDock algorithm
This readme file documents all of the required steps to run COBDock.

Note that code was implemented and tested on a Linux operating system only.

## How to set up environment

```bash
conda create -n cobdock-env -f environment.yml
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
mkdir bin/docking/plants
mkdir bin/docking/zdock
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
The PLANTS1.2 binary is available [here](http://www.tcd.uni-konstanz.de/plants_download/).

### ZDOCK

### fpocket

### P2Rank

## How to run

### Configuring environment variables
Modify the `.env` file to set the maximum number of processes for:

1. Preparing targets 
2. Executing Vina 
3. Executing GalaxyDock
4. Executing PLANTS
5. Executing ZDOCK

We strongly advise setting a low number of processes for PLANTS since it is memory intensive.

### Running the COBDock


### Output structure explained  
The directory `example_output` provides a sample output from a run


## Acknowledgements
We sincerely thank the authors of the software that comprises the COBDock pipeline.  

## References