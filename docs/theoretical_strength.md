<p align="center">
    <h1><b>Calculation of Theoretical Strength of Crystals</b></h1>
</p>

MaTKit currently supports the calculation of theoretical strength (TS) of crystals with *cubic* (face-centered cubic (fcc), cubic diamond, body-centered cubic (bcc)) and *hexagonal* (hexagonal close-packed (hcp)) symmetry. The TSs are calculated under hydrostatic tension, uniaxial tension along high symmetry directions, and pure shear on high symmetry planes and along high symmetry directions. Apart from these conventional scenarios of TS, MaTKit supports the calculation of TS in the presence of any arbitrary superimposed stress states. It should be noted that not all arbitrary stress states can be superimposed, primarily because the magnitudes of the stress components are limited by strength considerations.
 
## Code structure
As with any MatKit workflow, the TS workflow is organized as follows:

1. **Core Modules**: The core module for the TS calculation is [mod_incr_load.py](../matkit/core/mod_incr_load.py). The bulk of the module deals with a computational implementation of incremental loading simulation for a given loading program and crystal geometry. This module contains subroutines that (i) implement the principal logic behind the calculation of TS, (ii) set up the calculation workflow (directory structure, scripts that interface with calculators such as VASP (see the directory [matkit/calculators](../matkit/calculators)), and orchestrate the incremental loading job), and (iii) basic plotting routines.

2. **Application Interface**: The application interface to the core module mentioned above is the [app_incr_load.py](../matkit/apps/app_incr_load.py) application. This application serves as the user interface by reading in the problem setup and input parameters provided by the runner scripts.

3. **Runner Scripts**: Runner scripts are the user-level input settings that define the problem. We provide example scripts to compute the TS of fcc aluminum crystal in the directory [scripts/theoretical_strength/MRL/Al/ts](../scripts/theoretical_strength/MRL/Al/ts). Here the user is expected to provide details of the crystal system of study, loading planes and directions for TS calculation as labelled in the core module [mod_incr_load.py](../matkit/core/mod_incr_load.py), stress states to be superimposed, and some parallelization settings such as the number of computer cores to be used for each job and the total number of cores to be used for a set of jobs that run as a unit, under the resource-sharing architecture i.e the taskfarm (see the directory [matkit/utility/pytaskfarm](../matkit/utility/pytaskfarm)), etc.
