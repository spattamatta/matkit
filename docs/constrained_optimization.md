<p align="center">
    <h1><b>Constrained Optimization</b></h1>
</p>

Many problems of interest in materials science involve the study of response of a system under applied boundary consitions. In the mechanical context with in the MaTKit framework, one is usually interested in optimizing the supercell (cell shape, size and ionic degress of freedom) for a prescribed or target homogeneous stress strate in the material. By stress state we mean all the six independent componets of the symmetric Cauchy stress tensor. On similar lines one can also be interested in mixed boundary condition problems where only certain components of the target stress tensor are prescribed and the complementary displacemnsts are constrained. For example one can prescribe a biaxial stress in the x-y plane and fixed cell size in the z direction while relaxing any shear stresses. Such requiements typically arise in complex loading scenarios such as the calculation of theoretical strength of materials.

With in the MaTKit framework, we solve this self-consitent boundary value problem by iteratively opimizing the supercell until the target conditions are statisfied.

## Code structure
The constrained optimization workflow is organized as follows:

1. **Core Modules**: The core module for the constrained optimization workflow is [mod_constrained_optimization.py](../matkit/core/mod_constrained_optimization.py). The principal logic behind the iterative algorithm is that at each step, given the current supercell and the stress (obtained from DFT or MD), the supercell at the next iteration is predicted using elasticity equations that minimize the absolute error between current stress and target stress, on all componets of the stress tensor other than those where fixed displacemnets ar prescribed. The iterations are run until the target vales are attained with in certain tolerances.

2. **Application Interface** (optional): The application interface to the core module mentioned above is the [app_constrained_optimization.py](../matkit/apps/app_constrained_optimization.py) application. This application serves as the user interface by reading in the problem setup and input parameters provided by the runner scripts. However it shall be noted that the constrained optimization algorithm is rarely used as a stand alone program but rather as a part of other workflows such as the theoretical strength workflow.

3. **Runner Scripts**: Runner scripts are the user-level input settings that define the problem. We provide example scripts to optimize an fcc aluminum crystal under prescribed arbitrary target stress, in the directory [scripts/COPT_0K/Al](../scripts/COPT_0K/Al). Here the user is expected to provide details of the crystal system of study and the target stress (or mixed boundary condition), and some parallelization settings such as the number of computer cores to be used for the job.

