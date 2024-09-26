<p align="center">
    <h1><b>PyTaskFarmer</b></h1>
</p>

The PyTaskFarmer is a python-based, light-weight task farmer code to orchestrate the running of a large of independent jobs, not necessarily of the same granualarity (number of cores for each job) under a single submission (whether being directly run in a bash environment or under a SLURM submission environment), using resource sharing under a manager-worker paradigm.

Although embeded into MaTKit as an utility, the pytaskfarmer is an independent program and can be used to taskfarm any generic code. The details of the taks-farm logic and the usage with examples are explained in detail in the codes in directory [matkit/utility/pytaskfarm](../matkit/utility/pytaskfarm).

