'''-----------------------------------------------------------------------------
                                  setup.py

 Description: Creates setup files for pure metal relaxation to zero stress for
   the calculation of stress-free equilibrium lattice parameters.
   
   This setup can relax a large number of structures (in their 'natural'
   structure) and find their equilibrium lattice parameters.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
# None

# Externally installed modules
# None

# Local imports
from matkit.calculators import mod_calculator
from scripts.theoretical_strength import theoretical_strength_common_settings as ts_set

'''-----------------------------------------------------------------------------
                                 SUBROUTINES
-----------------------------------------------------------------------------'''
# None

'''-----------------------------------------------------------------------------
                                 MODULE VARIABLES
-----------------------------------------------------------------------------'''

# Copied from mod_calculator: template_sim_info. Default values are not copied
sim_info = mod_calculator.template_sim_info.copy()
sim_info['calculator'] = 'vasp'
sim_info['sim_type'] = 'relax_isv'
sim_info['n_proc'] = 8
sim_info['n_relax_iter'] = 2
sim_info['is_final_static'] = True

# We are relaxing a single element with the generic settings, but this setup can
# handle relaxation of a large number of element structures at once (just change
# the symbol list)
symbol_list = ['Al']
[structure_list, calculator_input_list, rel_dir_list] = ts_set.get_natural_structures(symbol_list=symbol_list)
sim_info_list = []

for idx in range(0, len(symbol_list)):
    calculator_input_list[idx].set(ncore=2)
    calculator_input_list[idx].set(kpar=2)
    calculator_input_list[idx].set(lreal=False)
    
    sim_info_list.append(sim_info.copy())

'''-----------------------------------------------------------------------------
                                  END OF SETUP
-----------------------------------------------------------------------------'''
