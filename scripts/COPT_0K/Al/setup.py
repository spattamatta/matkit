'''-----------------------------------------------------------------------------
                                  setup.py

 Description: Creates setup files for pure metal relaxation to target stress.
   
 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
# None

# Externally installed modules
# None

# Local imports
from matkit.calculators import mod_calculator
from matkit.core import mod_constrained_optimization
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
sim_info['sim_type'] = 'copt'
sim_info['n_proc'] = 8

# Create a COPT input. Default values are not set
copt_input = mod_constrained_optimization.template_copt_input.copy()
copt_input['flags_voigt'] = ['S', 'S', 'S', 'S', 'S', 'S']
copt_input['target_stress_voigt'] = [-4.0, -3.0, -2.0, 1.0, 1.0, 1.0] # This is a stress state, just for demonstration

# Attach the COPT input to the sim_info
sim_info['sim_specific_input'] = copt_input

# Create ase atoms and calculator for Aluminium
[structure_list, calculator_input_list, rel_dir_list] = ts_set.get_natural_structures(symbol_list=['Al'])

structure = structure_list[0]
calculator_input = calculator_input_list[0]
calculator_input.set(ncore=2)

'''-----------------------------------------------------------------------------
                                  END OF SETUP
-----------------------------------------------------------------------------'''
