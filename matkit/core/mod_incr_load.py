'''-----------------------------------------------------------------------------
                              mod_incr_load.py

 Description: Incremental loading algorithm at 0K and finite temperatures (using
  AIMD) along with phonon stability (for 0K only) analysis at each load point.

  Algorithm Operation:
  -------------------
  The algorithm operates by straining incrementally, each solution from its
  previous step along the loading path. This is physically relevant than
  directly loading to a particular strain, because direct loading "can ?"
  in some cases relax to a different structure. Physically, in laboratories load
  is applied continuously at a particular rate or incrementally.
       
  Termination Criterion:
  ---------------------
  The algorithm terminates when the peak stress (Frenkel-Orowan) is reached and
  instability (that is a drop in stress) is encountered along the loading path.
 
  The strength limit (Frenkel-Orowan) is monitored by  observing the
  corresponding stress / pressure component where a monotonically (in general
  nonlinear) increasing trend is interrupted, indicating strength limit has been
  crossed.
  
  Once the strength limit is reached (observed by a drop in the corresponding
  stress component of the controlled displacement), the algorithm either
  terminates or adds one or more partial back steps after the maximum strength
  step (i.e previous step) to better capture the maximum.
       
  Given the loading_spec and the stress tensor at the current and the previous
  loading points, termination criterion is determined by monitoring
  corresponding stress component.

  Phonon Instability:
  ------------------
  If phonon stability calculation is turned on, the code should first finish the
  load based incremental loading. Then in the phonon run mode, the code sets up phonon
  calculation(s) in a separate FORCE_CONSTANTS directory in the same root where
  LOAD resided. The user can try multiple supercell sizes.
  
  This run is separated from incremental loading because, force constant
  calculations need super cells, so they need different number of processors
  than the load incremental loading calculation.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json
import time
import math
import shutil
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)

# Externally installed modules
from ase.io.vasp import read_vasp, write_vasp

# Local imports
from matkit import CONFIG
from matkit.calculators import mod_calculator, mod_dft, mod_vasp
from matkit.core import mod_ase, mod_batch, mod_utility, mod_tensor, \
    mod_atomic_systems, mod_phonopy, mod_constrained_optimization
from matkit.apps import app_incr_load
from matkit.utility import mod_colors

'''-----------------------------------------------------------------------------
                                  MODULE VARIABLES
-----------------------------------------------------------------------------'''
module_name = 'mod_incr_load.py'

# Template for incremental loading information
template_incr_load_info = {
    'calculator' : 'vasp',                              # Currently vasp is the only supported calculator
    'rel_dir_name' : None,                              # Named after species, lattice type and loading type
    'system' : None,                                    # Something about the system such as fcc, cubic-diamond (same as fcc), bcc, hcp
    'material' : None,                                  # Title of the material
    'atoms': None,                                      # An Atoms structure for which the theoretical strength is to be calculated
    'calc_load' : None,                                 # Calculator for incremental loading
    'calc_force' : None,                                # Calculator for Force constants
    'loading_spec_list' : [],                           # ['hydrostatic'], ['normal', 0, 0], ['normal', 1, 2], ['shear', 'simple', 2, 0] first plane and second direction of applied load (x = 0, y = 1 and z = 2)
    'copt_flags_voigt' : None,                          # Flags for constrained cell relaxation, see mod_constrained_optimization.py
    'copt_target_stress_voigt' : None,                  # Values of stress in GPa that are superimposed on the system, otherthan the controlled displacement.
    'termination_criterion' : 'MAX_STRESS',             # What is the termination criterion, see header for explanation
    'max_load_steps' : 50,                              # Maximum number of loading steps, code terminates after that
    'incremental_deformation_parameter' : None,         # Incremental strain value be it hydrostatic(+/-), uniaxial(+/-) or shear(+/-)
    'is_incremental_loading' : True,                    # If the strain is applied incrementally or directly from reference
    'phonon_supercell_size_list' : [np.array([2,2,2])], # Size of supercell for phonon calculations
    'is_disturb' : False,                               # Should a random disturbance be applied to each of the ions and each of the 3 degrees of freedom
    'disturb_vec' : np.zeros(3),                        # Disturbance magnitudes in x, y and z directions
    'disturb_seed' : None,                              # Random number seed for disturbance generation
    'stress_voigt_convergence_tol_GPa' : np.array([1.0E-2, 1.0E-2, 1.0E-2, 1.0E-1, 1.0E-1, 1.0E-1]) # Convergence tolerance for stresses on non-displacemnt controlled components
}

#############################
#  List of loadings for FCC #
#############################

fcc_loading_system_info_dict = {

    #######################
    # Hydrostatic tension #
    #######################
    'fcc_hydrostatic_tension' : {
        'atomic_system' : 'fcc_primitive',
        'load_type' : 'hydrostatic_tension',
        'loading_spec_list' : ['hydrostatic_tension'],
        'label' : 'hydrostatic',
        'n_layer' : 1
    }, # n_layer is dummy for hydrostatic
    
    #####################
    # Uniaxial tensions #
    #####################
    # Tension along 001 z axis    
    'fcc001x100_tension_z' : {
        'atomic_system' : 'fcc001x100',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[001]$',
        'n_layer' : 1
    },
    
    # Tension along 110
    'fcc110x-110_tension_z' : {
        'atomic_system' : 'fcc110x-110',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[110]$',
        'n_layer' : 2
    },
    
    # Tension along 111
    'fcc111x11-2_tension_z' : {
        'atomic_system' : 'fcc111x11-2',
        'load_type' : 'normal_z', 
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[111]$',
        'n_layer' : 3
    },
    
    ######################
    # {111} slip systems #
    ######################
    
    # Symmetric: Pure shear on 111 (z) plane along -110 (x) direction (Symmetric w.r.t. shear along x) 
    'fcc111x-110_shear_zx' : {
        'atomic_system' : 'fcc111x-110',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Sym: \,$(111)[\bar{1}10]$',
        'n_layer' : 3
    },
    
    # Symmetric: Pure shear on 111 (z) plane along 1-10 (x) direction (Symmetric w.r.t. shear along -x) 
    'fcc111x1-10_shear_zx' : {
        'atomic_system' : 'fcc111x1-10',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Sym: \,$(111)[1\bar{1}0]$',
        'n_layer' : 3
    },   
    
    # Easy : Pure shear on 111 (z) plane along 11-2 (x) direction
    'fcc111x11-2_shear_zx' : {
        'atomic_system' : 'fcc111x11-2', 
        'load_type' : 'simple_shear_zx', 
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Easy: $(111)[11\bar{2}]$',
        'n_layer' : 3
    },
    
    # Hard: Pure shear on 111 (z) plane along -1-12 (x) direction    
    'fcc111x-1-12_shear_zx' : {
        'atomic_system' : 'fcc111x-1-12', 
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Hard:\,\,$(111)[\bar{1}\bar{1}2]$',
        'n_layer' : 3},

}

# Plotting info for FCC
fcc_plot_info_dict = {

    'hydrostatic' : {
        'loading_system_list' : ['fcc_hydrostatic_tension'],
        'strain_sign_list' : [+1],
        'line_color_list' : [mod_colors.line_color_set.red],
        'marker_type_list' : ['o'],
        'marker_face_color_list' : [mod_colors.line_color_set.red],
        'x_label' : r'Volumetric strain $\frac{\Delta V}{V_0}$',
        'y_label' : r'Hydrostatic stress $\sigma^h$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma^h_{\textnormal{max}}$'
    },
    
    'normal' : {
        'loading_system_list' : ['fcc001x100_tension_z', 'fcc110x-110_tension_z', 'fcc111x11-2_tension_z'],
        'strain_sign_list' : [+1, +1, +1],
        'line_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.blue],
        'marker_type_list' : ['o', 's', '^'],                                           
        'marker_face_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.blue],
        'x_label' : r'Normal strain',
        'y_label' : r'Normal stress $\sigma$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma_{\textnormal{max}}$'
    },

    'shear' : {
        'loading_system_list' : ['fcc111x-110_shear_zx', 'fcc111x1-10_shear_zx', 'fcc111x11-2_shear_zx', 'fcc111x-1-12_shear_zx'],
        'strain_sign_list' : [+1, -1, +1, -1], 
        'line_color_list' : [mod_colors.line_color_set.blue, mod_colors.line_color_set.blue, mod_colors.line_color_set.red, mod_colors.line_color_set.red],
        'marker_type_list' : ['o', '^', 'o', '^'],
        'marker_face_color_list' : [mod_colors.line_color_set.blue, mod_colors.line_color_set.blue, mod_colors.line_color_set.red, mod_colors.line_color_set.red],
        'x_label' : r'Shear strain $\gamma$',
        'y_label' : r'Shear stress $\tau$ (GPa)',
        'maximum_stress_label' : r'$\tau_{\textnormal{max}}$'  
    }

}

#######################################
#  List of loadings for cubic-diamond #
#######################################

cubic_diamond_loading_system_info_dict = {

    #######################
    # Hydrostatic tension #
    #######################
    'cubic-diamond_hydrostatic_tension' : {
        'atomic_system' : 'cubic-diamond_primitive',
        'load_type' : 'hydrostatic_tension',
        'loading_spec_list' : ['hydrostatic_tension'],
        'label' : 'hydrostatic',
        'n_layer' : 1
    }, # n_layer is dummy for hydrostatic
    
    #####################
    # Uniaxial tensions #
    #####################
    # Tension along 001 z axis    
    'cubic-diamond001x100_tension_z' : {
        'atomic_system' : 'cubic-diamond001x100',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[001]$',
        'n_layer' : 1
    },
    
    # Tension along 110
    'cubic-diamond110x-110_tension_z' : {
        'atomic_system' : 'cubic-diamond110x-110',
        'load_type' : 'normal_z', 
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[110]$',
        'n_layer' : 2
    },
    
    # Tension along 111
    'cubic-diamond111x11-2_tension_z' : {
        'atomic_system' : 'cubic-diamond111x11-2',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[111]$',
        'n_layer' : 3
    },
    
    ######################
    # {111} slip systems #
    ######################
    
    # Symmetric: Pure shear on 111 (z) plane along -110 (x) direction (Symmetric w.r.t. shear along x) 
    'cubic-diamond111x-110_shear_zx' : {
        'atomic_system' : 'cubic-diamond111x-110',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Sym: \,$(111)[\bar{1}10]$',
        'n_layer' : 3
    },
    
    # Symmetric: Pure shear on 111 (z) plane along 1-10 (x) direction (Symmetric w.r.t. shear along -x) 
    'cubic-diamond111x1-10_shear_zx' : {
        'atomic_system' : 'cubic-diamond111x1-10',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Sym: \,$(111)[1\bar{1}0]$',
        'n_layer' : 3
    },   
    
    # Easy : Pure shear on 111 (z) plane along 11-2 (x) direction
    'cubic-diamond111x11-2_shear_zx' : {
        'atomic_system' : 'cubic-diamond111x11-2',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Easy: $(111)[11\bar{2}]$',
        'n_layer' : 3
    },
    
    # Hard: Pure shear on 111 (z) plane along -1-12 (x) direction    
    'cubic-diamond111x-1-12_shear_zx' : {
        'atomic_system' : 'cubic-diamond111x-1-12',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Hard:\,\,$(111)[\bar{1}\bar{1}2]$',
        'n_layer' : 3
    },

}

# Plotting info for cubic-diamond
cubic_diamond_plot_info_dict = {
    'hydrostatic' : { 
        'loading_system_list' : ['cubic-diamond_hydrostatic_tension'],
        'strain_sign_list' : [+1],
        'line_color_list' : [mod_colors.line_color_set.red],
        'marker_type_list' : ['o'],
        'marker_face_color_list' : [mod_colors.line_color_set.red],
        'x_label' : r'Volumetric strain $\frac{\Delta V}{V_0}$',
        'y_label' : r'Hydrostatic stress $\sigma^h$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma^h_{\textnormal{max}}$'
    },
    
    'normal' : {
        'loading_system_list' : ['cubic-diamond001x100_tension_z', 'cubic-diamond110x-110_tension_z', 'cubic-diamond111x11-2_tension_z'],
        'strain_sign_list' : [+1, +1, +1],
        'line_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.blue],
        'marker_type_list' : ['o', 's', '^'],                                           
        'marker_face_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.blue],
        'x_label' : r'Normal strain',
        'y_label' : r'Normal stress $\sigma$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma_{\textnormal{max}}$'
    },

    'shear' : {
        'loading_system_list' : ['cubic-diamond111x-110_shear_zx', 'cubic-diamond111x1-10_shear_zx', 'cubic-diamond111x11-2_shear_zx', 'cubic-diamond111x-1-12_shear_zx'],
        'strain_sign_list' : [+1, -1, +1, -1], 
        'line_color_list' : [mod_colors.line_color_set.blue, mod_colors.line_color_set.blue, mod_colors.line_color_set.red, mod_colors.line_color_set.red],
        'marker_type_list' : ['o', '^', 'o', '^'],
        'marker_face_color_list' : [mod_colors.line_color_set.blue, mod_colors.line_color_set.blue, mod_colors.line_color_set.red, mod_colors.line_color_set.red],
        'x_label' : r'Shear strain $\gamma$',
        'y_label' : r'Shear stress $\tau$ (GPa)',
        'maximum_stress_label' : r'$\tau_{\textnormal{max}}$'  
    }

}

#############################
#  List of loadings for BCC #
#############################

bcc_loading_system_info_dict = {

    #######################
    # Hydrostatic tension #
    #######################
    'bcc_hydrostatic_tension' : { 
        'atomic_system' : 'bcc_primitive', 
        'load_type' : 'hydrostatic_tension',
        'loading_spec_list' : ['hydrostatic_tension'],
        'label' : 'hydrostatic',
        'n_layer' : 1
    }, # n_layer here is dummy
    
    #####################
    # Uniaxial tensions #
    #####################
    # Tension along 001 z axis
    'bcc001x100_tension_z' : {
        'atomic_system' : 'bcc001x100',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[001]$',
        'n_layer' : 1
    },
    
    # Tension along 110
    'bcc110x-111_tension_z' : {
        'atomic_system' : 'bcc110x-111',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[110]$',
        'n_layer' : 2
    },
    
    # Tension along 111
    'bcc111x-110_tension_z' : {
        'atomic_system' : 'bcc111x-110',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'$[111]$',
        'n_layer' : 3
    },
    
    # NOTE: For shears only pencil glides on {110} {112} and {123} planes along <111> directions are considered owing to lack of resources
    ######################
    # {110} slip systems #
    ######################
    # Symmetric: Pure shear on 110 (z) plane along -111 (x) direction (a pencil glide)
    'bcc110x-111_shear_zx' : {
        'atomic_system' : 'bcc110x-111',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Sym: $(110)[\bar{1}11]$',
        'n_layer' : 4
    },
    
    # Symmetric: Pure shear on 110 (z) plane along 1-1-1 (x) direction (a pencil glide)    
    'bcc110x1-1-1_shear_zx' : {
        'atomic_system' : 'bcc110x1-1-1', 
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Sym: $(110)[1\bar{1}\bar{1}]$',
        'n_layer' : 4
    },   
  
    ######################
    # {112} slip systems #
    ######################
    # Easy: Pure shear on 112 (z) plane along -1-11 (x) direction (a pencil glide) 
    'bcc112x-1-11_shear_zx' : {
        'atomic_system' : 'bcc112x-1-11',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Easy: $(112)[\bar{1}\bar{1}1]$',
        'n_layer' : 6
    },
    
    # Hard: Pure shear on 112 (z) plane along 11-1 (x) direction (a pencil glide)
    'bcc112x11-1_shear_zx' : {
        'atomic_system' : 'bcc112x11-1',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Hard: $(112)[11\bar{1}]$',
        'n_layer' : 6
    },

    ######################
    # {123} slip systems #
    ######################
    # Easy: Pure shear on 123 (z) plane along -1-11 (x) direction (a pencil glide) 
    'bcc123x-1-11_shear_zx' : {
        'atomic_system' : 'bcc123x-1-11',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Easy: $(123)[\bar{1}\bar{1}1]$',
        'n_layer' : 8
    },
    
    # Hard: Pure shear on 123 (z) plane along 11-1 (x) direction (a pencil glide)
    'bcc123x11-1_shear_zx' : {
        'atomic_system' : 'bcc123x11-1',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Hard: $(123)[11\bar{1}]$',
        'n_layer' : 8}, 
}

# Plotting info for BCC
bcc_plot_info_dict = {
    'hydrostatic' : {
        'loading_system_list' : ['bcc_hydrostatic_tension'],
        'strain_sign_list' : [+1],
        'line_color_list' : [mod_colors.line_color_set.red],
        'marker_type_list' : ['o'],
        'marker_face_color_list' : [mod_colors.line_color_set.red],                      
        'x_label' : r'Volumetric strain $\frac{\Delta V}{V_0}$',
        'y_label' : r'Hydrostatic stress $\sigma^h$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma^h_{\textnormal{max}}$'                       
    },

    'normal' : {
        'loading_system_list' : ['bcc001x100_tension_z', 'bcc110x-111_tension_z', 'bcc111x-110_tension_z'],
        'strain_sign_list' : [+1, +1, +1],
        'line_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.blue],
        'marker_type_list' : ['o', 's', '^'],                                           
        'marker_face_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.blue],                    
        'x_label' : r'Normal strain',
        'y_label' : r'Normal stress $\sigma$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma_{\textnormal{max}}$'
    },

    'shear' : {
        'loading_system_list' : ['bcc110x-111_shear_zx', 'bcc110x1-1-1_shear_zx', 'bcc112x-1-11_shear_zx', 'bcc112x11-1_shear_zx', 'bcc123x-1-11_shear_zx', 'bcc123x11-1_shear_zx'],
        'strain_sign_list' : [+1, -1, +1, -1, +1, -1],
        'line_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.green, mod_colors.line_color_set.blue, mod_colors.line_color_set.blue],
        'marker_type_list' : ['o', '^', 'o', '^', 'o', '^'],                                           
        'marker_face_color_list' : [mod_colors.line_color_set.red, mod_colors.line_color_set.red, mod_colors.line_color_set.green, mod_colors.line_color_set.green, mod_colors.line_color_set.blue, mod_colors.line_color_set.blue],                   
        'x_label' : r'Shear strain $\gamma$',
        'y_label' : r'Shear stress $\tau$ (GPa)',
        'maximum_stress_label' : r'$\tau_{\textnormal{max}}$'  
    }

}

#############################
#  List of loadings for HCP #
#############################

hcp_loading_system_info_dict = {

    #######################
    # Hydrostatic tension #
    #######################
    'hcp_hydrostatic_tension' : {
        'atomic_system' : 'hcp_primitive',
        'load_type' : 'hydrostatic_tension',
        'loading_spec_list' : ['hydrostatic_tension'], 
        'label' : 'hydrostatic',
        'n_layer' : 1
    },  
    
    #################
    # Basal Tension #
    #################
    'hcp0001x2-1-10_tension_z' : {
        'atomic_system' : 'hcp0001x2-1-10',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'Basal $[0001]$',
        'n_layer' : 1
    },
    
    ###############
    # Basal Shear #
    ###############
    # Shear along [2,-1,-1,0], NOT a SF direction
    'hcp0001x2-1-10_shear_zx' : {
        'atomic_system' : 'hcp0001x2-1-10',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Basal $(0001)[2\bar{1}\bar{1}0]$',
        'n_layer' : 1
    },
    
    # Shear along [1,-1,0,0], SF direction
    'hcp0001x1-100_shear_zx' : {
        'atomic_system' : 'hcp0001x1-100',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Basal $(0001)[1\bar{1}00]$',
        'n_layer' : 1
    },      
    
    # Shear along [1,0,-1,0], SF direction
    'hcp0001x10-10_shear_zx' : {
        'atomic_system' : 'hcp0001x10-10',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Basal $(0001)[10\bar{1}0]$',
        'n_layer' : 1
    },

    ###################
    # Prism I Tension #
    ###################
    'hcp01-10x-2110_tension_z' : {
        'atomic_system' : 'hcp01-10x-2110',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'Pris I $[10\bar{1}0]$',
        'n_layer' : 1
    },   
    
    #################
    # Prism I Shear #
    #################
    # Shear along [-2,1,1,0], SF1
    'hcp01-10x-2110_shear_zx'  : {
        'atomic_system' : 'hcp01-10x-2110',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Pris I $(01\bar{1}0)[\bar{2}110]$',
        'n_layer' : 1
    },         
    
    # Shear along [-2,1,1,3], SF2
    'hcp01-10x-2113_shear_zx'  : {
        'atomic_system' : 'hcp01-10x-2113', 
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Pris I $(01\bar{1}0)[\bar{2}113]$',
        'n_layer' : 1
    },
        
    ####################
    # Prism II Tension #
    ####################
    'hcp-12-10x-1010_tension_z' : {
        'atomic_system' : 'hcp-12-10x-1010',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'Pris\,II\,$[\bar{1}2\bar{1}0]$',
        'n_layer' : 1
    },
    
    ##################
    # Prism II Shear #
    ##################
    # NOTE: could not find proper shear direction for Prism II stacking faults
    
    #######################
    # Pyramidal I Tension #
    #######################
    'hcp01-11x-2110_tension_z' : {
        'atomic_system' : 'hcp01-11x-2110',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'Pyrs\,I $[01\bar{1}1]$',
        'n_layer' : 1
    },           
   
    #####################
    # Pyramidal I Shear #
    #####################
    # Shear along [-2,1,1,0] <a> slip
    'hcp01-11x-2110_shear_zx' : {
        'atomic_system' : 'hcp01-11x-2110',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Pyra\,I\,$(01\bar{1}1)[\bar{2}110]$',
        'n_layer' : 1
    },
    
    # Shear along [1,-2,1,3] <c+a> slip
    'hcp01-11x1-213_shear_zx' : {
        'atomic_system' : 'hcp01-11x1-213',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Pyra\,I\,$(01\bar{1}1)[1\bar{2}13]$',
        'n_layer' : 1
    },    
        
    # Shear along [0,-1,1,2], SF2
    'hcp01-11x0-112_shear_zx' : {
        'atomic_system' : 'hcp01-11x0-112',
        'load_type' : 'simple_shear_zx',
        'loading_spec_list' : ['shear', 'simple', 2, 0],
        'label' : r'Pyra\,I\,$(01\bar{1}1)[0\bar{1}12]$',
        'n_layer' : 1
    },        
   
    ########################
    # Pyramidal II Tension #
    ########################
    'hcp-12-12x-1010_tension_z' : {
        'atomic_system' : 'hcp-12-12x-1010',
        'load_type' : 'normal_z',
        'loading_spec_list' : ['normal', 2, 2],
        'label' : r'Pyr\,II $[\bar{1}2\bar{1}2]$',
        'n_layer' : 1
    },                
    
    ######################
    # Pyramidal II Shear #
    ######################
    # Shear along [1,-2,1,3], SF1
    'hcp-12-12x1-213_shear_zx' : {
        'atomic_system' : 'hcp-12-12x1-213',
         'load_type' : 'simple_shear_zx',
         'loading_spec_list' : ['shear', 'simple', 2, 0],
         'label' : r'Pyr\,II $(\bar{1}2\bar{1}2)[1\bar{2}13]$',
         'n_layer' : 1
    },

}

# Plotting info for HCP
hcp_plot_info_dict = {
    'hydrostatic' : {
        'loading_system_list' : ['hcp_hydrostatic_tension'],
        'strain_sign_list' : [+1],
        'line_color_list' : [mod_colors.line_color_set.red],
        'marker_type_list' : ['o'],
        'marker_face_color_list' : [mod_colors.line_color_set.red],                      
        'x_label' : r'Volumetric strain $\frac{\Delta V}{V_0}$',
        'y_label' : r'Hydrostatic stress $\sigma^h$ (GPa)',
        'maximum_stress_label' : r'$\sigma^h_{\textnormal{max}}$'                       
    },

    'normal' : {
        'loading_system_list' : ['hcp0001x2-1-10_tension_z', 'hcp01-10x-2110_tension_z', 'hcp-12-10x-1010_tension_z', 'hcp01-11x-2110_tension_z', 'hcp-12-12x-1010_tension_z'],
        'strain_sign_list' : [+1, +1, +1, +1, +1],
        'line_color_list' : [mod_colors.line_color_set.green, mod_colors.line_color_set.cyan, mod_colors.line_color_set.blue, mod_colors.line_color_set.yellow, mod_colors.line_color_set.magenta],
        'marker_type_list' : ['o', 's', '^', 'd', 'P'],                                           
        'marker_face_color_list' : [mod_colors.line_color_set.green, mod_colors.line_color_set.cyan, mod_colors.line_color_set.blue, mod_colors.line_color_set.yellow, mod_colors.line_color_set.magenta],                 
        'x_label' : r'Normal strain',
        'y_label' : r'Normal stress $\sigma$ (GPa)',                      
        'maximum_stress_label' : r'$\sigma_{\textnormal{max}}$'                                          
    },

    'shear' : {
        'loading_system_list' : ['hcp0001x2-1-10_shear_zx', 'hcp0001x1-100_shear_zx', 'hcp0001x10-10_shear_zx', 'hcp01-10x-2110_shear_zx', 'hcp01-10x-2113_shear_zx', 'hcp01-11x-2110_shear_zx', 'hcp01-11x1-213_shear_zx', 'hcp01-11x0-112_shear_zx', 'hcp-12-12x1-213_shear_zx'],
        'strain_sign_list' : [+1, +1, +1, +1, +1, +1, +1, +1, +1],
        'line_color_list' : [mod_colors.line_color_set.green, mod_colors.line_color_set.green, mod_colors.line_color_set.green, mod_colors.line_color_set.cyan, mod_colors.line_color_set.cyan, mod_colors.line_color_set.yellow, mod_colors.line_color_set.yellow, mod_colors.line_color_set.yellow, mod_colors.line_color_set.magenta],
        'marker_type_list' : ['o', '^', 's', 'd', 'o', '^', 'd', 's', 'o'],
        'marker_face_color_list' : [mod_colors.line_color_set.green, mod_colors.line_color_set.green, mod_colors.line_color_set.green, mod_colors.line_color_set.cyan, mod_colors.line_color_set.cyan, mod_colors.line_color_set.yellow, mod_colors.line_color_set.yellow, mod_colors.line_color_set.yellow, mod_colors.line_color_set.magenta],
        'x_label' : r'Shear strain $\gamma$',
        'y_label' : r'Shear stress $\tau$ (GPa)',
        'maximum_stress_label' : r'$\tau_{\textnormal{max}}$'                                            
    }

}

#######################################
# Loading systems for common crystals #
#######################################

all_loading_system_info_dict = {
   'fcc' : fcc_loading_system_info_dict,
   'cubic-diamond' : cubic_diamond_loading_system_info_dict,
   'bcc' : bcc_loading_system_info_dict,
   'hcp' : hcp_loading_system_info_dict,
}

all_loading_plot_info_dict = {
   'fcc' : fcc_plot_info_dict,
   'cubic-diamond' : cubic_diamond_plot_info_dict,
   'bcc' : bcc_plot_info_dict,
   'hcp' : hcp_plot_info_dict,
}

# What VASP files can be deleted safely for this module
incr_load_clean_file_list = ['CHG', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 
    'XDATCAR', 'PROCAR', 'REPORT', 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2',
    'ase-sort.dat', 'OSZICAR', 'vasprun.xml', 'POTCAR', 'y_max_force',
    '#DONE#', 'console.report.vasp', 'WAVECAR', 'CHGCAR']

'''----------------------------------------------------------------------------
                                  SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################


def get_biaxial_stress_voigt_list(
        stress_1_list, idx_1, idx_2, stress_2_list=None, load_ratio=None):

    '''
    Allowed combinations:
        (a). stress_1_list, stress_2_list, idx_1 and idx_2
        (b). stress_1_list, load_ratio, idx_1 and idx_2
    this subroutine creates a list of voigt stress (biaxial) list
    
    NOTE: load_ratio = (load along idx_2) / (load along idx_1)
    So if you want balanced biaxial stress, load_ratio=1
    if you want uniaxial, then load_ratio=0
    '''
   
    # Sanity check
    if (stress_2_list is None) and (load_ratio is None):
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'get_biaxial_stress_voigt_list'\n")
        sys.stderr.write("       Both 'stress_2_list' and 'load_ratio' are None\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
        
    # Sanity check
    if (stress_2_list is not None) and (load_ratio is not None):
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'get_biaxial_stress_voigt_list'\n")
        sys.stderr.write("       Either 'stress_2_list' or 'load_ratio' should be provided, not both\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
    
    # Sanity check
    if stress_2_list is not None:
        if len(stress_1_list) != len(stress_2_list):
            sys.stderr.write("Error: In module '%s'\n" %(module_name))
            sys.stderr.write("       In subroutine 'get_biaxial_stress_voigt_list'\n")
            sys.stderr.write("       len(stress_1_list) != len(stress_2_list)\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

    # Calculate stresses
    biaxial_stress_voigt_list = []
    for sidx, stress_1 in enumerate(stress_1_list, start=0):
        stress_voigt = np.zeros(6)
        stress_voigt[idx_1] = stress_1
        
        if stress_2_list is not None:
            stress_voigt[idx_2] = stress_2_list[sidx]
        else:
            stress_voigt[idx_2] = stress_1 * load_ratio

        biaxial_stress_voigt_list.append(stress_voigt)

    return biaxial_stress_voigt_list
   
##################
### SUBROUTINE ###
##################


def get_triaxial_stress_voigt_list(stress_3_list, idx_3, lambda_list):

    ids = [0, 1, 2]
    ids.remove(idx_3)
    idx_1 = ids[0]
    idx_2 = ids[1]
    
    triaxial_stress_voigt_list = []
    for lambda_val in lambda_list:
        for stress_3 in stress_3_list:
            stress_voigt = np.zeros(6)
            stress_voigt[idx_3] = stress_3
            stress_voigt[idx_1] = stress_voigt[idx_2] = lambda_val * stress_3
            triaxial_stress_voigt_list.append(stress_voigt)

    return triaxial_stress_voigt_list

##################
### SUBROUTINE ###
##################


def setup_incr_load_info_list(
        species_list, initial_magmom_per_site, lattice_type, lattice, calc_load,
        stress_strain_info_dict, termination_criterion='MAX_STRESS', max_load_steps=50,
        is_incremental_loading=True,
        phonon_supercell_size_list = [],
        is_disturb=False, disturb_vec = np.zeros(3), disturb_seed=None,
        stress_voigt_convergence_tol_GPa = np.array([1.0E-2, 1.0E-2, 1.0E-2, 1.0E-1, 1.0E-1, 1.0E-1]),
        calculator='vasp'):

    '''
    Returns a list of loading dictionaries for failure strength calculations

    biaxial_stress_value_list will be used for tensile test
    '''

    # Four types of calculators will be used
    # 1: For general incremental loading: Use calc_input_load, make sure isym=0 and isif=2, (since copt_inbuilt' is used by default
    calc_load_general = deepcopy(calc_load)
    calc_load_general.set(isym=0)
    calc_load_general.set(isif=2)
    
    # 2: For general phonon calculations
    calc_force_general = deepcopy(calc_load_general)
      
    # 3: For hydrostatic strain no stress should be imposed, just relax the
    # ionic degrees of freedom while holding the volume constant, at the applied dilatation
    calc_load_hydrostatic = deepcopy(calc_load)
    calc_load_hydrostatic.set(isym=2)
    calc_load_hydrostatic.set(isif=4)
    
    # 4: For hydrostatic phonon calculations
    calc_force_hydrostatic = deepcopy(calc_load_hydrostatic)
  
    # output Variables
    lcdict_list = []
    
    #-------------------------------------------------------------------------#
    #            CREATE INCREMENTAL LOADING INFO LISTS IN ORDER               #
    #----------------------------------------------------------+--------------#
    loading_system_info_dict = all_loading_system_info_dict[lattice_type]
    for load_key, load_info in loading_system_info_dict.items():

        # Some loading modes may be skipped
        if load_key not in stress_strain_info_dict:
            continue

        # Setup atoms
        atoms = mod_atomic_systems.get_atomic_system(
            atomic_system=load_info['atomic_system'],
            lattice=lattice, species_list=species_list,
            n_layer=load_info['n_layer'],
            initial_magmom_per_site=initial_magmom_per_site, is_primitive=True)

        #atoms = mod_atomic_systems.get_atomic_system_ase(
        #    atomic_system=load_info['atomic_system'],
        #    lattice=lattice, species_list=species_list,
        #    n_layer=load_info['n_layer'],
        #    initial_magmom_per_site=initial_magmom_per_site)         

        supercell = mod_phonopy.get_minimal_supercell(in_atoms=atoms)
            
        # Setup calculator and incremental loading information
        if load_info['load_type'] == 'hydrostatic_tension':
           
            lcdict = template_incr_load_info.copy()
            
            lcdict['calculator'] = calculator
            lcdict['rel_dir_name'] = load_key
            lcdict['system'] = lattice_type
            lcdict['material'] = species_list[0]
            lcdict['atoms'] = atoms.copy()
            lcdict['calc_load'] = deepcopy(calc_load_hydrostatic)
            lcdict['calc_force'] = deepcopy(calc_force_hydrostatic)
            lcdict['loading_spec_list'] = load_info['loading_spec_list']
            lcdict['copt_flags_voigt'] = None
            lcdict['copt_target_stress_voigt'] = None
            lcdict['termination_criterion'] = termination_criterion
            lcdict['max_load_steps'] = max_load_steps
            lcdict['incremental_deformation_parameter'] = stress_strain_info_dict[load_key]['incremental_deformation_parameter']
            lcdict['is_incremental_loading'] = is_incremental_loading
            lcdict['phonon_supercell_size_list'] = [supercell]
            lcdict['is_disturb'] = is_disturb
            lcdict['disturb_vec'] = disturb_vec
            lcdict['disturb_seed'] = disturb_seed
            lcdict['stress_voigt_convergence_tol_GPa'] = stress_voigt_convergence_tol_GPa
               
            # Push to list
            lcdict_list.append(lcdict)
            
        else:          
            # For each superposed stress
            for sidx, stress_voigt in enumerate(stress_strain_info_dict[load_key]['stress_voigt_list_list'], start=0):

                lcdict = template_incr_load_info.copy()
            
                lcdict['calculator'] = calculator
                lcdict['rel_dir_name'] = load_key + '/STRESS_' + str(sidx).zfill(3)
                lcdict['system'] = lattice_type
                lcdict['material'] = species_list[0]
                lcdict['atoms'] = atoms.copy()
                lcdict['calc_load'] = deepcopy(calc_load_general)
                lcdict['calc_force'] = deepcopy(calc_force_general)
                lcdict['loading_spec_list'] = load_info['loading_spec_list']
                lcdict['copt_flags_voigt'] = get_copt_flags_voigt(load_info['load_type'])
                lcdict['copt_target_stress_voigt'] = stress_strain_info_dict[load_key]['stress_voigt_list_list'][sidx]
                lcdict['termination_criterion'] = termination_criterion
                lcdict['max_load_steps'] = max_load_steps
                lcdict['incremental_deformation_parameter'] = stress_strain_info_dict[load_key]['incremental_deformation_parameter']
                lcdict['is_incremental_loading'] = is_incremental_loading
                lcdict['phonon_supercell_size_list'] = [supercell]
                lcdict['is_disturb'] = is_disturb
                lcdict['disturb_vec'] = disturb_vec
                lcdict['disturb_seed'] = disturb_seed
                lcdict['stress_voigt_convergence_tol_GPa'] = stress_voigt_convergence_tol_GPa
                
                # Push to list
                lcdict_list.append(lcdict)

    return lcdict_list  

##################
### SUBROUTINE ###
##################


def get_hydrostatic_defgrad(
        step_idx_1_based, incremental_stretch, is_incremental):

    '''
    Computes incremental or direct deformation gradient for hydrostatic strain.
    step_idx_1_based : Index counting starts from 1
    '''

    F = np.eye(3)
    linear_stretch_curr = 1.0 + step_idx_1_based * incremental_stretch

    # Incremental deformation
    if is_incremental:
        linear_stretch_prev = 1.0 + (step_idx_1_based - 1) * incremental_stretch
        inc_linear_stretch = linear_stretch_curr / linear_stretch_prev
        F = inc_linear_stretch * F

    # Direct stretch w.r.t reference undeformed state
    else:
        F = F * linear_stretch_curr
        
    return F

##################
### SUBROUTINE ###
##################


def get_uniaxial_stretch_defgrad(
        step_idx_1_based, incremental_stretch, is_incremental,
        stretch_direction_idx=2):

    '''
    Computes incremental or direct deformation gradient for uniaxial stretch.
    step_idx_1_based : Index countig starts from 1
    '''

    F = np.eye(3)
    linear_stretch_curr = 1.0 + step_idx_1_based * incremental_stretch

    # Incremental deformation
    if is_incremental:
        linear_stretch_prev = 1.0 + (step_idx_1_based - 1) * incremental_stretch
        inc_linear_stretch = linear_stretch_curr / linear_stretch_prev
        F[stretch_direction_idx, stretch_direction_idx] = inc_linear_stretch

    # Direct stretch w.r.t reference undeformed state
    else:
        F[stretch_direction_idx, stretch_direction_idx] = linear_stretch_curr
        
    return F

##################
### SUBROUTINE ###
##################


def get_shear_defgrad(
        step_idx_1_based, incremental_shear, is_incremental, shear_type,
        plane_normal_direction_idx=2, shear_direction_idx=0):

    '''
    Computes incremental or direct shear deformation gradient.

    Currently only simple shear is implemented because it preserves the 
    orientation of the normal to plane and direction unlike pre shear that has
    a rotation part
    '''

    F = np.eye(3)

    # Simple shear
    if shear_type == 'simple':

        # Incremental deformation
        if is_incremental:
            F[shear_direction_idx, plane_normal_direction_idx] = incremental_shear

        # Direct stretch w.r.t reference undeformed state
        else:
            gamma = step_idx_1_based * incremental_shear
            F[shear_direction_idx, plane_normal_direction_idx] = gamma

        return F

    # Pure shear
    else:
        sys.stderr.write("Error: In module %s\n" %(module_name))
        sys.stderr.write("       In subroutine 'get_shear_defgrad'\n")
        sys.stderr.write("       Only simple shear is implemented\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def get_copt_flags_voigt(load_type):

   
    if load_type == 'hydrostatic_tension':
        copt_flags_voigt=None # Just run a fixed volume calculation.

    if load_type == 'normal_x':
        copt_flags_voigt=['F', 'S', 'S', 'S', 'S', 'S']

    elif load_type == 'normal_y':
        copt_flags_voigt=['S', 'F', 'S', 'S', 'S', 'S']

    elif load_type == 'normal_z':
        copt_flags_voigt=['S', 'S', 'F', 'S', 'S', 'S']

    # For shear, first and second directions are normal to plane and shear direction
    # Example shear_zx means plane normal is z axis and shear direction is x axis
    elif (load_type == 'pure_shear_zx') or (load_type == 'simple_shear_zx'):
        copt_flags_voigt=['S', 'S', 'S', 'S', 'F', 'S']

    elif (load_type == 'pure_shear_zy') or (load_type == 'simple_shear_zy'):
        copt_flags_voigt=['S', 'S', 'S', 'F', 'S', 'S']

    elif (load_type == 'pure_shear_xy') or (load_type == 'simple_shear_xy'):
        copt_flags_voigt=['S', 'S', 'S', 'S', 'S', 'F']

    return copt_flags_voigt
      
##################
### SUBROUTINE ###
##################


def setup_incr_load(work_dir, incr_load_info, batch_object=None,
          is_taskfarm=False, n_taskfarm=1, taskfarm_n_proc=None,
          n_proc_per_atom=16, calculator='vasp'):

    '''
    Sets up incremental loading job in LOAD directory.
    '''

    # Create and move to work directory
    old_dir = os.getcwd()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)

    # Incremental loading directory
    incr_load_dir = work_dir + '/' + incr_load_info['rel_dir_name']
    if not os.path.exists(incr_load_dir):
        os.makedirs(incr_load_dir)
    os.chdir(incr_load_dir)
   
    # Abridged incremental loading info for saving to file for future use
    short_incr_load_info = incr_load_info.copy()
    for key in ['rel_dir_name', 'atoms', 'calc_load', 'calc_force']:
        del short_incr_load_info[key]
    
    # Create incremental loading status dictionary
    incr_load_status = {
        'header' : 'Information about the last successful incremental loading step',
        'step_idx' : -1,
        'step_dir' : 'REFERENCE_UNRELAXED',
        'max_step_idx' : incr_load_info['max_load_steps'],
        'stress_voigt_previous' : None,
        'stress_voigt_current' : None  ,
        'load_success_dir_list' : []      
    }
    
    n_proc_local = len(incr_load_info['atoms'].positions) * n_proc_per_atom
    n_core = 1
    if (n_proc_local == 4) or (n_proc_local == 8):
        n_core = 2
    elif (n_proc_local == 12) or (n_proc_local == 16) or (n_proc_local == 24):
        n_core = 4
    else:
        n_core = int(n_proc_local / n_proc_per_atom)

    incr_load_info['calc_load'].set(ncore=n_core)
    incr_load_info['calc_load'].set(kpar=4)
    
    sim_info = mod_calculator.template_sim_info.copy()
    sim_info['calculator'] = calculator
    if incr_load_info['calc_load'].int_params['isif'] == 4:
        sim_info['sim_type'] = mod_vasp.get_sim_type_from_isif(incr_load_info['calc_load'].int_params['isif'])
    else:
        sim_info['sim_type'] = 'copt'
    sim_info['n_proc'] = n_proc_local
    sim_info['n_relax_iter'] = 1
    sim_info['is_final_static'] = False
    sim_info['is_direct_coordinates'] = True
    
    # COPT is only needed for non hydrostatic cases
    if sim_info['sim_type'] != 'relax_is': #(ISIF=4)
        # Create a COPT input. Default values are not set
        copt_input = mod_constrained_optimization.template_copt_input.copy()
        copt_input['flags_voigt'] = incr_load_info['copt_flags_voigt']
        copt_input['target_stress_voigt'] =  incr_load_info['copt_target_stress_voigt']

        # Attach the COPT input to the sim_info
        sim_info['sim_specific_input'] = copt_input

    # Create Reference directory with necessary inputs
    mod_dft.setup_basic_dft_calculations(
        work_dir=incr_load_dir,
        rel_dir_list=["REFERENCE_UNRELAXED"],
        structure_list=[incr_load_info['atoms']],
        calculator_input_list=[incr_load_info['calc_load']],
        sim_info_list = [sim_info],
        batch_object = batch_object)

    # Delete unnecessary files in the REFERENCE directory
    os.chdir('REFERENCE_UNRELAXED')

    # Setup DFPT calculations if required (just for INCAR_DFPT)
    mod_calculator.pre_setup_force_constants(
        calculator = calculator,
        in_atoms=incr_load_info['atoms'],
        in_calc_fc=incr_load_info['calc_force'],
        super_cell=None, work_dir="./",
        force_constant_method='dfpt', is_sym=True) # NOTE: is_sym=True uses IBRION = 8 else IBRION = 7 is used

    # Delete unnecessary files
    mod_utility.delete_files(file_list=['ase-sort.dat', '#DO_FINAL_STATIC#', 'job_driver.py', 'submit.sh', '#WAITING#', 'job_info.json'])
    os.chdir('../')

    # This submit script is for the actual incremental loading job in the work_dir (the previous ones are dummy inside the REFERENCE dir)
    # Write a submit script (only the header)
    batch_object.ntasks = n_proc_local
    mod_batch.generate_submit_script(batch_object=batch_object)

    # Append job_driver.py processing call to the slurm file
    with open(batch_object.file_name, 'a') as fh:
        fh.write("python job_driver.py %s\n"  % (mod_batch.n_proc_str()))

    # Create job_driver.py (header)
    mod_batch.write_job_driver(work_dir="./")

    # Append commands specific to sim_type to job_driver.py
    with open("job_driver.py", "a") as fh:
        fh.write("from matkit.core import mod_incr_load\n"
                 "mod_incr_load.run_incr_load_job"\
                 "(work_dir='./', n_proc=n_proc, is_final_static=False, hostfile=hostfile,"\
                 " parset_filename=None, n_relax_iter=1, calculator='%s')\n" %(calculator))

    # Add a waiting tag
    open("#WAITING#", 'a').close()
    
    # Initially do a incremental loading
    # n_proc inside each job will later be picked up by setup_tasfarm in mod_batch
    # to enable setting different processors for different tasks in the taskfarm
    job_info_dict = {
        'n_proc' : n_proc_local,
    }
    with open('job_info.json', 'w') as fh:
        json.dump(job_info_dict, fh, indent=4, cls=mod_utility.json_numpy_encoder)

    # Store incremental loading information
    with open('incr_load_info.json', 'w') as fh:
        json.dump(short_incr_load_info, fh, indent=4, cls=mod_utility.json_numpy_encoder)

    # Store incremental loading status information
    with open('incr_load_status.json', 'w') as fh:
        json.dump(incr_load_status, fh, indent=4, cls=mod_utility.json_numpy_encoder)

    # Move back to where you were
    os.chdir(old_dir)
    
##################
### SUBROUTINE ###
##################


def setup_force_constant_calculations(
        abs_load_incr_load_dir, abs_force_constants_dir, incr_load_info=None, n_proc_per_atom=4, job_driver_filename='job_driver.py'):

    '''
    Once the incremental loading run is completed (or to the extent completed),
    this subroutine sets up force constants calculations in a separate parallel
    directory structure.
    
                   ROOT (Ex: FCC)
                LOAD            FORCE_CONSTANTS
              -- mode1                -- mode1
                 -- STEP1                -- STEP1
                 -- STEP2 ...            -- STEP2 ...
              -- mode2 ...            -- mode2
    
    Inputs:
      abs_load_base_dir: Where the incremental loading run for a certain loading mode is present
      abs_force_constants_dir: Where the force constants input are to be set for each loading value corresponding to the incremental loading
    '''

    # No moving to any directories, stay where you are    
    if not os.path.exists(abs_force_constants_dir):
        os.makedirs(abs_force_constants_dir)

    # Read incremental loading info if not provided
    if incr_load_info is None:
        with open(abs_load_incr_load_dir + '/incr_load_info.json') as fh:
            incr_load_info = json.load(fh)
    else:
        incr_load_info = incr_load_info
        
    # Read incremental loading status to get the loading step directories
    with open(abs_load_incr_load_dir + '/incr_load_status.json') as fh:
        incr_load_status = json.load(fh)
        
    # Read job info of the incremental loading to get the number of perocessors per unit cell
    with open(abs_load_incr_load_dir + '/job_info.json') as fh:
        incr_load_job_info = json.load(fh)

    # Create phonon incremental loading for each super cell size   
    for supercell in incr_load_info['phonon_supercell_size_list']:
    
    
        abs_fc_supercell_dir = abs_force_constants_dir + '/FC_' + '_'.join([str(x) for x in supercell])
        
        # Create force constants status dictionary
        force_constants_incr_load_status = {
            'header' : 'Information about the last successful force constants step',
            'step_idx' : -1,
            'max_step_idx' : incr_load_status['max_step_idx'], # Limit to the maximum theoretical strength, no point of doing the additional point after max strength is reached
            'force_constants_success_dir_list' : []      
        }
        
        # Create force constants status dictionary
        force_constants_incr_load_info = {
            'header' : 'Information about force constants along incremental loading path',
            'load_success_dir_list' : incr_load_status['load_success_dir_list']
        }

        if not os.path.exists(abs_fc_supercell_dir) and (incr_load_status['step_idx'] > 0):
        
            # Set up phonon calculations corresponding to each of the successfully computed incremental loading directories
            for idx, rel_load_step_dir in enumerate(incr_load_status['load_success_dir_list'], start=0):
        
                abs_load_step_dir = abs_load_incr_load_dir + '/' + rel_load_step_dir
                abs_fc_step_dir = abs_fc_supercell_dir + '/' + rel_load_step_dir
                
                # Copy DFPT incar from REFERENCE to current directory
                shutil.copyfile(abs_load_incr_load_dir + '/REFERENCE_UNRELAXED/INCAR_DFPT', abs_load_step_dir + '/INCAR_DFPT')
            
                mod_calculator.setup_force_constants(
                    super_cell=supercell, src_dir=abs_load_step_dir, dest_dir=abs_fc_step_dir, force_constant_method='dfpt')
                  
                #NOTE: For some reaosn setting NCORE in DFPT calculations results in error in VASP due to k-point change
                mod_utility.replace(file_path=abs_fc_step_dir + '/INCAR',
                                    pattern='NCORE', subst='', replace_entire_line=True)
                                    
                # Remove unnecessary job_deriver.py file in DFPT directory
                mod_utility.delete_files(file_list=[abs_fc_step_dir + '/job_driver.py'])              
                '''
                # Update NCORE and KPAR parameters in the INCAR file
                ncore = n_proc_per_supercell / n_proc_per_atom
                with open(abs_fc_dir + '/INCAR', 'a') as fh:
                    fh.write(" NCORE = %d\n" %(ncore))
                    fh.write(" KPAR = 2")
                '''
                  
            # Add a waiting tag
            open(abs_fc_supercell_dir + "/#WAITING#", 'a').close()
            n_proc_per_supercell=incr_load_job_info['n_proc']*supercell[0]
            job_info_dict = {
                'n_proc' : n_proc_per_supercell
            }
            with open(abs_fc_supercell_dir + '/job_info.json', 'w') as fh:
                json.dump(job_info_dict, fh, indent=4, cls=mod_utility.json_numpy_encoder)                

            # Store force constants incremental loading status
            with open(abs_fc_supercell_dir + '/force_constants_incr_load_status.json', 'w') as fh:
                json.dump(force_constants_incr_load_status, fh, indent=4, cls=mod_utility.json_numpy_encoder)
                
            # Store force constants incremental loading information
            with open(abs_fc_supercell_dir + '/force_constants_incr_load_info.json', 'w') as fh:
                json.dump(force_constants_incr_load_info, fh, indent=4, cls=mod_utility.json_numpy_encoder)                
                
            # Setup job driver exclusiverly to DFPT
            mod_batch.write_job_driver(work_dir=abs_fc_supercell_dir)
    
            # Append DFPT processing instruction to the job driver in work_dir
            with open(abs_fc_supercell_dir + '/' + job_driver_filename, 'a') as fh:
                fh.write("from matkit.core import mod_incr_load\n")
                fh.write("mod_incr_load.run_force_constants_incr_load_job"\
                  "(work_dir='%s', n_proc=n_proc, hostfile=hostfile,"\
                  " parset_filename=None)\n" %(abs_fc_supercell_dir))

    return

##################
### SUBROUTINE ###
##################


def is_terminate_stress_based(loading_spec_list, stress_voigt_previous,
                              stress_voigt_current):
               
    '''
      This subroutine given a loading specification and the stress at the
      current and previous loading steps from DFT, tells if the ideal strength
      if the appropriate stress measure at the current loading step is less
      than that at the previous step.
    '''
   
    stress_measure_prev = get_stress_measure(loading_spec_list, stress_voigt_previous)
    stress_measure_curr = get_stress_measure(loading_spec_list, stress_voigt_current)    

    if stress_measure_prev > stress_measure_curr:
        return True
    else:
        return False
        
##################
### SUBROUTINE ###
##################


def get_stress_measure(loading_spec_list, stress_voigt):

    '''
      This subroutine, given a loading specification and the stress result from
      DFT, returns the appropriate stress measure.
    '''
    
    stress_measure = None
    
    if loading_spec_list[0] == 'hydrostatic_tension':
    
        stress_measure = (stress_voigt[0] + stress_voigt[1] + stress_voigt[2]) / 3.0
   
    elif loading_spec_list[0] == 'normal':
    
        idx = loading_spec_list[1]
        sign = math.copysign(1, loading_spec_list[2])
        stress_measure = sign * stress_voigt[idx]
    
    elif loading_spec_list[0] == 'shear':
    
        normal_idx = loading_spec_list[2]
        direction_idx = abs(loading_spec_list[3])
        sign = math.copysign(1, loading_spec_list[3])
        idx = None
        if ((normal_idx == 1) and (direction_idx == 2)) or ((normal_idx == 2) and (direction_idx == 1)):
            idx = 3
        elif ((normal_idx == 0) and (direction_idx == 2)) or ((normal_idx == 2) and (direction_idx == 0)):
            idx = 4
        elif ((normal_idx == 0) and (direction_idx == 1)) or ((normal_idx == 1) and (direction_idx == 0)):
            idx = 5
            
        stress_measure = sign * stress_voigt[idx]

    return stress_measure
    
##################
### SUBROUTINE ###
##################


def get_strain_measures(loading_spec_list, def_grad):

    '''
      This subroutine, given a loading specification and the deformation
      gradient results from DFT, returns the appropriate strain measures.
      
      1. Green Lagrange strain component
      2. Small strain component
    '''
    
    strain_measure_nonlinear = None
    strain_measure_linear = None
    
    # Green - Lagrange strain tensor
    E = mod_tensor.green_lagrange_strain_tensor(def_grad)
    
    # Small strain tensor
    e = mod_tensor.small_strain_tensor(def_grad)
   
    if loading_spec_list[0] == 'hydrostatic_tension':
    
        strain_measure_nonlinear = np.linalg.det(def_grad)
        strain_measure_linear = e[0][0] + e[1][1] + e[2][2]
   
    elif loading_spec_list[0] == 'normal':
    
        idx = loading_spec_list[1]
        sign = math.copysign(1, loading_spec_list[2])
        
        strain_measure_nonlinear = sign * E[idx][idx] # Why sgn ? Think about it
        strain_measure_linear = sign * e[idx][idx] # Why sgn ? Think about it
    
    elif loading_spec_list[0] == 'shear':
    
        normal_idx = loading_spec_list[2]
        direction_idx = abs(loading_spec_list[3])
        sign = math.copysign(1, loading_spec_list[3])
            
        # NOTE: shear strain Gamma = 2 \epsilon_{ij}, so using 2
        strain_measure_nonlinear = 2.0 * sign * E[normal_idx][direction_idx]
        strain_measure_linear = 2.0 * sign * e[normal_idx][direction_idx]
    
    return (strain_measure_nonlinear, strain_measure_linear)

##################
### SUBROUTINE ###
##################


def run_incr_load_job(
        work_dir, n_proc, is_final_static=False, n_relax_iter=1, hostfile=None,
        parset_filename=None, is_remove_final_static=True, calculator='vasp'):
        
    '''
    Runs a single incremental loading job
    '''

    # Move to the work_dir
    old_dir = os.getcwd()
    os.chdir(work_dir)

    # Check if there is a #WAITING# tag file
    if os.path.isfile("#WAITING#"):
        os.rename("#WAITING#", "#PROCESSING#")
    else:
        return False

    # Read incremental loading info
    with open('incr_load_info.json') as fh:
        incr_load_info = json.load(fh)

    # Read incremental load status
    with open('incr_load_status.json') as fh:
        incr_load_status = json.load(fh)

    # STEP COUNTING:
    # -1 => REFERENCE (unrelaxed, no calculation will be done)
    #  0 => REFERENCE_RELAXED, relaxed to applied stresses with all [S, S, S, S, S, S] before any strain is applied
    # 1, 2, ... n => Loading steps       

    # Incremental loading loop
    while (incr_load_status['step_idx'] < incr_load_status["max_step_idx"]):

        # Incremnetal loading step directory
        curr_step_dir = "STEP_" + str(incr_load_status['step_idx']+1).zfill(3)

        # Get strain with respect to the previos step
        if incr_load_status['step_idx'] == -1:
            prev_atoms = read_vasp(incr_load_status['step_dir'] + '/POSCAR')
            strained_atoms = prev_atoms
            F = np.eye(3)

        else:
            # Get deformation gradient
            if incr_load_info['loading_spec_list'][0] == 'hydrostatic_tension':
                F = get_hydrostatic_defgrad(step_idx_1_based=incr_load_status['step_idx']+1,
                                            incremental_stretch=incr_load_info['incremental_deformation_parameter'],
                                            is_incremental=incr_load_info['is_incremental_loading'])
            
            elif incr_load_info['loading_spec_list'][0] == 'normal':
                F = get_uniaxial_stretch_defgrad(step_idx_1_based=incr_load_status['step_idx']+1,
                                                 incremental_stretch=incr_load_info['incremental_deformation_parameter'],
                                                 is_incremental=incr_load_info['is_incremental_loading'],
                                                 stretch_direction_idx=incr_load_info['loading_spec_list'][1])

            elif incr_load_info['loading_spec_list'][0] == 'shear':
                F = get_shear_defgrad(step_idx_1_based=incr_load_status['step_idx']+1,
                                      incremental_shear=incr_load_info['incremental_deformation_parameter'],
                                      is_incremental=incr_load_info['is_incremental_loading'],
                                      shear_type=incr_load_info['loading_spec_list'][1],
                                      plane_normal_direction_idx=incr_load_info['loading_spec_list'][2], 
                                      shear_direction_idx=incr_load_info['loading_spec_list'][3])

            else:
                sys.stderr.write("Error: In module '%s'\n" %(module_name))
                sys.stderr.write("       In subroutine 'run_incr_load_job'\n")
                sys.stderr.write("       Unknown loading type in loading_spec_list\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)
            
            # Strain atoms
            prev_atoms = read_vasp(incr_load_status['step_dir'] + '/CONTCAR')
            strained_atoms = mod_ase.defgrad_ase_atoms(in_atoms=prev_atoms, F=F)

        # Setup DFT directory: If current directory exists, remove it and create new one (should be done for restart calulcations)
        if os.path.isdir(curr_step_dir):
            shutil.rmtree(curr_step_dir)
        os.mkdir(curr_step_dir)

        # Copy INCAR, POTCAR from reference to current
        shutil.copyfile('REFERENCE_UNRELAXED/INCAR', curr_step_dir + '/INCAR')
        shutil.copyfile('REFERENCE_UNRELAXED/POTCAR', curr_step_dir + '/POTCAR')
        
        # For the first relaxation we will not apply any strain and relax the structure to the input stresses (if any : given by mode_id)
        if incr_load_status['step_idx'] == -1:
            if os.path.isfile('REFERENCE_UNRELAXED/COPT_INPUT'):
                copt_input = mod_constrained_optimization.template_copt_input.copy()
                copt_input['flags_voigt'] = ['S', 'S', 'S', 'S', 'S', 'S']
                copt_input['target_stress_voigt'] =  incr_load_info['copt_target_stress_voigt']
                mod_constrained_optimization.write_constrained_optimization_input(copt_input_dict=copt_input, copt_input_file_path=curr_step_dir + '/COPT_INPUT')
            else:
                # For hydrostatic loading, relax to zero pressure first
                mod_utility.replace(file_path= curr_step_dir + '/INCAR',
                                    pattern='ISIF', subst='ISIF=3',
                                    replace_entire_line=True)
        else:
            if os.path.isfile('REFERENCE_UNRELAXED/COPT_INPUT'):
                shutil.copyfile('REFERENCE_UNRELAXED/COPT_INPUT', curr_step_dir + '/COPT_INPUT')

        # Copy KPOINTS from reference to current (Sometimes KSPACING may be provided in INCAR)
        if os.path.isfile('REFERENCE_UNRELAXED/KPOINTS'):
            shutil.copyfile('REFERENCE_UNRELAXED/KPOINTS', curr_step_dir + '/KPOINTS')
        
        # WAVECAR has no use expect for starting next incremental loading job, so move it altogether to current directory from previous if present
        if os.path.isfile(incr_load_status['step_dir'] + '/WAVECAR'):
            shutil.move(incr_load_status['step_dir'] + '/WAVECAR', curr_step_dir + '/WAVECAR')            

        # CHGCAR has further use at each step for plotiing charge densities, so copy it to current from previous
        if os.path.isfile(incr_load_status['step_dir'] + '/CHGCAR'):
            shutil.copyfile(incr_load_status['step_dir'] + '/CHGCAR', curr_step_dir + '/CHGCAR')
                    
        # Create a waiting file
        open(curr_step_dir + '/#WAITING#', 'a').close()

        # Copy hostfile if needed. If it exists, it is in the work_dir
        if hostfile is not None:
            shutil.copyfile(hostfile, curr_step_dir + '/' + hostfile)

        # Move to the current step directory
        os.chdir(curr_step_dir)

        # Create POSCAR in the current directory
        write_vasp(file='POSCAR', atoms=strained_atoms)

        # Run DFT job
        time.sleep(CONFIG.SLEEP_TIME)
        curr_dir_abs_path = os.getcwd()
        if os.path.isfile('COPT_INPUT'):
            mod_constrained_optimization.run(work_dir=curr_dir_abs_path, n_proc=n_proc, hostfile=hostfile)
        else:
            mod_calculator.run_dft_job(calculator=calculator, n_proc=n_proc, n_iter=n_relax_iter,
                is_final_static=is_final_static, hostfile=hostfile,
                wdir=curr_dir_abs_path, parset_filename=parset_filename)
            
        # Clean up the current directory, for space storage
        mod_vasp.clean(file_list=incr_load_clean_file_list)
        if os.path.isdir('FINAL_STATIC'):
            if is_remove_final_static:
                shutil.rmtree('FINAL_STATIC')
            else:
                os.chdir('FINAL_STATIC')
                mod_vasp.clean(file_list=incr_load_clean_file_list.extend(['WAVECAR', 'CHGCAR']))
                os.chdir('../')

        # Check convergence
        # NOTE: If a run fails to converge delete the directory
        # Always read relaxation results
        with open('dft_results_0.json') as fh:
            dft_results_0 = json.load(fh)
        
        # Change back to work_dir
        os.chdir('../')

        # Update status (Assume successful step), modify this later
        # step_idx and other fields in status should reflect the successfully completed step after this update
        is_converged = True
        if is_converged:

            is_terminate = False
            # Modify termination step index if stress based termination
            # NOTE: Atleast 2 steps must be finished to enter this, i.e. 0th and 1st
            if (incr_load_info['termination_criterion'] == 'MAX_STRESS') and \
                    (incr_load_status['step_idx']+1 < incr_load_status['max_step_idx']) and \
                    (incr_load_status['step_idx']>=1):
                    
                is_terminate = is_terminate_stress_based(
                    loading_spec_list=incr_load_info['loading_spec_list'],
                    stress_voigt_previous=incr_load_status['stress_voigt_current'],
                    stress_voigt_current=dft_results_0['stress_voigt_GPa'])
                    
            if is_terminate:
                # Rename current step as last step
                last_step_dir = "STEP_" + str(incr_load_status['step_idx']+2).zfill(3)
                os.rename(curr_step_dir, last_step_dir)
                incr_load_status['max_step_idx'] = incr_load_status['step_idx'] + 1
                incr_load_info['incremental_deformation_parameter'] = incr_load_info['incremental_deformation_parameter'] / 2.0
                incr_load_status['load_success_dir_list'].append(last_step_dir)
            else:

                incr_load_status['step_idx'] = incr_load_status['step_idx'] + 1
                incr_load_status['step_dir'] = curr_step_dir
                incr_load_status['load_success_dir_list'].append(curr_step_dir)

                # Update stresses
                if incr_load_status['step_idx']>0:
                    incr_load_status['stress_voigt_previous'] = incr_load_status['stress_voigt_current'].copy()
                incr_load_status['stress_voigt_current'] = dft_results_0['stress_voigt_GPa'].copy()

        incr_load_status['load_success_dir_list'].sort()
        with open('incr_load_status.json', 'w') as fh:
            json.dump(incr_load_status, fh, indent=4, cls=mod_utility.json_numpy_encoder)
            
    # Change the processing tag in work_dir to done
    os.rename("#PROCESSING#", "#LOAD_DONE#")

    # Move to old directory where we first began
    os.chdir(old_dir)
    
##################
### SUBROUTINE ###
##################


def run_force_constants_incr_load_job(
        work_dir, n_proc, hostfile=None, parset_filename=None):
        
    '''
      Runs the force constants incremental loading job for a particular loading and a supercell size
    '''

    # Move to the work_dir
    old_dir = os.getcwd()
    os.chdir(work_dir)

    # Check if there is a #WAITING# tag file
    if os.path.isfile("#WAITING#"):
        os.rename("#WAITING#", "#PROCESSING#")
    else:
        return False

    # Read force constants incremental loading status
    with open('force_constants_incr_load_status.json') as fh:
        force_constants_incr_load_status = json.load(fh)
        
    # Read force constants incremental loading information
    with open('force_constants_incr_load_info.json') as fh:
        force_constants_incr_load_info = json.load(fh)        
        
    # Force constant incremental loading loop
    while (force_constants_incr_load_status['step_idx'] < force_constants_incr_load_status["max_step_idx"]):
        
        # incremental loading relative directory
        curr_step_dir = force_constants_incr_load_info['load_success_dir_list'][ force_constants_incr_load_status['step_idx'] + 1 ]
        
        # Copy hostfile if needed. If it exists, it is in the work_dir
        if hostfile is not None:
            shutil.copyfile(hostfile, curr_step_dir + '/' + hostfile)
        
        mod_vasp.run_dfpt(work_dir=curr_step_dir, n_proc=n_proc, hostfile=hostfile)
        
        is_success  = True

        if is_success:
            force_constants_incr_load_status['step_idx'] = force_constants_incr_load_status['step_idx'] + 1
            force_constants_incr_load_status['force_constants_success_dir_list'].append(curr_step_dir)
            is_terminate = False
            if is_terminate:
                force_constants_incr_load_status["max_step_idx"] = force_constants_incr_load_status['step_idx']
                
        with open('force_constants_incr_load_status.json', 'w') as fh:
            json.dump(force_constants_incr_load_status, fh, indent=4, cls=mod_utility.json_numpy_encoder)

    # Change the processing tag in work_dir to done
    os.rename("#PROCESSING#", "#FORCE_CONSTANTS_DONE#")

    # Move to old directory where we first began
    os.chdir(old_dir)        
        
        
##################
### SUBROUTINE ###
##################


def process_incr_load(work_dir):

    # Move to the work_dir
    old_dir = os.getcwd()
    os.chdir(work_dir)
    
    print(f"Processing directory: {work_dir}")

    # Read incremental loading info
    with open('incr_load_info.json') as fh:
        incr_load_info = json.load(fh)
        
    # Read incremental loading status
    with open('incr_load_status.json') as fh:
        incr_load_status = json.load(fh)

    # Results dictionary
    incr_load_results = {
        'energy_eV_list' : [],
        'volume_A3_list' : [],
        'stress_voigt_atom_basis_GPa_list' : [],
        'resulting_def_grad_list' : [],
        'stress_measure_list' : [],
        'nonlinear_strain_measure_list' : [],
        'linear_strain_measure_list' : [],
        'is_calculation_complete' : False,
        'superposed_stress_voigt_GPa' : incr_load_info['copt_target_stress_voigt'],
        'maximum_stress_GPa' : None,
        'n_atoms' : None
    }

    # Concatenated raw results
    concatenated_raw_results = {
        "dft_results_0_list": [],
        "dft_results_1_list": []
    }

    reference_cell_vecs_row_wise = None
    # Read results from each step
    for idx, curr_step_dir in enumerate(incr_load_status['load_success_dir_list'], start=0):

        # Read stresses from dft_results_0
        with open(curr_step_dir + '/dft_results_0.json') as fh:
            dft_results_0 = json.load(fh)
            
        # Save to raw results
        concatenated_raw_results["dft_results_0_list"].append(dft_results_0)

        # Volume of the cell
        incr_load_results['volume_A3_list'].append(dft_results_0['volume_A3'])

        # Just copy the number of atoms once and get the first reference cell
        if idx == 0:
            incr_load_results['n_atoms'] = dft_results_0['n_atoms']
            # NOTE: cell vectors are row wise
            reference_cell_vecs_col_wise = np.array(dft_results_0['cellvecs_A']).transpose()

        current_cell_vecs_col_wise = np.array(dft_results_0['cellvecs_A']).transpose()
        def_grad = np.matmul(current_cell_vecs_col_wise, np.linalg.inv(reference_cell_vecs_col_wise))  
        incr_load_results['resulting_def_grad_list'].append( def_grad )
        
        # Stress in atom basis
        stress_voigt_atom_basis = dft_results_0['stress_voigt_GPa']
        incr_load_results['stress_voigt_atom_basis_GPa_list'].append(stress_voigt_atom_basis)
        
        # Stress measure
        stress_measure = get_stress_measure(loading_spec_list=incr_load_info['loading_spec_list'], stress_voigt=stress_voigt_atom_basis)
        incr_load_results['stress_measure_list'].append(stress_measure)
        
        # Strain measures
        [strain_measure_nonlinear, strain_measure_linear] = get_strain_measures(loading_spec_list=incr_load_info['loading_spec_list'], def_grad=def_grad)   
        incr_load_results['nonlinear_strain_measure_list'].append(strain_measure_nonlinear) 
        incr_load_results['linear_strain_measure_list'].append(strain_measure_linear)

        # Read energy from dft_results_1 if it exists or else read from dft_results_0
        if os.path.isfile(curr_step_dir + '/dft_results_1.json'):
            with open(curr_step_dir + '/dft_results_1.json') as fh:
                dft_results_1 = json.load(fh)
            incr_load_results['energy_eV_list'].append(dft_results_1['energy_eV'])
            
            concatenated_raw_results['dft_results_1_list'].append(dft_results_1)
        else:
            incr_load_results['energy_eV_list'].append(dft_results_0['energy_eV'])
            concatenated_raw_results["dft_results_1_list"].append(None)

    if len(incr_load_results['stress_measure_list']) > 0:
        incr_load_results['maximum_stress_GPa'] = np.amax(np.array(incr_load_results['stress_measure_list']))
        
        # If stress increases and then decreases, that is a sign of completion of the calculation
        if np.argmax(np.array(incr_load_results['stress_measure_list'])) < len(np.array(incr_load_results['stress_measure_list'])) - 1:
            incr_load_results['is_calculation_complete'] = True
        

    # Store results
    with open('incr_load_results.json', 'w') as fh:
        json.dump(incr_load_results, fh, indent=4, cls=mod_utility.json_numpy_encoder)
        
    with open('concatenated_raw_results.json', 'w') as fh:
        json.dump(concatenated_raw_results, fh, indent=4,
                  cls=mod_utility.json_numpy_encoder)

    os.chdir(old_dir)
    
##################
### SUBROUTINE ###
##################


def read_results(work_dir):

    '''
    This subroutine is useful to extract results from already processed results
    of each incremental loading.
    
    For eaxmple, we study the effect of variying biaxial stress or multiaxial
    stresses on each incremental loading. In such cases, for each incremental loading, we wish
    to know the how the theoretical strength varied as a function of the 
    superposed stress.
    '''

    # Read incremental loading results
    with open(work_dir + '/incr_load_results.json') as fh:
        incr_load_results = json.load(fh)

    results = {
        'superposed_stress_voigt_GPa' : incr_load_results['superposed_stress_voigt_GPa'],
        'maximum_stress_GPa' : incr_load_results['maximum_stress_GPa'],
        'is_calculation_complete' : incr_load_results['is_calculation_complete']
    }

    return results
    
##################
### SUBROUTINE ###
##################


def plot_loading(work_dir, strain_measure_type='linear', plot_dir='plots'):

    '''
      Plots loading stress vs strain and energy per atom versus strain diagrams
      NOTE: This subroutine works for only precoded lattice types such as fcc, cubic-diamond (same as fcc), bcc and hcp
    '''  
    
    # Move to the work_dir
    old_dir = os.getcwd()
    os.chdir(work_dir)
    
    # Open any of the incr_load_info file and get the system or lattice_type or loading_system
    with open('incr_load_rel_dir_info.json', 'r') as fh:
        incr_load_rel_dir_list = json.load(fh)['incr_load_rel_dir_list']
        
    with open(incr_load_rel_dir_list[0] + '/incr_load_info.json') as fh:
        incr_load_info = json.load(fh)
        system = incr_load_info['system']
        material = incr_load_info['material']
    
    # Read the dictionaries  
    plot_info_dict = all_loading_plot_info_dict[system]
    loading_system_info_dict = all_loading_system_info_dict[system]
    
    mod_utility.warn_create_dir(plot_dir, is_user_prompt=False)
    
    # For each of the loading family (i.e hydrostatic, normal or shear) and plot the results in each plot
    for loading_family in plot_info_dict:
    
        loading_system_list = []
        strain_sign_list = []
        line_color_list = []
        marker_type_list = []
        marker_face_color_list = []
        
        for idx, (loading_system, strain_sign, line_color, marker_type, marker_face_color) in \
                enumerate(zip(plot_info_dict[loading_family]['loading_system_list'],\
                              plot_info_dict[loading_family]['strain_sign_list'],\
                              plot_info_dict[loading_family]['line_color_list'],\
                              plot_info_dict[loading_family]['marker_type_list'],\
                              plot_info_dict[loading_family]['marker_face_color_list']), start=0):
        
            for rel_dir in incr_load_rel_dir_list:
                if loading_system in rel_dir:
                    loading_system_list.append(loading_system)
                    strain_sign_list.append(strain_sign)
                    line_color_list.append(line_color) 
                    marker_type_list.append(marker_type)
                    marker_face_color_list.append(marker_face_color)
        
        x_label = plot_info_dict[loading_family]['x_label']
        y_label = plot_info_dict[loading_family]['y_label']
        
        legend_label_list = []
        for loading_system in loading_system_list:
            legend_label = loading_system_info_dict[loading_system]['label']
            legend_label_list.append(legend_label)
            
        stress_legend_label_list = legend_label_list.copy()
        
        # Read necessary results from each loading system in the current loading family
        stress_measure_arr_list = []
        strain_measure_arr_list = []
        energy_per_atom_arr_list = []
        
        # Salient points
        dict_salient_points = {
            "elastic_stability_limit_stress_measure_list" : [],
            "elastic_stability_limit_strain_measure_list" : [],
            "approx_elastic_stability_limit_stress_measure_list" : [],
            "approx_elastic_stability_limit_strain_measure_list" : [],            
            "phonon_stability_limit_stress_measure_list" : [],
            "phonon_stability_limit_strain_measure_list" : []            
        }
        
        for idx, (loading_system, strain_sign) in enumerate(zip(loading_system_list, strain_sign_list), start=0):

            # Get the relative directory, rel dir need not be in the same order of loading system list (use loading system list as reference for all ordering)
            incr_load_rel_dir = None
            for rel_dir in incr_load_rel_dir_list:

                if loading_system in rel_dir:
                    incr_load_rel_dir = rel_dir
                    break
            if incr_load_rel_dir is None:
                continue
                    
            with open(incr_load_rel_dir + '/incr_load_results.json') as fh:
                results = json.load(fh)
              
            # Read stress measure along the incremental loading path  
            stress_measure_arr_list.append( np.array(results['stress_measure_list']) )
            
            max_stress = results['maximum_stress_GPa']
            stress_legend_label_list[idx] = stress_legend_label_list[idx] + ', ' + plot_info_dict[loading_family]['maximum_stress_label'] + ' : ' + '%.2f' %(max_stress)
            
            # Read the desired strain measure along the incremental loading path
            if strain_measure_type == 'linear':
                strain_measure_arr_list.append( strain_sign * np.array(results['linear_strain_measure_list']) )
                
            elif strain_measure_type == 'nonlinear':
                strain_measure_arr_list.append( strain_sign * np.array(results['nonlinear_strain_measure_list']) )
                
            # Read the energy per atom along the incremental loading path
            energy_per_atom_arr = np.array( results['energy_eV_list'] ) / results['n_atoms']
            energy_per_atom_arr_list.append(energy_per_atom_arr)
            
            # Read salient points
            if "elastic_stability_limit_stress_measure" in results:
                dict_salient_points["elastic_stability_limit_stress_measure_list"].append( results["elastic_stability_limit_stress_measure"] )
                dict_salient_points["elastic_stability_limit_strain_measure_list"].append( results["elastic_stability_limit_strain_measure"] )
            else:
                dict_salient_points["elastic_stability_limit_stress_measure_list"].append( None )
                dict_salient_points["elastic_stability_limit_strain_measure_list"].append( None )
                
            if "approx_elastic_stability_limit_stress_measure" in results:
                dict_salient_points["approx_elastic_stability_limit_stress_measure_list"].append( results["approx_elastic_stability_limit_stress_measure"] )
                dict_salient_points["approx_elastic_stability_limit_strain_measure_list"].append( results["approx_elastic_stability_limit_strain_measure"] )
            else:
                dict_salient_points["approx_elastic_stability_limit_stress_measure_list"].append( None )
                dict_salient_points["approx_elastic_stability_limit_strain_measure_list"].append( None )                
                
            if "phonon_stability_limit_stress_measure" in results:
                dict_salient_points["phonon_stability_limit_stress_measure_list"].append( results["phonon_stability_limit_stress_measure"] )
                dict_salient_points["phonon_stability_limit_strain_measure_list"].append( results["phonon_stability_limit_strain_measure"] )
            else:
                dict_salient_points["phonon_stability_limit_stress_measure_list"].append( None )
                dict_salient_points["phonon_stability_limit_strain_measure_list"].append( None )                
            
            
        # Plot stress vs strain
        plot_strength(loading_family=loading_family,
                          x_label=x_label,
                          y_label=y_label,
                          x_arr_list=strain_measure_arr_list,                          
                          y_arr_list=stress_measure_arr_list,
                          dict_salient_points=dict_salient_points,
                          legend_label_list=stress_legend_label_list,
                          legend_title = r'{\Large\textbf{' + material + '}} (' + system + ')',
                          line_color_list=line_color_list,
                          marker_type_list=marker_type_list,
                          marker_face_color_list=marker_face_color_list,
                          plot_filename=plot_dir + '/' + material + '_' + loading_family + '_stress.pdf')
                          
        # Plot energy per atom vs strain
        plot_strength(loading_family=loading_family,
                          x_label=x_label,
                          y_label='Energy per atom (eV)',
                          x_arr_list=strain_measure_arr_list,                          
                          y_arr_list=energy_per_atom_arr_list,
                          dict_salient_points=None,                          
                          legend_label_list=legend_label_list,
                          legend_title = r'\textbf{' + material + '} (' + system + ')',
                          line_color_list=line_color_list,
                          marker_type_list=marker_type_list,
                          marker_face_color_list=marker_face_color_list,
                          plot_filename=plot_dir + '/' + material + '_' + loading_family + '_energy.pdf')                 
        
    # Move back to where you were
    os.chdir(old_dir)
    
    return

##################
### SUBROUTINE ###
##################


def plot_strength(
        loading_family, x_label, y_label, x_arr_list, y_arr_list, dict_salient_points,
        legend_label_list, line_color_list, marker_type_list, marker_face_color_list, plot_filename, fontsize=14, legend_title=None):
        
    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(frame_on=True)
    
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top=False, labelsize=12)
    ax.xaxis.set_tick_params(which='minor', size=2.5, width=0.5, direction='in', top=False)
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right=False, labelsize=12)
    ax.yaxis.set_tick_params(which='minor', size=2.5, width=0.5, direction='in', right=False)
    
    for (x_arr, y_arr, legend_label, line_color, marker_type, marker_face_color) in zip(x_arr_list, y_arr_list, legend_label_list, line_color_list, marker_type_list, marker_face_color_list):
        plt.plot(x_arr, y_arr, label=legend_label, color=line_color, linewidth=1.25, marker=marker_type, markerfacecolor=marker_face_color, markersize=4)
        
    # Elastic stability limit
    if dict_salient_points is not None:
        for (elastic_stability_limit_stress_measure, elastic_stability_limit_strain_measure) in \
                zip( dict_salient_points["elastic_stability_limit_stress_measure_list"], dict_salient_points["elastic_stability_limit_strain_measure_list"] ):
                
            if elastic_stability_limit_stress_measure is not None:
                plt.plot(elastic_stability_limit_strain_measure, elastic_stability_limit_stress_measure,  marker='o', markeredgecolor='gold', markerfacecolor='none', markersize=8)
                
    # Approximate elastic stability limit
    if dict_salient_points is not None:
        for (approx_elastic_stability_limit_stress_measure, approx_elastic_stability_limit_strain_measure, marker_face_color) in \
                zip( dict_salient_points["approx_elastic_stability_limit_stress_measure_list"], dict_salient_points["approx_elastic_stability_limit_strain_measure_list"], marker_face_color_list ):
                
            if approx_elastic_stability_limit_stress_measure is not None:
                plt.plot(approx_elastic_stability_limit_strain_measure, approx_elastic_stability_limit_stress_measure,  marker='o', markeredgecolor='olive', markerfacecolor=marker_face_color, markersize=8)                
                
    # Phonon stability limit
    if dict_salient_points is not None:
        for (phonon_stability_limit_stress_measure, phonon_stability_limit_strain_measure) in \
                zip( dict_salient_points["phonon_stability_limit_stress_measure_list"], dict_salient_points["phonon_stability_limit_strain_measure_list"] ):
                
            if phonon_stability_limit_stress_measure is not None:
                plt.plot(phonon_stability_limit_strain_measure, phonon_stability_limit_stress_measure,  marker='o', markeredgecolor='deeppink', markerfacecolor='none', markersize=8)                
    
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    legend = ax.legend(loc='best', fontsize=9, framealpha=0.6)
    legend_frame = legend.get_frame()
    legend_frame.set_edgecolor('black')
    legend_frame.set_linewidth(0.3)

    if legend_title is not None:
        legend.set_title(legend_title,prop={'size':'large'})

    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    return

'''-----------------------------------------------------------------------------
                                END OF MODULE
-----------------------------------------------------------------------------'''
