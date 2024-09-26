'''-----------------------------------------------------------------------------
                               setup.py

 Description: Setup for theoretical strength calculation for fcc Aluminium.
 
   This setup file can be adapted to to perform ideal strength calculations on
   bcc and hcp strcutures.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import numpy as np

# Externally installed modules
# None

# Local imports
from matkit.core import mod_incr_load
from scripts.theoretical_strength import theoretical_strength_common_settings as ts_set

'''-----------------------------------------------------------------------------
                                 SUBROUTINES
-----------------------------------------------------------------------------'''
   
##################
### SUBROUTINE ###
##################


def get_stress_strain_info_dict_fcc(
        inc_def_par_list,
        stress_voigt_list_list_list=None,
        do_hard=True, do_sym=True, do_non_critical=True,
        is_hydrostatic=True, is_normal=True, is_shear=True):
    
    # If no superimposed stresses are provided
    if stress_voigt_list_list_list is None:
        stress_voigt_list_list_list = [None] # None for hydrostatic loading
        zero_stress_voigt_list_list = mod_incr_load.get_biaxial_stress_voigt_list(stress_1_list=np.linspace(0.0, 0.0, 1), idx_1=0, idx_2=1, load_ratio=1.0)
        for i in range(1, 8):
            stress_voigt_list_list_list.append(zero_stress_voigt_list_list)

    # Create an empty dictionary
    stress_strain_info_dict = {}
    
    # Hydrostatic
    if is_hydrostatic:
        stress_strain_info_dict['fcc_hydrostatic_tension'] = { 
            'incremental_deformation_parameter': inc_def_par_list[0], 
            'stress_voigt_list_list' : None}

    # Normal stress
    if is_normal:
        stress_strain_info_dict['fcc001x100_tension_z'] = {
            'incremental_deformation_parameter': inc_def_par_list[1], 
            'stress_voigt_list_list' : stress_voigt_list_list_list[1]
        }
    
        stress_strain_info_dict['fcc110x-110_tension_z'] = {
            'incremental_deformation_parameter': inc_def_par_list[2],
            'stress_voigt_list_list' : stress_voigt_list_list_list[2]
        }
    
        stress_strain_info_dict['fcc111x11-2_tension_z'] = {
            'incremental_deformation_parameter': inc_def_par_list[3],
            'stress_voigt_list_list' : stress_voigt_list_list_list[3]
        }
        
    # Shear stress
    if is_shear:
    
        # Non Critical shear
        # Usually for FCC lattices shear in [-110] direction on (111) plane is
        # not critical as it is higher compared to shear along [11-2] direction
        # on (111) plane. So this case may be left out
        if do_non_critical:
            # Shear symmetric (skip second one if symmetric exploration is not required)
            stress_strain_info_dict['fcc111x-110_shear_zx'] = {
                'incremental_deformation_parameter': inc_def_par_list[4],
                'stress_voigt_list_list' : stress_voigt_list_list_list[4]
            }
            if do_sym:
                stress_strain_info_dict['fcc111x1-10_shear_zx'] = {
                    'incremental_deformation_parameter': inc_def_par_list[5],
                    'stress_voigt_list_list' : stress_voigt_list_list_list[5]
                }

        # Critical Shear
        # Easy direction
        stress_strain_info_dict['fcc111x11-2_shear_zx'] = {
            'incremental_deformation_parameter': inc_def_par_list[6],
            'stress_voigt_list_list' : stress_voigt_list_list_list[6]
        }
        # Hard direction
        if do_hard:
            stress_strain_info_dict['fcc111x-1-12_shear_zx'] = {
                'incremental_deformation_parameter': inc_def_par_list[7],
                'stress_voigt_list_list' : stress_voigt_list_list_list[7]
            }
    
    return stress_strain_info_dict

'''----------------------------------------------------------------------------
                              MODULE VARIABLES
----------------------------------------------------------------------------'''

def get_element_incr_load_info_list(element):

    if element == 'Al':
    
        # Incremental deformation (like strain) along the loading paths.
        # NOTE: FCC has a total of 8 loading path types.
        inc_def_par_list = np.array([1, 1, 1, 1, 1, 1, 1, 1]) * 0.01
        
        # DFT parameters specific to Aluminium. Other default paramters are specified in ts_set
        ediff = 1.0E-8
        ediffg = -1.0E-4
        kspacing = 0.06
        encut=520
        sigma=1.0
        pp_setups = ts_set.get_pp_setups(element)
        ispin = ts_set.atom_info[element]['ispin']
        isif = 2
        
        # Setup common calculator for constrained cell relaxation
        calc_load = ts_set.setup_vasp_calc(
            system='ideal_strength', encut=encut, setups=pp_setups, ispin=ispin,
            kspacing=kspacing, sigma=sigma, isif=isif, ediff=ediff, ediffg=ediffg, isym=0)

        # Change this case to perdorm various kinds (superimposed) stress based calculations of ideal strength
        case = 'zero_stress'

        if case == 'zero_stress':
            stress_strain_info_dict = get_stress_strain_info_dict_fcc(inc_def_par_list=inc_def_par_list)
        
        if case == 'biaxial_superposed_on_normal':
            # FCC has 8 cases, even if not used, just populate them
            stress_voigt_list_list_list = [None] * 8
            
            # Get a list of balanced biaxial normal streses ranging in x and y directions
            # fcc001x100_tension_z
            stress_voigt_list_list_list[1] = mod_incr_load.get_biaxial_stress_voigt_list(stress_1_list=np.linspace(-8, 10, 19), idx_1=0, idx_2=1, load_ratio=1.0)
            # fcc110x-110_tension_z
            stress_voigt_list_list_list[2] = mod_incr_load.get_biaxial_stress_voigt_list(stress_1_list=np.linspace(-6, 10, 17), idx_1=0, idx_2=1, load_ratio=1.0)
            # fcc111x11-2_tension_z
            stress_voigt_list_list_list[3] = mod_incr_load.get_biaxial_stress_voigt_list(stress_1_list=np.linspace(-8, 10, 19), idx_1=0, idx_2=1, load_ratio=1.0)
            
            # get dict
            stress_strain_info_dict = get_stress_strain_info_dict_fcc(
                inc_def_par_list=inc_def_par_list,
                stress_voigt_list_list_list=stress_voigt_list_list_list,
                do_hard=True, do_sym=False, do_non_critical=True,
                is_hydrostatic=False, is_normal=True, is_shear=False)

        if case == 'triaxial_superposed_on_shear':
            stress_voigt_list_list_list = [None] # None for hydrostatic loading
            triaxial_stress_voigt_list_list = mod_incr_load.get_triaxial_stress_voigt_list(
                stress_3_list=np.linspace(-10,5,16),
                idx_3=2,
                lambda_list=[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
                #lambda_list = [0.0])
            # FCC has 8 cases, even if not used, just populate them
            for i in range(1, 8):
                stress_voigt_list_list_list.append(triaxial_stress_voigt_list_list)
            
            # get dict
            stress_strain_info_dict = get_stress_strain_info_dict_fcc(
                inc_def_par_list=inc_def_par_list,
                stress_voigt_list_list_list=stress_voigt_list_list_list,
                do_hard=False, do_sym=False, do_non_critical=False,
                is_hydrostatic=False, is_normal=False, is_shear=True)
                
       
        if case == 'triaxial_superposed_on_shear_1':
            stress_voigt_list_list_list = [None] # None for hydrostatic loading
            triaxial_stress_voigt_list_list = mod_incr_load.get_triaxial_stress_voigt_list(
                stress_3_list=np.concatenate( (np.linspace(-3,0,7), np.linspace(1,5,5) )),
                idx_3=2,
                lambda_list=[-0.75])
            # FCC has 8 cases, even if not used, just populate them
            for i in range(1, 8):
                stress_voigt_list_list_list.append(triaxial_stress_voigt_list_list)
            
            # get dict
            stress_strain_info_dict = get_stress_strain_info_dict_fcc(
                inc_def_par_list=inc_def_par_list,
                stress_voigt_list_list_list=stress_voigt_list_list_list,
                do_hard=False, do_sym=False, do_non_critical=False,
                is_hydrostatic=False, is_normal=False, is_shear=True)
                
        dict_load_cont_info_list = mod_incr_load.setup_incr_load_info_list(
            species_list = [element],
            initial_magmom_per_site = 5.0,
            lattice_type = ts_set.get_structure(element),
            lattice = ts_set.get_lattice(element),
            calc_load = calc_load,
            stress_strain_info_dict = stress_strain_info_dict,
            phonon_supercell_size_list = [],
            is_disturb=False, disturb_vec = np.zeros(3), disturb_seed=None,
            stress_voigt_convergence_tol_GPa = np.array([1.0E-2, 1.0E-2, 1.0E-2, 1.0E-1, 1.0E-1, 1.0E-1]))
                                                                                  
    # Return the load continuation dictionary list of the material with various loading cases
    return dict_load_cont_info_list

'''----------------------------------------------------------------------------
                                  END OF SETUP
----------------------------------------------------------------------------'''
