'''----------------------------------------------------------------------------
                         theoretical_strength_common_settings.py

 Description: Information about parameters for various elements to be used in
   theoretical strength calculations.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import sys
import math
import numpy as np

# Externally installed modules
from ase.build import bulk
from ase.lattice.cubic import Diamond
from pymatgen.core.lattice import Lattice
from ase.calculators.vasp import Vasp

# Local imports
# None

'''----------------------------------------------------------------------------
                                SUBROUTINES
----------------------------------------------------------------------------'''

def setup_vasp_calc(system=None, encut=None,  setups=None, 
                    kpoints=(15,15,15), kspacing=None, ismear=1, sigma=0.1,
                    isif=3, ibrion=2, ediff=1.0E-8, ediffg=-1.0E-4, isym=0,
                    nelm=100, nsw=60, potim=0.3, ispin=2,
                    is_md=False, tebeg=None, teend=None, smass=0, nblock=1, kblock=100):
                    
    # NOTE on EDIFFG:
    # Force convergence of 0.01 ev/A is very stringent. Such values can arise
    # from noise if you do not use extremely tight convergence parameters on
    # energy in SCF loop i.e (EDIFF=~10-8), a high cutoff, etc

    # ASE VASP calculator settings
    calc = Vasp(
        # Pseudo potential parameters
        xc='PBE',
        gga='PE',
        # Start parameters for this run
        prec='Accurate',   # High depends on ENAUG
        istart=0,          # Job   : 0-new  1-cont  2-samecut, Do not read WAVECAR as the system is under incremnetal loading
        icharg=1,          # Charge: 1-file 2-atom 10-const
        addgrid=True,
        lasph=True,        # Aspherical contributions
        ispin=ispin,
        lorbit=10,
        # Electronic relaxation parametrs
        nelm=nelm,         # Max SCF steps
        nelmin=5,          # Min SCF steps
        ediff=ediff,       # Stopping criterion for ELM
        lreal=False,
        voskown=1,
        algo='Normal',
        nelmdl=-5,
        # Ionic relaxation parameters
        nsw=nsw,            # Number of steps for ionic motion
        isif=isif,
        ediffg=ediffg,
        ibrion=ibrion,     # 0 - MD, 1 - RMM-DIIS, 2 - CG
        potim=potim,       # Time step for ionic motion
        isym=isym,
        # DOS related               
        ismear=ismear,          # 1st order Methfessel-Paxton, -5 tetrahedron 
        sigma=sigma,       # Broadening in eV
        # Write flags
        lwave=False,       # Write WAVECAR
        lcharg=True,       # Write CHGCAR
        lelf=False,        # Write electronic localization function (ELF)
        symprec=1.0E-8     
    )

    # Comment: Name of the calculation (Start parameter)
    if system is not None:
        calc.set(system=system)
        
    # If encut is not provided uses default encut in the pseudo potenial (Part of electronic relaxation parameter)
    if encut is not None:
        calc.set(encut=encut)   
        
    # If semi core potentials are needed (Part of electronic relaxation parameter)
    if setups is not None:
        calc.set(setups=setups)             

    # Kpoints related: kspacing is the preferred method

    if kspacing is None:
        calc.set(kpts=kpoints)
        calc.set(gamma=True)
    else:
        calc.set(kspacing=kspacing)
        calc.set(kgamma=True)

    if is_md:
    
        # Specific to molecular dynamics
        if tebeg is not None:
            calc.set(tebeg=tebeg)
        
        if teend is not None:
            calc.set(teend=teend)
            
        if ibrion != 0:
            calc.set(ibrion=0)
        
        calc.set(smass=smass)
        
        calc.set(nblock=nblock)
        calc.set(kblock=kblock)

    return calc

'''----------------------------------------------------------------------------
                                 MODULE VARIABLES
----------------------------------------------------------------------------'''

# Common settings
encut = 800.0 # eV
kspacing = 0.1 # 2.0 * math.pi * 0.015915494309189534 # per Angstrom
sigma=0.1
initial_magmom = 5.0

atom_info = {
    ###########
    ### FCC ###
    ###########
    'Al' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 4.039368108000, 'ispin' : 1},
    'Ni' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.519544860000, 'ispin' : 2},
    'Cu' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.630311514000, 'ispin' : 1},
    'Rh' : {'structure' : 'fcc', 'setups' : '_pv', 'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.830162490000, 'ispin' : 2},
    'Pd' : {'structure' : 'fcc', 'setups' : '_pv', 'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.942933846000, 'ispin' : 2},
    'Ag' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 4.146990364000, 'ispin' : 2},
    'Ir' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.872677134000, 'ispin' : 2},
    'Pt' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.967431832000, 'ispin' : 1},
    'Au' : {'structure' : 'fcc', 'setups' : None,  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 4.156145254000, 'ispin' : 2},
    'Pb' : {'structure' : 'fcc', 'setups' : '_d',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 5.039542326000, 'ispin' : 2},
    ###########
    ### BCC ###
    ###########
    'Li' : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.428332390000, 'ispin' : 2},
    'Na' : {'structure' : 'bcc', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 4.196858544000, 'ispin' : 2},
    'K'  : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 5.282269846000, 'ispin' : 2},
    'V'  : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.997823746000, 'ispin' : 2},
    'Cr' : {'structure' : 'bcc', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.846443340000, 'ispin' : 2},
    'Fe' : {'structure' : 'bcc', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.831399118000, 'ispin' : 2},
    'Rb' : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 5.668441620000, 'ispin' : 2},
    'Nb' : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.307527580000, 'ispin' : 2},
    'Mo' : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.162988074000, 'ispin' : 2},
    'Ba' : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 5.032541202000, 'ispin' : 2},
    'Ta' : {'structure' : 'bcc', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.319410100000, 'ispin' : 2},
    'W'  : {'structure' : 'bcc', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.185475034000, 'ispin' : 2},
    ###########
    ### HCP ###
    ###########
    'Be' : {'structure' : 'hcp', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.264661895000, 'c': 3.569770405000, 'ispin' : 2},
    'Mg' : {'structure' : 'hcp', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.188654941000, 'c': 5.211595300000, 'ispin' : 2},
    'Sc' : {'structure' : 'hcp', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.321353370000, 'c': 5.159590564000, 'ispin' : 2},
    'Ti' : {'structure' : 'hcp', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.936599357000, 'c': 4.646872381000, 'ispin' : 2},
    'Co' : {'structure' : 'hcp', 'setups' : None,   'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.490531332000, 'c': 4.024796987000, 'ispin' : 2},
    'Zn' : {'structure' : 'hcp', 'setups' : None,   'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.665057213000, 'c': 4.936818585000, 'ispin' : 2},
    'Y'  : {'structure' : 'hcp', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.655036207000, 'c': 5.665632718000, 'ispin' : 2},
    'Zr' : {'structure' : 'hcp', 'setups' : '_sv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.232709167000, 'c': 5.170213713000, 'ispin' : 2},
    'Tc' : {'structure' : 'hcp', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.750778576000, 'c': 4.398434003000, 'ispin' : 2},
    'Ru' : {'structure' : 'hcp', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.722063681000, 'c': 4.290611256000, 'ispin' : 2},
    'Cd' : {'structure' : 'hcp', 'setups' : None,   'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.025185151000, 'c': 5.803974053000, 'ispin' : 2},
    'Hf' : {'structure' : 'hcp', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 3.201647680000, 'c': 5.056931272000, 'ispin' : 2},
    'Re' : {'structure' : 'hcp', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.779812867000, 'c': 4.492939401000, 'ispin' : 2},
    'Os' : {'structure' : 'hcp', 'setups' : '_pv',  'encut' : encut, 'kspacing' : kspacing, 'sigma' : sigma, 'a': 2.759752015000, 'c': 4.354195210000, 'ispin' : 2},
    ##############################
    ### CUBIC DIAMOND SRUCTURE ###
    ##############################
    'Si' : {'structure' : 'cubic-diamond', 'setups' : None, 'encut' : 600, 'kspacing' : kspacing, 'sigma' : 0.05, 'a' : 5.468753059, 'ispin' : 1, 'is_conv_cell' : True, 'ismear' : 0},
}

'''----------------------------------------------------------------------------
                                 SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################

def get_lattice(symbol):

    lattice = None
    atom_dict = atom_info[symbol]
    structure = atom_dict['structure']
    if structure == 'fcc':
        lattice = Lattice.cubic(a=atom_dict['a'])
    elif structure == 'cubic-diamond':
        lattice = Lattice.cubic(a=atom_dict['a'])
    elif structure == 'bcc':
        lattice = Lattice.cubic(a=atom_dict['a'])    
    elif structure == 'hcp':
        lattice = Lattice.hexagonal(a=atom_dict['a'], c=atom_dict['c'])
    return lattice

##################
### SUBROUTINE ###
##################

def get_structure(symbol):

    atom_dict = atom_info[symbol]
    return atom_dict['structure']
    
##################
### SUBROUTINE ###
##################

def get_pp_setups(symbol):

    atom_dict = atom_info[symbol]
    return {symbol : atom_dict['setups'] }
    
##################
### SUBROUTINE ###
##################


def get_natural_structures(symbol_list):

    #------------------------#
    # Create these variables #
    #------------------------#
    atoms_list = []
    calc_list = []
    label_list = []
    #------------------------#

    # Iterate over each element
    for symbol in symbol_list:

        # Read element dictionary
        atom_dict = atom_info[symbol]

        # Irrestpective of ferromagnetic or not set initial magnetic moment = 5
        magmom = initial_magmom

        # Pseudo potential additional setups
        setups = { symbol : atom_dict['setups'] }

        # Create atoms in natural structure
        structure = atom_dict['structure']

        if structure == "fcc":

            a = atom_dict["a"]
            atoms = bulk(symbol, crystalstructure=structure, a=a)
            atoms.set_initial_magnetic_moments(np.array([magmom]))

        elif structure == "cubic-diamond":

            a = atom_dict["a"]
            if atom_dict['is_conv_cell']:
                atoms = Diamond(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(1,1,1), symbol=symbol, latticeconstant=a)
                atoms.set_initial_magnetic_moments(np.ones(8)*magmom)
            else:
                atoms = bulk(symbol, crystalstructure=structure, a=a)
                atoms.set_initial_magnetic_moments(np.array([magmom, magmom]))

        elif structure == "bcc":

            a = atom_dict["a"]
            atoms = bulk(symbol, crystalstructure=structure, a=a)
            atoms.set_initial_magnetic_moments(np.array([magmom]))

        elif structure == "hcp":

            a = atom_dict["a"]
            c = atom_dict["c"]
            atoms = bulk(symbol, crystalstructure=structure, a=a, c=c)
            atoms.set_initial_magnetic_moments(np.array([magmom, magmom]))

        else:
            sys.stderr.write("Error: In module '%s'\n" %(module_name))
            sys.stderr.write("       In subroutine 'main'\n")
            sys.stderr.write("       Unknown structure, Only 'fcc', 'bcc' or 'hcp' allowed.\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # Setup calculator
        calc = setup_vasp_calc( \
            system=symbol +'_relax_isv',
            encut=atom_dict['encut'],
            setups=setups, 
            kspacing=atom_dict['kspacing'],
            sigma=atom_dict['sigma'],
            isif=3, ibrion=2, ediff=1.0E-8, ediffg=-1.0E-4, isym=2, ispin=atom_dict['ispin'])
            
        if 'ismear' in atom_dict:
            calc.set(ismear=atom_dict['ismear'])

        # Push atoms and calc to arrays
        atoms_list.append(atoms)
        calc_list.append(calc)
        label_list.append(symbol)

    return (atoms_list, calc_list, label_list)

'''----------------------------------------------------------------------------
                                  END OF SETUP
----------------------------------------------------------------------------'''
