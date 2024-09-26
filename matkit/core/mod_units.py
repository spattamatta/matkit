'''----------------------------------------------------------------------------
                             mod_units.py

 Description: Physical constants, conversions etc.
              All values taken from NIST reference.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
# None

# Externally installed modules
# None

# Local imports
# None

'''----------------------------------------------------------------------------
                              MODULE VARIABLES
----------------------------------------------------------------------------'''
kB_to_J_per_K = 1.380649E-23  #[J/K]
eV_to_J = 1.602176634E-19 #[J]
kB_eV_per_K = kB_to_J_per_K / eV_to_J
Angstrom_to_Meter = 1.0E-10   #[m]
eV_per_cubic_Angstrom_to_Pa = eV_to_J/(Angstrom_to_Meter**3.0)
eV_per_cubic_Angstrom_to_J_per_cubic_meter = eV_to_J/(Angstrom_to_Meter**3.0)

J_per_cubic_meter_to_eV_per_cubic_Angstrom = 1.0/eV_per_cubic_Angstrom_to_J_per_cubic_meter

Avogadro = 6.02214076E23    #[mol-1]

eV_to_kJ_per_mol = (eV_to_J/1000.0)*Avogadro # [kJ/mol] 96.4853910

me_rest_kg = 9.1093837015E-31   # Mass of electron in kg
e_charge_C = 1.602176634E-19    # Elementary charge in coulomb
eps0_F_per_m = 8.8541878128E-12 # Electric permittivity of free space F m-1
h_Js = 6.62607015E-34           # Plank's constant in Joule second
c_m_per_s = 299792458           # Speed of light in vacuum m s-1

Ry_to_J = me_rest_kg*(e_charge_C**4)/(8.0*(eps0_F_per_m**2)*(h_Js**2))
Ry_to_eV = Ry_to_J /eV_to_J
eV_to_Ry = 1.0 / Ry_to_eV
universal_gas_constant_J_per_K_per_mol = 8.314462618 # J mol-1 K-1

bohr_to_meter = 5.29177210903E-11
bohr_to_angstrom = bohr_to_meter / Angstrom_to_Meter
Ry_per_bhor_to_eV_per_Angstrom = Ry_to_eV / bohr_to_angstrom
eV_per_Angstrom_to_Ry_per_bhor = 1.0 / Ry_per_bhor_to_eV_per_Angstrom

giga = 1.0E9
mega = 1.0E6


'''----------------------------------------------------------------------------
                               END OF MODULE
----------------------------------------------------------------------------'''
