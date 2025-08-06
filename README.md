# 1D-sinusoidal-undulation-in-bilayer-hBN
This repository contains Python scripts used in paper https://doi.org/10.1103/bxw4-7y5c. 

# Generate periodic 1D sinusoidal bilayer hBN structure
The *GeneratePOSCARforSinusoidalBilayerwith26Rings.py* is very limited. It can only generate the structures of four cases as discussed in the paper: AA/AA' stacking bilayer undulated along zigzag/armchair direction. It requires a supercell structure file in VASP format of flat bilayer hBN as the starting point, which must have the same shape as the final undulated structure. In the last step of writing the structure file, one-to-one correspondence of atom index is implemented. The core idea of algorithm is parallel curves, which is introduced in Supplemental Material SM-1. Here, I present an example of AA stacking bilayer undulated along zigzag direction with amplitude of sine as 0.3 nm. 

# Macroscopic averaging, tailored for curved, layered atomic structures
The *Average_unit_cell_charge.py* was used to generate the contents in Fig. 4bc in the paper. It reads charge density files from separate calculations for bilayer and two individual layers to get so-called differential charge density. The average is done in a block unit where the fineness is controlled by "grid_tangential" and "window_length". The number of data you will get is the same as the rectangular unit cell number in undulation direction when you design the structure. The core idea of algorithm is introduced in the Methods section of the paper. 

# Extracting local potential steps for curved bilayer
The *Get_potential_step_w_slope.py* was used to generate the out-of-plane potential step in Fig. 2a. It reads the so-called LOCPOT output file from VASP. The core idea of algorithm is introduced in Supplemental Material SM-2.
