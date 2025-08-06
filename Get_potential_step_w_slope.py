import numpy as np
from matplotlib import pyplot as plt
from ase.io import read
import os
from scipy.interpolate import griddata

# from scipy.integrate import quad
# from scipy.signal import savgol_filter
from sympy import *

# from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def read_volumetric_data(filename="filename"):
    with open(filename, "r") as f:
        lines = f.readlines()
    for i_line, line in enumerate(lines):
        if "Direct" in line:
            line_n_atom_per_type = i_line - 1
    n_atom_per_type = [int(elem) for elem in lines[line_n_atom_per_type].split()]
    n_atom = sum(n_atom_per_type)
    line_grid = line_n_atom_per_type + 3 + n_atom
    grid = [int(elem) for elem in lines[line_grid].split()]
    n_vol_data = grid[0] * grid[1] * grid[2]
    line_vol_data_begin = line_grid + 1
    vol_data = []
    if n_vol_data % 5 == 0:
        for i_line_vol_data in range(int(n_vol_data / 5)):
            line = lines[line_vol_data_begin + i_line_vol_data]
            for i_data_per_line in range(5):
                vol_data.append(float(line.split()[i_data_per_line]))
    else:
        for i_line_vol_data in range(int(n_vol_data / 5) + 1):
            line = lines[line_vol_data_begin + i_line_vol_data]
            if i_line_vol_data == int(n_vol_data / 5):
                n_data_this_line = len(line.split())
                for i_data_per_line in range(n_data_this_line):
                    vol_data.append(float(line.split()[i_data_per_line]))
            else:
                for i_data_per_line in range(5):
                    vol_data.append(float(line.split()[i_data_per_line]))

    data_3D = np.zeros((grid[2], grid[1], grid[0]))
    for i_z in range(grid[2]):
        for i_y in range(grid[1]):
            for i_x in range(grid[0]):
                data_3D[i_z, i_y, i_x] = vol_data[
                    i_z * grid[1] * grid[0] + i_y * grid[0] + i_x
                ]
    return data_3D, grid


def read_cell_info(geomfile="POSCAR", outputfile=None):
    geom = read(geomfile)
    a, b, c, ang_bc, ang_ac, ang_ab = geom.cell.cellpar()
    if outputfile is not None:
        with open(outputfile, "r") as f:
            lines = f.readlines()
        for i_line, line in enumerate(lines):
            if "VOLUME and BASIS-vectors are now" in line:
                line_vol = i_line + 3
        vol = float(lines[line_vol].split()[-1])
        return a, b, c, vol
    else:
        return a, b, c


def GetCentralSinLocalSlope(amplitude=3, cell_dimension_a=60, posi_x=0):
    x = symbols("x")
    central_sin = 12 + amplitude * sin(
        x * 2 * np.pi / cell_dimension_a
    )  # 12 is the z position of central sine line, be careful
    diff_central_sin = diff(central_sin, x)
    slope_central_sin_lam = lambdify(x, diff_central_sin)
    slope_central_sin_val = slope_central_sin_lam(posi_x)

    central_sin_lam = lambdify(x, central_sin)
    central_sin_val = central_sin_lam(posi_x)
    return posi_x, central_sin_val, slope_central_sin_val


def GetLocalPerpendicularLine(
    slope_central_line=-1, passpoint=[0, 1], line_z_list=np.zeros(10)
):
    slope_perpen_line = -1 / slope_central_line
    line_x_list = (line_z_list - passpoint[1]) / slope_perpen_line + passpoint[0]
    return line_z_list, line_x_list


def Regroup(line_x_list=np.zeros(10), cell_dimension_a=60):
    line_x_list_regourp = np.zeros(len(line_x_list))
    for i_elemt in range(len(line_x_list)):
        if line_x_list[i_elemt] < 0:
            line_x_list_regourp[i_elemt] = line_x_list[i_elemt] + cell_dimension_a
        elif line_x_list[i_elemt] > cell_dimension_a:
            line_x_list_regourp[i_elemt] = line_x_list[i_elemt] - cell_dimension_a
        else:
            line_x_list_regourp[i_elemt] = line_x_list[i_elemt]
    return line_x_list_regourp


def GetLineLength(cell_dimension_c=20, slope_central_sin_val=1, grid=20):
    line_length = cell_dimension_c / np.cos(np.abs(np.arctan(slope_central_sin_val)))
    line_length_list = np.linspace(0, line_length, grid)
    return line_length_list


def GetPotStep(line_length_list=np.zeros(5), pot_along_line=np.zeros(5)):
    array_length = len(line_length_list)
    interval = int(
        0.1 * array_length
    )  # This number 0.x control the distance between two sampling points, be careful
    line1_x1 = line_length_list[0]
    line1_y1 = pot_along_line[0]
    line1_x2 = line_length_list[0 + interval]
    line1_y2 = pot_along_line[0 + interval]

    line2_x1 = line_length_list[-1]
    line2_y1 = pot_along_line[-1]
    line2_x2 = line_length_list[-1 - interval]
    line2_y2 = pot_along_line[-1 - interval]

    slope_line1 = (line1_y2 - line1_y1) / (line1_x2 - line1_x1)
    slope_line2 = (line2_y2 - line2_y1) / (line2_x2 - line2_x1)
    slope_line = (slope_line1 + slope_line2) / 2

    intercept1 = line1_y1 - slope_line * line1_x1
    intercept2 = line2_y1 - slope_line * line2_x1

    distance_para_line = intercept2 - intercept1 / np.sqrt(1 + slope_line**2)
    return (
        distance_para_line,
        slope_line,
        intercept1,
        intercept2,
        line1_x1,
        line1_y1,
        line1_x2,
        line1_y2,
        line2_x1,
        line2_y1,
        line2_x2,
        line2_y2,
        slope_line1,
        slope_line2,
    )


cell_dimen_a, cell_dimen_b, cell_dimen_c = read_cell_info("POSCAR_AAp_ZZ_sin3")
data_3D_bilayer, grid = read_volumetric_data("LOCPOT_AAp_ZZ_sin3")

x_mesh, z_mesh = np.meshgrid(
    np.linspace(0, cell_dimen_a, int(grid[0])),
    np.linspace(0, cell_dimen_c, int(grid[2])),
)
data_2D_xz_bilayer = data_3D_bilayer.mean(1)

# central_sin_x_list = np.linspace(0, cell_dimen_a, 50)
central_sin_x_list = np.array([0.2 * cell_dimen_a])
pot_step_list = []
for x_val in central_sin_x_list:
    posi_x, central_sin_val, slope_central_sin_val = GetCentralSinLocalSlope(
        3, cell_dimen_a, posi_x=x_val
    )
    line_z_list, line_x_list = GetLocalPerpendicularLine(
        slope_central_sin_val,
        passpoint=[posi_x, central_sin_val],
        line_z_list=np.linspace(0, cell_dimen_c, int(grid[2])),
    )
    line_x_list_regroup = Regroup(line_x_list, cell_dimen_a)
    pot_along_line = griddata(
        (x_mesh.flatten(), z_mesh.flatten()),
        data_2D_xz_bilayer.flatten(),
        (line_x_list_regroup, line_z_list),
        method="cubic",
    )
    line_length_list = GetLineLength(
        cell_dimen_c, slope_central_sin_val, grid=int(grid[2])
    )
    (
        pot_step,
        slope_line,
        intercept1,
        intercept2,
        line1_x1,
        line1_y1,
        line1_x2,
        line1_y2,
        line2_x1,
        line2_y1,
        line2_x2,
        line2_y2,
        slope_line1,
        slope_line2,
    ) = GetPotStep(line_length_list, pot_along_line)
    pot_step_list.append(pot_step)

# fig, ax = plt.subplots()
# ax.plot(central_sin_x_list, pot_step_list, marker='o', fillstyle='none')
# plt.tight_layout()
# plt.show()

# with open('../toplayer flexo pot step', 'w') as f:
#     for x_i, pot_step_i in zip(central_sin_x_list, pot_step_list):
#         f.write(f'{x_i}   {pot_step_i}\n')

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.contourf(x_mesh, z_mesh, data_2D_xz_bilayer, levels=50)
# ax.plot(line_x_list_regroup, line_z_list, color='tab:red', linewidth=1)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.savefig('figs8_panel3.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
# plt.tight_layout()
# plt.show()

# # print(pot_step, slope_line1, slope_line2)
fig, ax = plt.subplots()
ax.plot(line_length_list, pot_along_line)
# ax.scatter(line1_x1, line1_y1, color='red', marker='x')
# ax.scatter(line1_x2, line1_y2, color='red', marker='x')
# ax.scatter(line2_x1, line2_y1, color='red', marker='x')
# ax.scatter(line2_x2, line2_y2, color='red', marker='x')
ax.plot(line_length_list, slope_line * line_length_list + intercept1)
ax.plot(line_length_list, slope_line * line_length_list + intercept2)
ax.set_ylim(max(pot_along_line) - 0.5, max(pot_along_line) + 0.1)
ax.set_xlim(0, line_length_list[-1])
ax.set_ylabel("average potential", fontsize=14)
ax.set_xlabel("line direction (Å)", fontsize=14)
ax.tick_params(axis="both", direction="in", labelsize=14)
plt.tight_layout()
plt.show()
