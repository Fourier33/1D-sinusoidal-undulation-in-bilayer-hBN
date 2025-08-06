import numpy as np
from matplotlib import pyplot as plt
from ase.io import read
import os

# from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline

# from scipy.integrate import quad
# from scipy.signal import savgol_filter
from sympy import *
from tqdm import tqdm

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
    slope_central_line=-1, passpoint=[0, 1], line_length=10, grid=100
):
    slope_perpen_line = -1 / slope_central_line
    line_proj_z_length_half = (line_length / 2) * np.sin(
        np.abs(np.arctan(slope_perpen_line))
    )
    line_z_list = np.linspace(
        passpoint[1] - line_proj_z_length_half,
        passpoint[1] + line_proj_z_length_half,
        grid,
    )
    line_x_list = (line_z_list - passpoint[1]) / slope_perpen_line + passpoint[0]
    return line_x_list, line_z_list


def Regroup(
    line_x_list=np.zeros(10),
    cell_dimension_a=60,
    line_z_list=np.zeros(10),
    cell_dimension_b=60,
):
    line_x_list_regourp = np.zeros(len(line_x_list))
    for i_elemt in range(len(line_x_list)):
        if line_x_list[i_elemt] < 0:
            line_x_list_regourp[i_elemt] = line_x_list[i_elemt] + cell_dimension_a
        elif line_x_list[i_elemt] > cell_dimension_a:
            line_x_list_regourp[i_elemt] = line_x_list[i_elemt] - cell_dimension_a
        else:
            line_x_list_regourp[i_elemt] = line_x_list[i_elemt]

    line_z_list_regourp = np.zeros(len(line_z_list))
    for i_elemt in range(len(line_z_list)):
        if line_z_list[i_elemt] < 0:
            line_z_list_regourp[i_elemt] = line_z_list[i_elemt] + cell_dimension_b
        elif line_z_list[i_elemt] > cell_dimension_b:
            line_z_list_regourp[i_elemt] = line_z_list[i_elemt] - cell_dimension_b
        else:
            line_z_list_regourp[i_elemt] = line_z_list[i_elemt]
    return line_x_list_regourp, line_z_list_regourp


cell_dimen_a, cell_dimen_b, cell_dimen_c = read_cell_info("POSCAR_bot")
data_3D_bi, grid = read_volumetric_data("CHGCAR_bi")
data_3D_bot, grid = read_volumetric_data("CHGCAR_bot")
data_3D_top, grid = read_volumetric_data("CHGCAR_top")
data_3D_diff = (data_3D_bi - data_3D_bot - data_3D_top) / (
    cell_dimen_a * cell_dimen_b * cell_dimen_c
)

x_mesh, z_mesh = np.meshgrid(
    np.linspace(0, cell_dimen_a, int(grid[0])),
    np.linspace(0, cell_dimen_c, int(grid[2])),
)
data_2D_xz = data_3D_diff.mean(1)
data_2D_xz_spline = RectBivariateSpline(
    np.linspace(0, cell_dimen_c, int(grid[2])),
    np.linspace(0, cell_dimen_a, int(grid[0])),
    data_2D_xz,
)


grid_tangential = 60
window_length = 14
grid_normal = int(window_length / 0.05)

with open("central line x position for central line", "r") as file:
    lines_tmp = file.readlines()
central_line_x_position_top_layer = []
for j_line in range(len(lines_tmp)):
    central_line_x_position_top_layer.append(float(lines_tmp[j_line].split()[0]))
central_line_x_position_top_layer = np.array(central_line_x_position_top_layer)
central_line_x_position_top_layer[-1] = cell_dimen_c

with open("net charge unit cell 14", "w") as f:
    for i_unit_cell in tqdm(range(26)):
        central_sin_x_list = np.linspace(
            central_line_x_position_top_layer[2 * i_unit_cell],
            central_line_x_position_top_layer[2 * i_unit_cell + 2],
            grid_tangential,
        )
        # central_sin_x_list = np.array([0.75*cell_dimen_a])
        window_ave_chg_density_sum = np.zeros(grid_normal)
        for x_val in central_sin_x_list:
            posi_x, central_sin_val, slope_central_sin_val = GetCentralSinLocalSlope(
                3, cell_dimen_a, posi_x=x_val
            )
            line_x_list, line_z_list = GetLocalPerpendicularLine(
                slope_central_sin_val,
                passpoint=[posi_x, central_sin_val],
                line_length=window_length,
                grid=grid_normal,
            )
            line_x_list_regroup, line_z_list_regroup = Regroup(
                line_x_list, cell_dimen_a, line_z_list, cell_dimen_c
            )
            ave_chg_density_along_line = data_2D_xz_spline.ev(
                line_z_list_regroup, line_x_list_regroup
            )
            window_ave_chg_density_sum = (
                window_ave_chg_density_sum + ave_chg_density_along_line
            )
        window_ave_charge = (window_ave_chg_density_sum / grid_tangential).mean()

        # fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
        # axs[0].set_aspect('equal')
        # axs[0].contourf(x_mesh, z_mesh, data_2D_xz, levels=50)
        # axs[0].scatter(line_x_list_regroup, line_z_list_regroup, color='red', s=0.5)

        # # fig, ax = plt.subplots()
        # axs[1].plot(np.linspace(0, window_length, grid_normal), ave_chg_density_along_line, label='griddata')
        # axs[1].legend()
        # axs[1].set_xlim(0, window_length)
        # axs[1].set_ylabel(r'charge density difference (e/Å$^3$)')
        # axs[1].set_xlabel('line direction (Å)')
        # plt.tight_layout()
        # plt.show()

        net_charge = window_ave_charge * (window_length * cell_dimen_b * 2.5124)
        f.write(f"{net_charge}\n")


# fig, ax = plt.subplots()
# ax.plot(np.linspace(0, 10, 100), window_ave_charge)
# ax.axhline(y=net_charge, color='tab:red', ls='--')
# ax.axhline(y=0, color='gray', ls='--')
# plt.tight_layout()
# plt.show()
