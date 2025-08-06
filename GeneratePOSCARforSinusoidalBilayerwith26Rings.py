# Last modified by Xingfu Li @Rice, Aug.2025
# needs flat bilayer POSCAR in Cartesian coordinates with 26 Rings
# then calculate shear between two layers

from sympy import *
import math as m
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

L0 = 65.3223953247  # lattice constant of modulation direction
A = 3  # amplitude of sine
d = 3.253 / 2  # half of interlayer distance

# ***calculate new lattice constant of modulation direction***#
x = symbols("x")
l_upper = 66
l_lower = 50
eps1 = 1
while eps1 > 0.00001:
    l = (l_upper + l_lower) / 2
    z1 = A * sin(x * 2 * m.pi / l)
    dz1 = diff(z1, x)
    f1 = sqrt(1 + dz1**2)
    f1_n = lambdify(x, f1)
    I1 = quad(f1_n, 0, l)
    eps1 = abs(I1[0] - L0)
    if I1[0] > L0:
        l_upper = l
    else:
        l_lower = l
print(l)

# ***generate coordinates***#
# get x and z coordinates for bot layer
z2 = 12 + A * sin(x * 2 * m.pi / l)
dz2 = diff(z2, x)
g = sqrt(1 + dz2**2)
xd_b = x + d * dz2 / g
zd_b = z2 - d / g
dxd_b = diff(xd_b, x)
dzd_b = diff(zd_b, x)
f2_b = sqrt(dxd_b**2 + dzd_b**2)
f2_b_n = lambdify(x, f2_b)
x_t = np.zeros(53)
for i in range(52):
    t_upper_max = 68
    t_upper_min = x_t[i]
    t_lower = x_t[i]
    eps = 1
    while eps > 0.00001:
        t_upper = (t_upper_max + t_upper_min) / 2
        I2 = quad(f2_b_n, t_lower, t_upper)
        eps = abs(I2[0] - 1.256199910090385)
        if I2[0] > 1.256199910090385:
            t_upper_max = t_upper
        else:
            t_upper_min = t_upper
    x_t[i + 1] = t_upper

xd_b_n = lambdify(x, xd_b)
xd_b_n_v = xd_b_n(x_t)
zd_b_n = lambdify(x, zd_b)
zd_b_n_v = zd_b_n(x_t)

# get x and z coordinates for top layer
xd_t = x - d * dz2 / g
zd_t = z2 + d / g
dxd_t = diff(xd_t, x)
dzd_t = diff(zd_t, x)
f2_t = sqrt(dxd_t**2 + dzd_t**2)
f2_t_n = lambdify(x, f2_t)
x_t = np.zeros(53)
for i in range(52):
    t_upper_max = 68
    t_upper_min = x_t[i]
    t_lower = x_t[i]
    eps = 1
    while eps > 0.00001:
        t_upper = (t_upper_max + t_upper_min) / 2
        I2 = quad(f2_t_n, t_lower, t_upper)
        eps = abs(I2[0] - 1.256199910090385)
        if I2[0] > 1.256199910090385:
            t_upper_max = t_upper
        else:
            t_upper_min = t_upper
    x_t[i + 1] = t_upper

xd_t_n = lambdify(x, xd_t)
xd_t_n_v = xd_t_n(x_t)
zd_t_n = lambdify(x, zd_t)
zd_t_n_v = zd_t_n(x_t)

# ***generate POSCAR***#
f = open("bilayer flat supercell.vasp", "r")
line = f.readlines()
line[2] = line[2].replace(line[2].split()[0], str(l))

# position of top layer
for i in range(26):
    line[8 + i] = line[8 + i].replace(line[8 + i].split()[0], str(xd_t_n_v[1 + 2 * i]))
    line[8 + i] = line[8 + i].replace(line[8 + i].split()[2], str(zd_t_n_v[1 + 2 * i]))
    line[190 + i] = line[190 + i].replace(
        line[190 + i].split()[0], str(xd_t_n_v[1 + 2 * i])
    )
    line[190 + i] = line[190 + i].replace(
        line[190 + i].split()[2], str(zd_t_n_v[1 + 2 * i])
    )
for i in range(26):
    line[34 + i] = line[34 + i].replace(line[34 + i].split()[0], str(xd_t_n_v[2 * i]))
    line[34 + i] = line[34 + i].replace(line[34 + i].split()[2], str(zd_t_n_v[2 * i]))
    line[164 + i] = line[164 + i].replace(
        line[164 + i].split()[0], str(xd_t_n_v[2 * i])
    )
    line[164 + i] = line[164 + i].replace(
        line[164 + i].split()[2], str(zd_t_n_v[2 * i])
    )
# position of bot layer
for i in range(26):
    line[86 + i] = line[86 + i].replace(
        line[86 + i].split()[0], str(xd_b_n_v[1 + 2 * i])
    )
    line[86 + i] = line[86 + i].replace(
        line[86 + i].split()[2], str(zd_b_n_v[1 + 2 * i])
    )
    line[112 + i] = line[112 + i].replace(
        line[112 + i].split()[0], str(xd_b_n_v[1 + 2 * i])
    )
    line[112 + i] = line[112 + i].replace(
        line[112 + i].split()[2], str(zd_b_n_v[1 + 2 * i])
    )
for i in range(26):
    line[60 + i] = line[60 + i].replace(line[60 + i].split()[0], str(xd_b_n_v[2 * i]))
    line[60 + i] = line[60 + i].replace(line[60 + i].split()[2], str(zd_b_n_v[2 * i]))
    line[138 + i] = line[138 + i].replace(
        line[138 + i].split()[0], str(xd_b_n_v[2 * i])
    )
    line[138 + i] = line[138 + i].replace(
        line[138 + i].split()[2], str(zd_b_n_v[2 * i])
    )

nline = line
f = open("test.vasp", "w")
f.writelines(nline)
f.close()
