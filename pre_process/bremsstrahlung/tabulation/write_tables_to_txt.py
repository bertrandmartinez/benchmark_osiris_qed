import numpy as np
import qed_rates
# This file creates and maintains 4 tables for bremsstrahlung emission.
# For it to work in your computer you just need to change the path to where you want the text files to be saved

# The first table is for the values of g1 - 1:
# IMPORTANT: this is already - 1 because it starts at 0


def make_Z_table_ge_m1(Z, path, start, end, pts):

    table = np.zeros((1, pts))
    table[0] = np.logspace(np.log10(start), np.log10(end), pts)

    np.savetxt(path + "/axis_elec.txt", table, fmt='%.8e',
               delimiter=",  ", newline=", &\n")
    return table

# The second table is for the values of gk:


def make_Z_table_gk(Z, path, start, end, pts):

    table = np.zeros((1, pts))
    #table[0] = np.linspace(start, end, pts)
    table[0][1::] = np.logspace(np.log10(start), np.log10(end), int(pts-1))

    np.savetxt(path + "/axis_phot.txt", table, fmt='%.8e',
               delimiter=",  ", newline=", &\n")
    return table

# need to check


# The third table is for values of the total cross section


def make_Z_cs_table(Z, axis_ge, path, mode=None):

    table = []
    ext = ""

    table.append([qed_rates.br_cs(Z, ge_m1 + 1, mode)
                  for ge_m1 in axis_ge[0]])

    if mode == "k":
        ext = "_k"

    np.savetxt(path + "/cs_Z_{:d}.txt".format(Z), table,  fmt='%.8e',
               delimiter=",  ", newline=", &\n")

# Fourth table is for cdf values


def make_Z_cdf_table(Z, axis_ge, axis_gk, path, mode=None):

    table = []
    ext = ""

    if mode == "k":
        ext = "_k"
        for i in range(len(axis_ge[0])):
            ge_m1 = axis_ge[0][i]
            table.append([qed_rates.br_cdf(Z, k * (ge_m1), ge_m1 + 1)
                          for k in axis_gk[0]])
    else:
        for i in range(len(axis_ge[0])):
            ge_m1 = axis_ge[0][i]
            result = [qed_rates.br_cdf_gauleg(Z, k * (ge_m1), ge_m1 + 1, True)
                          for k in axis_gk[0][1::]]
            result.insert(0, 0.)
            table.append(result)

    np.savetxt(path + "/cdf_Z_{:d}.txt".format(Z), table,  fmt='%.8e',
               delimiter=",  ", newline=", &\n")


# Parameters
path = "tables/"

# atomic number
Z = 13

# kinetic energy of the incident electron
start = 1 * 10 ** - 3 # 0.5 keV
end = 4.e4            # 20 GeV
pts = 70              # number of points to discretise

# Axis with the discretization of electron kinetic energy
axis_ge_m1 = make_Z_table_ge_m1(Z, path, start, end, pts)

# Ratio = photon energy / kinetic energy of electron
start = 1.e-7
end = 1
pts = 70

# Axis with the discretization of photon energy
axis_gk = make_Z_table_gk(Z, path, start, end, pts)

# Table with total cross-section
make_Z_cs_table(Z, axis_ge_m1, path+"/Z={:d}".format(Z))

# Table with the CDF of the differential cross section
make_Z_cdf_table(Z, axis_ge_m1, axis_gk, path+"/Z={:d}".format(Z))
