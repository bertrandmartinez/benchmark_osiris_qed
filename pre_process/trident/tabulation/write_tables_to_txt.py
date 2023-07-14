import numpy as np
import matplotlib.pyplot as plt
import qed_rates
plt.style.use('~/Documents/studies/Libraries/presentation.style')

# path to write tables
path = 'tables'

# Z atomic number
Z = 13

# Number of points on each axis of the table
N_g1 = 40   # Energy of incident electron
N_gpa = 40  # energy of the pair
N_gp = 20   # energy of the emitted positron

# Array initialisation
cs_tot = np.zeros([N_g1])
cdf_diff = np.zeros([N_g1, N_gpa])
cdf_ddiff = np.zeros([N_gp])

# Energy of incident electron (mc2)
min_g1, max_g1 = 6.0, 4.0e4
axis_g1 = np.logspace(np.log10(min_g1), np.log10(max_g1), N_g1)

# Table with range for incident electron energy
np.savetxt(path + "/axis_elec.txt", axis_g1, fmt='%.8e',
            delimiter=",  ", newline=", ")

# Energy of pair energy
min_gpa, max_gpa = 1.e-5, 1.e0
axis_gpa = np.zeros(N_gpa)
axis_gpa[1::] = np.logspace(np.log10(min_gpa), np.log10(max_gpa), N_gpa-1)

# Table with range for pair energy
np.savetxt(path + "/axis_pair.txt", axis_gpa, fmt='%.8e',
            delimiter=",  ", newline=", ")

# Energy of emitted positron (mc2)
min_gp, max_gp = 0.0, 1.0
axis_gp = np.linspace(min_gp, max_gp, N_gp)

# Table with range for incident electron energy
np.savetxt(path + "/axis_posi.txt", axis_gp, fmt='%.8e',
            delimiter=",  ", newline=", ")

#---------------------------------------------------------------------
# Total cross-section
#---------------------------------------------------------------------

# The dependence in Z is trivial, so we will include it directly in Osiris
# This way, we have the table for all atomic numbers
cs_tot = np.array([qed_rates.ct_cs_tot(Z, g1) for g1 in axis_g1])
cs_tot /= Z**2

# Table with total cross-section
np.savetxt(path + "/cs_Z_all.txt", cs_tot, fmt='%.8e',
            delimiter=",  ", newline=", ")

#---------------------------------------------------------------------
# CDF of the differential cross-section in pair energy
#---------------------------------------------------------------------

error_ddiff, error_diff = np.zeros([N_gp]), np.zeros([N_g1, N_gpa])

# Loop on electron energy
for i_g1 in range(N_g1):

    g1 = axis_g1[i_g1]

    # Loop on pair energy
    for i_gpa in range(1,N_gpa-1):

        gpa = 2.0 + axis_gpa[i_gpa] * (g1 - 3.0)
        cdf_diff[i_g1,i_gpa] = qed_rates.ct_cs_diff_cdf(Z, g1, gpa, 'gauleg_log')

    cdf_diff[i_g1,N_gpa-1] = 1.0

# Generic CDF
gpa = 1000.
cdf_ddiff = np.array([qed_rates.ct_cs_ddif_cdf_generic(gpa, 1.0 + axis_gp[i_gp] * (gpa - 2.0)) for i_gp in range(N_gp)])

#---------------------------------------------------------------------
# Sanity checks on the table
#---------------------------------------------------------------------

# Loop on electron energy
for i_g1 in range(N_g1):
    for i_gpa in range(1,N_gpa):
        bounds = (round(cdf_diff[i_g1,i_gpa],8)>1.0) or (round(cdf_diff[i_g1,i_gpa],8)<0.0)
        monotony = (cdf_diff[i_g1,i_gpa-1]>cdf_diff[i_g1,i_gpa])
        if bounds or monotony :
            error_diff[i_g1,i_gpa] = True
            print(axis_g1[i_g1], 2.0 + axis_gpa[i_gpa] * (axis_g1[i_g1] - 3.0))

for i_gp in range(1,N_gp):
    bounds = (round(cdf_ddiff[i_gp],8)>1.0) or (round(cdf_ddiff[i_gp],8)<0.0)
    monotony = (round(cdf_ddiff[i_gp-1],8)>round(cdf_ddiff[i_gp],8))
    if bounds or monotony :
        error_ddiff[i_gp] = True

print('\nerror spotted in cdf_diff = {}'.format(error_diff.any()))
print('error spotted in cdf_ddiff = {}'.format(error_ddiff.any()))

#---------------------------------------------------------------------
# Write the tables
#---------------------------------------------------------------------

np.savetxt(path + "/cdf1_Z_all.txt", cdf_diff,  fmt='%.8e',
            delimiter=",  ", newline=", &\n")

np.savetxt(path + "/cdf2_Z_all.txt", cdf_ddiff, fmt='%.8e',
            delimiter=",  ", newline=", ")