import numpy as np
import matplotlib.pyplot as plt
import qed_rates
plt.style.use('~/Documents/studies/Libraries/presentation.style')

# path to write tables
path = 'tables'

# Z atomic number
Z = 29

#----------------------------------------------------------------
# Range for photon energy and positron energy
#----------------------------------------------------------------

Nk, Ng = 40, 20
k_range = np.logspace(np.log10(2.1e0), np.log10(4.e4), Nk)
g_range = np.linspace(0.0, 1.0, Ng)

# Table with range for incident electron energy
np.savetxt(path + "/axis_phot.txt", k_range, fmt='%.8e',
            delimiter=",  ", newline=", ")

# Table with range for incident electron energy
np.savetxt(path + "/axis_posi.txt", g_range, fmt='%.8e',
            delimiter=",  ", newline=", ")

#----------------------------------------------------------------
# Theoretical cdf
#----------------------------------------------------------------

table = []
cdf_the = np.zeros([Nk, Ng])
for ik in range(Nk):
    table.append([qed_rates.bh_cdf(Z, k_range[ik], g) for g in g_range])

np.savetxt(path + "/Z={:d}/cdf_Z_{:d}.txt".format(Z,Z), table,  fmt='%.8e',
          delimiter=",  ", newline=", &\n")

#----------------------------------------------------------------
# Polynomial coefficients for the total cross section
# Three regimes, and 5 coefficients for each
#----------------------------------------------------------------

N = 100
iflogx, iflogf = True, True
polyn_coefs = []

#----------------------------------------------------------------
# From 1 to 2 MeV (relative difference on whole interval is < 2%)
#----------------------------------------------------------------

k_range = np.logspace(np.log10(2.1e0), np.log10(4.e0), N)
degree = 4
cs_tot_the = np.array([ qed_rates.bh_cs(Z, k) for k in k_range ])
cs_tot_fit_numpy_1D_low, coefs = qed_rates.fit_numpy_1D(k_range, cs_tot_the, degree, iflogx, iflogf, 'auto')
polyn_coefs.append(coefs)

#----------------------------------------------------------------
# From 2 MeV to 1 GeV (relative difference on whole interval is < 2%)
#----------------------------------------------------------------

k_range = np.logspace(np.log10(4.e0), np.log10(2.e3), N)
degree = 4
cs_tot_the = np.array([ qed_rates.bh_cs(Z, k) for k in k_range ])
cs_tot_fit_numpy_1D_mid, coefs = qed_rates.fit_numpy_1D(k_range, cs_tot_the, degree, iflogx, iflogf, 'auto')
polyn_coefs.append(coefs)

#----------------------------------------------------------------
# From 1 GeV to 20 GeV (relative difference on whole interval is < 2%)
#----------------------------------------------------------------

k_range = np.logspace(np.log10(2.0e3), np.log10(4.0e4), N)
degree = 4
cs_tot_the = np.array([ qed_rates.bh_cs(Z, k) for k in k_range ])
cs_tot_fit_numpy_1D_hig, coefs = qed_rates.fit_numpy_1D(k_range, cs_tot_the, degree, iflogx, iflogf, 'auto')
polyn_coefs.append(coefs)

# Save the coefficiencts in a txt file
np.savetxt(path + "/Z={:d}/cs_Z_{:d}.txt".format(Z,Z), polyn_coefs,  fmt='%.8e',
          delimiter=",  ", newline=", &\n")