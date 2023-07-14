import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
import qed_rates, read_files
plt.style.use('~/Documents/studies/Libraries/presentation.style')

# Normalisation factor for weight
omega_0 = 1.88e15
n_0 = m_e * epsilon_0 * omega_0**2 / e**2
norm_weight = n_0 * ( c / omega_0 )**3
norm_length = n_0 * ( c / omega_0 )
norm_ene = 1.e-6 * m_e * c**2 / e

# Usefull function to calculate integrals
def integral(datax, dataf):
    result = 0.
    L = len(dataf)
    for i in range(L-1):
        result += dataf[i] * (datax[i+1] - datax[i])
    return result

subfig = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]

# Physical parameters
Z = 29.
k = np.array([2.54, 3.52, 19.6, 196., 1957., 39139.])
sigma = np.array([qed_rates.bh_cs(Z, elem) for elem in k])
Vg = 8.0 * 8.0
ng = 1.e-4
ni = 100.

# Numerical parameters
cell_vol = 0.04**2
dt = 0.02
N_dt = 1

# Bounds for positron energy
min_p = 1.0
max_p = k - 1.0

#--------------------------------------------------------------------
# Simulated spectrum
#--------------------------------------------------------------------

# Read the RAW file
folder = "/Users/bertrand/Documents/osiris/simulations/main_bertrand/unit_test_qed/betheheitler/"
case = ["betheheitler_phot_1.3MeV.2d/", "betheheitler_phot_1.8MeV.2d/", "betheheitler_phot_10MeV.2d/", "betheheitler_phot_100MeV.2d/", "betheheitler_phot_1GeV.2d/", "betheheitler_phot_20GeV.2d/"]
file = 'MS/RAW/positrons/RAW-positrons-000001.h5'
labels = [r"$ k mc^2 = 1.3 \, \rm MeV $", r"$ k mc^2 = 1.8 \, \rm MeV $", r"$ k mc^2 = 10 \, \rm MeV $", r"$ k mc^2 = 100 \, \rm MeV $", r"$ k mc^2 = 1 \, \rm GeV $", r"$ k mc^2 = 10 \, \rm GeV $"]

# Array initialisation
N_case = len(case)
dNdp_raw, dNdp_the, gamp_raw = [[] for n1 in range(N_case)], [[] for n1 in range(N_case)], [[] for n1 in range(N_case)]
gamp_the = [[] for n1 in range(N_case)]

for i_case in range(N_case):

    path = folder + case[i_case] + file
    quants = ['q', 'ene']
    data_raw = read_files.read_raw(path, quants)
    data_raw['gamma'] = data_raw['ene'] + 1.0

    # Total number of pairs in the simulation
    number_raw = np.sum(data_raw['q']) * cell_vol

    # Simulated spectrum for Trident Coulomb
    N = 200
    dNdp_raw[i_case], bin_edges = np.histogram(data_raw['gamma'], bins=N, range=(min_p, max_p[i_case]))
    dNdp_raw[i_case] = np.array(dNdp_raw[i_case], dtype='float64')

    # We arrange the positron energy axis
    gamp_raw[i_case] = (bin_edges[1::] + bin_edges[0:-1]) / 2.

    # We normalise its integral by the number of positrons in the simulation
    norm = integral(gamp_raw[i_case], dNdp_raw[i_case])
    dNdp_raw[i_case] *= (number_raw/norm)

    # We convert it from plasma units to physical units
    dNdp_raw[i_case] *= norm_weight

    #--------------------------------------------------------------------
    # Theoretical spectrum
    #--------------------------------------------------------------------

    # Number of photons
    N_elec = ng * Vg

    # Probability for pair creation
    var = sigma[i_case] * norm_length * ni * dt
    proba = 1.0 - np.exp(-var)

    # Total number of pairs created
    number_the = N_elec * proba * N_dt

    # Theoretical spectrum for Bremsstrahlung
    M = 10000
    gamp_the[i_case] = np.linspace(min_p, max_p[i_case], M)
    dNdp_the[i_case] = np.array([qed_rates.bh_cs_dif(elem, k[i_case], Z) for elem in gamp_the[i_case]])

    # We normalise its integral to the number of pairs expected
    norm = integral(gamp_the[i_case], dNdp_the[i_case])
    dNdp_the[i_case] *= (number_the/norm)

    # We convert it from plasma units to physical units
    dNdp_the[i_case] *= norm_weight

    # Sanity check
    integral_raw = integral(gamp_raw[i_case], dNdp_raw[i_case])
    integral_theo = integral(gamp_the[i_case], dNdp_the[i_case])
    rel_diff = 100 * abs(integral_theo - integral_raw) / integral_theo
    print('\nrel diff = {:.2f} %'.format(rel_diff))

#----------------------------------------------------------------
# Single figure with all subfigures
#----------------------------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(24,10))

for i_case in range(N_case):

    i = i_case // 3
    j = i_case % 3
    ax = axs[i,j]

    ymax = np.max(dNdp_raw[i_case])
    if i_case == 0:
        ax.plot((gamp_the[i_case]-1.)/(k[i_case]-2.), dNdp_the[i_case], c='k', label=r"$\rm theory $")
    else:
        ax.plot((gamp_the[i_case]-1.)/(k[i_case]-2.), dNdp_the[i_case], c='k')
    ax.plot((gamp_raw[i_case]-1.)/(k[i_case]-2.), dNdp_raw[i_case], label=labels[i_case])
    ax.set_xlabel(r"$ (\gamma_+-1) / (k-2) $")
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylabel(r"$ dN/d\gamma_+ $")
    ax.set_ylim([-0.05*ymax, 1.2*ymax])
    ax.legend(loc='best')
    ax.text(0.02, 0.9, subfig[i_case], ha='left', va='center', transform=ax.transAxes)

plt.tight_layout()
filename = 'pics/valid_beth_sample'
#fig.savefig(filename)
plt.show()