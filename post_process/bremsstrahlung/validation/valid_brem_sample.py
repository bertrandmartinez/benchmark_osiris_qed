import numpy as np
import matplotlib.pyplot as plt
import qed_rates, read_files
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha
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
p1 = np.array([6.266e-2, 6.555e-1, 10.7, 79.3, 1958.0, 39140.0])
gam1 = np.sqrt(1.0 + p1**2)
vrel = np.sqrt(1.0 - 1.0 / gam1**2)
sigma = np.array([qed_rates.br_cs(Z, elem) for elem in gam1])
Vg = 8.0 * 8.0
ng = 1.e-4
ni = 100.

# Numerical parameters
cell_vol = 0.04**2
dt = 0.02
N_dt = 1

# Bounds for photon energy
min_k = 1.e-8*(gam1-1.0)
max_k = (gam1-1.0)

#--------------------------------------------------------------------
# Simulated spectrum
#--------------------------------------------------------------------

# Read the RAW file
folder = "/Users/bertrand/Documents/osiris/simulations/main_bertrand/unit_test_qed/bremsstrahlung/"
case = ["bremsstrahlung_elec_1keV.2d/", "bremsstrahlung_elec_100keV.2d/", "bremsstrahlung_elec_5MeV.2d/", "bremsstrahlung_elec_40MeV.2d/", "bremsstrahlung_elec_1GeV.2d/", "bremsstrahlung_elec_20GeV.2d/"]
file = 'MS/RAW/photons/RAW-photons-000001.h5'
labels = [r"$ (\gamma_1 - 1) mc^2 = 1 \, \rm keV $", r"$ (\gamma_1 - 1) mc^2 = 100 \, \rm keV $", r"$ (\gamma_1 - 1) mc^2 = 5 \, \rm MeV $", r"$ (\gamma_1 - 1) mc^2 = 40 \, \rm MeV $", r"$ (\gamma_1 - 1) mc^2 = 1 \, \rm GeV $", r"$ (\gamma_1 - 1) mc^2 = 20 \, \rm GeV $"]

# Array initialisation
N_case = len(case)
dNdp_raw, dNdp_the, gamp_raw = [[] for n1 in range(N_case)], [[] for n1 in range(N_case)], [[] for n1 in range(N_case)]
cs = [[] for n1 in range(N_case)]

for i_case in range(N_case):

    path = folder + case[i_case] + file
    quants = ['q', 'ene', 'p1']
    data_raw = read_files.read_raw(path, quants)
    data_raw['gamma'] = data_raw['ene'] + 1.0

    # Total number of pairs in the simulation
    number_raw = np.sum(data_raw['q']) * cell_vol

    # Simulated spectrum for Trident Coulomb
    N = 400
    dNdp_raw[i_case], bin_edges = np.histogram(np.log10(data_raw['p1']), bins=N, range=(np.log10(min_k[i_case]), np.log10(max_k[i_case])))
    dNdp_raw[i_case] = np.array(dNdp_raw[i_case], dtype='float64')

    # We arrange the photon energy axis
    gamp_raw[i_case] = (bin_edges[1::] + bin_edges[0:-1]) / 2.
    gamp_raw[i_case] = 10**gamp_raw[i_case]
    dNdp_raw[i_case] /= gamp_raw[i_case]

    # We normalise its integral by the number of positrons in the simulation
    norm = integral(gamp_raw[i_case], dNdp_raw[i_case])
    dNdp_raw[i_case] *= (number_raw/norm)

    # We convert it from plasma units to physical units
    dNdp_raw[i_case] *= norm_weight

    #--------------------------------------------------------------------
    # Theoretical spectrum
    #--------------------------------------------------------------------

    # Number of electrons
    N_elec = ng * Vg

    # Probability for pair creation
    var = sigma[i_case] * norm_length * ni * dt * vrel[i_case]
    proba = 1.0 - np.exp(-var)

    # Total number of pairs created
    number_the = N_elec * proba * N_dt

    # Theoretical spectrum for Bremsstrahlung
    cs[i_case] = np.array([qed_rates.br_cs_dif(Z, k, gam1[i_case]) for k in gamp_raw[i_case]])
    dNdp_the[i_case] = np.array([qed_rates.br_cs_dif(Z, k, gam1[i_case]) for k in gamp_raw[i_case]])

    # We normalise its integral to the number of pairs expected
    norm = integral(gamp_raw[i_case], dNdp_the[i_case])
    dNdp_the[i_case] *= (number_the/norm)

    # We convert it from plasma units to physical units
    dNdp_the[i_case] *= norm_weight

    # Sanity check
    integral_raw = integral(gamp_raw[i_case], dNdp_raw[i_case])
    integral_theo = integral(gamp_raw[i_case], dNdp_the[i_case])
    rel_diff = 100 * abs(integral_theo - integral_raw) / integral_theo
    print('relative diff = {:.2f} %'.format(rel_diff))

    dNdp_raw[i_case] *= gamp_raw[i_case]
    dNdp_the[i_case] *= gamp_raw[i_case]

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
        ax.plot(gamp_raw[i_case]/(gam1[i_case]-1.), dNdp_the[i_case], c='k', label=r"$ \rm theory $")
    else:
        ax.plot(gamp_raw[i_case]/(gam1[i_case]-1.), dNdp_the[i_case], c='k')
    ax.plot(gamp_raw[i_case]/(gam1[i_case]-1.), dNdp_raw[i_case], label=labels[i_case])
    ax.set_xlabel(r"$ k / (\gamma_1 -1) $")
    ax.set_xscale('log')
    location = np.logspace(-8., 0., 5)
    ax.xaxis.set_ticks(location)
    ax.xaxis.set_ticklabels([ticks_pow_notation(elem,0) for elem in location])
    ax.set_ylabel(r"$ k dN/dk $")
    ax.set_ylim([-0.05*ymax, 1.2*ymax])
    ax.legend(loc='best')
    ax.text(0.02, 0.9, subfig[i_case], ha='left', va='center', transform=ax.transAxes)

plt.tight_layout()
filename = 'pics/valid_brem_sample'
fig.savefig(filename)

plt.show()