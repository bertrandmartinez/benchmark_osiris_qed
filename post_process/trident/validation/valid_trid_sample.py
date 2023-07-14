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
p1 = np.array([6.8, 8.77, 20.5, 197., 1958., 39140.])
gam1 = np.sqrt(1.0 + p1**2)
vrel = np.sqrt(1.0 - 1.0 / gam1**2)
sigma = np.array([qed_rates.ct_cs_tot(Z, elem) for elem in gam1])
Vg = 8.0 * 8.0
ng = 1.e-4
ni = 100.

# Numerical parameters
cell_vol = 0.04**2
dt = 0.02
N_dt = 1

# Bounds for positrons energy
min_p = 1.0
max_p = (gam1 - 1.0) / 2.0

#--------------------------------------------------------------------
# Simulated spectrum
#--------------------------------------------------------------------

# Read the RAW file
folder = "/Users/bertrand/Documents/osiris/simulations/main_bertrand/unit_test_qed/trident_coul/"
case = ["tridentcoul_elec_3MeV.2d/", "tridentcoul_elec_4MeV.2d/", "tridentcoul_elec_10MeV.2d/", "tridentcoul_elec_100MeV.2d/", "tridentcoul_elec_1GeV.2d/", "tridentcoul_elec_20GeV.2d/"]
file = 'MS/RAW/positrons/RAW-positrons-000001.h5'
labels = [r"$(\gamma_1 - 1) mc^2 = 3 \, \rm MeV $", r"$(\gamma_1 - 1) mc^2 = 4 \, \rm MeV $", r"$(\gamma_1 - 1) mc^2 = 10 \, \rm MeV $", r"$(\gamma_1 - 1) mc^2 = 100 \, \rm MeV $", r"$(\gamma_1 - 1) mc^2 = 1 \, \rm GeV $", r"$(\gamma_1 - 1) mc^2 = 20 \, \rm GeV $"]

# Array initialisation
N_case = len(case)
dNdp_raw, gamp_raw = [[] for n1 in range(N_case)], [[] for n1 in range(N_case)]
gamp_the, dNdp_the = [[] for n1 in range(N_case)], [[] for n1 in range(N_case)]
gamp_the_full, dNdp_the_full = [[] for n1 in range(N_case)], [[] for n1 in range(N_case)]

for i_case in range(N_case):

    path = folder + case[i_case] + file
    quants = ['q', 'ene']
    data_raw = read_files.read_raw(path, quants)
    data_raw['gamma'] = data_raw['ene'] + 1.0

    # Total number of pairs in the simulation
    number_raw = np.sum(data_raw['q']) * cell_vol

    # Simulated spectrum for Trident Coulomb
    N = 400
    if i_case > 3:
        N = 100000
    dNdp_raw[i_case], bin_edges = np.histogram(data_raw['gamma'], bins=N, range=(min_p, max_p[i_case]))
    dNdp_raw[i_case] = np.array(dNdp_raw[i_case], dtype='float64')

    # We arrange the positron energy axis
    gamp_raw[i_case] = (bin_edges[1::] + bin_edges[0:-1]) / 2.

    # We normalise its integral by the number of positrons in the simulation
    norm = integral(gamp_raw[i_case], dNdp_raw[i_case])
    dNdp_raw[i_case] *= (number_raw/norm)

    # We convert it from plasma units to physical units
    dNdp_raw[i_case] *= norm_weight

    # Convert the spectrum of positron to the spectrum of pairs
    # assuming gam_pa = 2 * gam_p as in Osiris in the test input deck
    gamp_raw[i_case] *= 2.0
    dNdp_raw[i_case] /= 2.0

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

    # Theoretical spectrum for Trident
    min_gpa, max_gpa, M = 2.0, gam1[i_case]-1.0, 1000
    gamp_the[i_case] = np.logspace(np.log10(min_gpa), np.log10(max_gpa), M)

    dNdp_the[i_case] = np.array([ qed_rates.ct_cs_dif(Z, gam1[i_case], gamp) for gamp in gamp_the[i_case]])

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

    gamp_the_full[i_case] = np.linspace(min_p, max_p[i_case], M)
    dNdp_the_full[i_case] = np.array([ qed_rates.ct_cs_dif(Z, gam1[i_case], gamp) for gamp in gamp_the_full[i_case]])
    
    # We normalise its integral to the number of pairs expected
    #norm = integral(gamp_the_full[i_case], dNdp_the_full[i_case])
    dNdp_the_full[i_case] *= (number_the/norm)

    # We convert it from plasma units to physical units
    dNdp_the_full[i_case] *= norm_weight

#----------------------------------------------------------------
# Single figure with all subfigures
#----------------------------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(24,10))

for i_case in range(N_case):

    i = i_case // 3
    j = i_case % 3
    ax = axs[i,j]

    xmin = max(1.e-5, 1./(gam1[i_case]-1.))
    ymax = np.max(dNdp_raw[i_case])
    ax.plot((gamp_raw[i_case]-2.)/(gam1[i_case]-3), dNdp_raw[i_case], label=labels[i_case])
    if i_case ==2 :
        ax.plot((gamp_the[i_case]-2.)/(gam1[i_case]-3), dNdp_the[i_case], c='k', label=r"$ \rm theory $")
    else:
        ax.plot((gamp_the[i_case]-2.)/(gam1[i_case]-3), dNdp_the[i_case], c='k')
    ax.set_xlabel(r"$ (\gamma_p-2) / (\gamma_1 - 3) $")
    
    ax.set_xscale('log')
    ax.set_xlim([0.5*xmin, 1.1])
    ax.set_ylabel(r"$ dN/d\gamma_p $")
    if i_case > 2 :
        ax.set_yscale('log')
        ax.set_ylim([1.e-10, 7.e-5])
    else:
        ax.set_ylim([-0.05*ymax, 1.5*ymax])
    ax.legend()
    ax.text(0.02, 0.9, subfig[i_case], ha='left', va='center', transform=ax.transAxes)

plt.tight_layout()
filename = 'pics/valid_trid_sample'
fig.savefig(filename)
plt.show()