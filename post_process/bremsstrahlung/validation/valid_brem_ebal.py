import numpy as np
import matplotlib.pyplot as plt
import qed_rates, read_files
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha
plt.style.use('~/Documents/studies/Libraries/presentation.style')

#--------------------------------------------------------------------
# Simulated spectrum
#--------------------------------------------------------------------

# Read the RAW file
folder = "/Users/bertrand/Documents/osiris/simulations/main_bertrand/unit_test_qed/betheheitler/"
case = "energy_balance.2d/"

#----------------------------------------------------------------
# Total energy balance
#----------------------------------------------------------------

file_ene = 'HIST/pho01_pairs'
path = folder + case + file_ene
with open(path,"r") as f:
    data = f.readlines()

N_dump = len(data[1:])
time_posi, etot_posi = np.zeros([N_dump]), np.zeros([N_dump])

for i_dump, line in zip( range(N_dump) , data[1:] ) :
    line = line.split()
    time_posi[i_dump] = float(line[1])
    etot_posi[i_dump] = float(line[7])

file_ene = 'HIST/pho01_ene'
path = folder + case + file_ene
with open(path,"r") as f:
    data = f.readlines()

N_dump = len(data[2:])
time_elec, etot_elec = np.zeros([N_dump]), np.zeros([N_dump])

for i_dump, line in zip( range(N_dump) , data[2:] ) :
    line = line.split()
    time_elec[i_dump] = float(line[1])
    etot_elec[i_dump] = float(line[3])

fig, axs = plt.subplots(1, 1, figsize=(9,6))
axs.plot(time_elec, etot_elec, label=r'$ E_\gamma(t) $')
axs.plot(time_posi[1::], etot_elec[0] - etot_posi[0:-1], marker='o', ls='', label=r'$ E_\gamma(0) - E_\pm(t) $')
axs.set_xlabel(r"$ t \, (\omega_0^{-1}) $")
axs.set_ylabel(r"$ E (t) \, \rm (a.u.) $")
plt.legend(loc='best')
plt.tight_layout()
filename = 'pics/valid_brem_ebal.png'
fig.savefig(filename)
plt.show()