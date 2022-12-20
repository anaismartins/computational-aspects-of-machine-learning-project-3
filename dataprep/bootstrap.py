import random
import numpy as np

injections = np.load("../datasets/injection_triggers.npy")
blips = np.load("../datasets/blip_triggers.npy")
fast_scatterings = np.load("../datasets/fast_scattering_triggers.npy")
koyfishes = np.load("../datasets/koyfish_triggers.npy")
lowfreqs = np.load("../datasets/lowfreq_triggers.npy")
tomtes = np.load("../datasets/tomte_triggers.npy")
whistles = np.load("../datasets/whistle_triggers.npy")

blips_boot = np.array([])
fast_scattering_boot = np.array([])
koyfish_boot = np.array([])
lowfreq_boot = np.array([])
tomte_boot = np.array([])
whistle_boot = np.array([])

biggest = 2649

for i in range(0, 100):
    blips_boot = np.append(blips_boot, random.sample(blips.to_list(), round(biggest/100)))
    fast_scattering_boot = np.append(fast_scattering_boot, random.sample(fast_scatterings.to_list(), round(biggest/100)))
    koyfish_boot = np.append(koyfish_boot, random.sample(koyfishes.to_list(), round(biggest/100)))
    lowfreq_boot = np.append(lowfreq_boot, random.sample(lowfreqs.to_list(), round(biggest/100)))
    tomte_boot = np.append(tomte_boot, random.sample(tomtes.to_list(), round(biggest/100)))
    whistle_boot = np.append(whistle_boot, random.sample(whistles.to_list(), round(biggest/100)))

print(blips_boot.shape)
