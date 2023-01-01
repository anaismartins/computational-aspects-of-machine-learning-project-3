import random
import numpy as np

injections = np.load("../datasets/injection_triggers.npy")
blips = np.load("../datasets/blip_triggers.npy")
fast_scatterings = np.load("../datasets/fast_scattering_triggers.npy")
koyfishes = np.load("../datasets/koyfish_triggers.npy")
lowfreqs = np.load("../datasets/lowfreq_triggers.npy")
tomtes = np.load("../datasets/tomte_triggers.npy")
whistles = np.load("../datasets/whistle_triggers.npy")

biggest = 2649

#blips
def bootstrap(glitch):
    glitch_boot = [[]]
    col = []
    nums = np.array([])

    for n in range(0, 7):
        for i in range(0, 100):
            for j in range (0, glitch.shape[0]):
                col.append(glitch[j][n])
            nums = np.append(nums, random.sample(col, round(biggest/100)))

        if n == 0:
            glitch_boot[0] = nums
        else:
            glitch_boot.append(nums)

        nums = np.array([]) 
        col = []

    result = [[glitch_boot[j][i] for j in range(len(glitch_boot))] for i in range(len(glitch_boot[0]))]

    for i in range(len(glitch_boot)):
        for j in range(len(glitch_boot[0])):
            result[j][i] = glitch_boot[i][j]

    return result

blip_boot = bootstrap(blips)
fast_scattering_boot = bootstrap(fast_scatterings)
koyfish_boot = bootstrap(koyfishes)
lowfreq_boot = bootstrap(lowfreqs)
tomte_boot = bootstrap(tomtes)
whistle_boot = bootstrap(whistles)

injection_boot = injections[0:round(biggest/100)]

dataset = np.append(injection_boot, blip_boot, axis = 0)
dataset = np.append(dataset, fast_scattering_boot, axis = 0)
dataset = np.append(dataset, koyfish_boot, axis = 0)
dataset = np.append(dataset, lowfreq_boot, axis = 0)
dataset = np.append(dataset, tomte_boot, axis = 0)
dataset = np.append(dataset, whistle_boot, axis = 0)

np.random.shuffle(dataset)

np.save('../datasets/dataset_all_h1_bootstrap.npy', dataset)
print("Saved.")