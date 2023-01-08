import random
import numpy as np

detector = "V1"

injections = np.load("../datasets/injection_triggers_" + detector + ".npy")
blips = np.load("../datasets/blip_triggers_" + detector + ".npy")
if detector != "V1":
    fast_scatterings = np.load("../datasets/fast_scattering_triggers_" + detector + ".npy")
koyfishes = np.load("../datasets/koyfish_triggers_" + detector + ".npy")
lowfreqs = np.load("../datasets/lowfreq_triggers_" + detector + ".npy")
tomtes = np.load("../datasets/tomte_triggers_" + detector + ".npy")
whistles = np.load("../datasets/whistle_triggers_" + detector + ".npy")

if detector != "V1":
    biggest = max(len(blips), len(fast_scatterings), len(koyfishes), len(lowfreqs), len(tomtes), len(whistles))
else:
    biggest = max(len(blips), len(koyfishes), len(lowfreqs), len(tomtes), len(whistles))

print("biggest: " + str(biggest))

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
if detector != "V1":
    fast_scattering_boot = bootstrap(fast_scatterings)
koyfish_boot = bootstrap(koyfishes)
lowfreq_boot = bootstrap(lowfreqs)
tomte_boot = bootstrap(tomtes)
whistle_boot = bootstrap(whistles)

injection_boot = injections[0:round(biggest/100)]

dataset = np.append(injection_boot, blip_boot, axis = 0)
if detector != "V1":
    dataset = np.append(dataset, fast_scattering_boot, axis = 0)
dataset = np.append(dataset, koyfish_boot, axis = 0)
dataset = np.append(dataset, lowfreq_boot, axis = 0)
dataset = np.append(dataset, tomte_boot, axis = 0)
dataset = np.append(dataset, whistle_boot, axis = 0)

np.random.shuffle(dataset)

np.save('../datasets/dataset_all_' + detector + '_bootstrap.npy', dataset)
print("Saved.")