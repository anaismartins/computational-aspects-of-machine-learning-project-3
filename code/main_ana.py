# general modules
import os
import matplotlib.pyplot as plt

#torch
from torch.utils.data import DataLoader

# my modules
from LoadCSV_ana import LoadCSV
import globals_ana as g
        
train_data_blip = []
test_data_blip = []
train_data_injections = []
test_data_injections = []

filenames = os.listdir(g.path_to_data)

for filename in filenames:
    if ".csv" in filename:
        for classification in g.classifications:
            if classification in filename:
                if "O3a" and "Blip" in filename:
                    train_data_blip.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3b_old" and "Blip" in filename:
                    test_data_blip.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3a" and "Injections" in filename:
                    train_data_injections.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3b_old" and "Injections" in filename:
                    test_data_injections.append(LoadCSV(filename, g.path_to_data, classification))
    
#train_dataloader = DataLoader(train_data, batch_size = g.batch_size, shuffle = True)
#test_dataloader = DataLoader(test_data, batch_size = g.batch_size, shuffle = True)

# plot

plt.scatter(train_data_blip[0].snr, train_data_blip[0].snr, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].snr, train_data_injections[0].snr, label = "Injections", s = 1)
plt.legend()
plt.title("SNR vs SNR")
plt.savefig("SNRvsSNR.png")
plt.clf()

plt.scatter(train_data_blip[0].chisq, train_data_blip[0].chisq, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].chisq, train_data_injections[0].chisq, label = "Injections", s = 1)
plt.legend()
plt.title("Chisq vs Chisq")
plt.savefig("ChisqvsChisq.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_1, train_data_blip[0].mass_1, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_1, train_data_injections[0].mass_1, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 1 vs Mass 1")
plt.savefig("Mass1vsMass1.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_2, train_data_blip[0].mass_2, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_2, train_data_injections[0].mass_2, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 2 vs Mass 2")
plt.savefig("Mass2vsMass2.png")
plt.clf()

plt.scatter(train_data_blip[0].spin1z, train_data_blip[0].spin1z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].spin1z, train_data_injections[0].spin1z, label = "Injections", s = 1)
plt.legend()
plt.title("Spin 1 vs Spin 1")
plt.savefig("Spin1vsSpin1.png")
plt.clf()

plt.scatter(train_data_blip[0].spin2z, train_data_blip[0].spin2z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].spin2z, train_data_injections[0].spin2z, label = "Injections", s = 1)
plt.legend()
plt.title("Spin 2 vs Spin 2")
plt.savefig("Spin2vsSpin2.png")
plt.clf()

plt.scatter(train_data_blip[0].snr, train_data_blip[0].chisq, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].snr, train_data_injections[0].chisq, label = "Injections", s = 1)
plt.legend()
plt.title("SNR vs Chisq")
plt.savefig("SNRvsChisq.png")
plt.clf()

plt.scatter(train_data_blip[0].snr, train_data_blip[0].mass_1, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].snr, train_data_injections[0].mass_1, label = "Injections", s = 1)
plt.legend()
plt.title("SNR vs Mass 1")
plt.savefig("SNRvsMass1.png")
plt.clf()

plt.scatter(train_data_blip[0].snr, train_data_blip[0].mass_2, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].snr, train_data_injections[0].mass_2, label = "Injections", s = 1)
plt.legend()
plt.title("SNR vs Mass 2")
plt.savefig("SNRvsMass2.png")
plt.clf()

plt.scatter(train_data_blip[0].snr, train_data_blip[0].spin1z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].snr, train_data_injections[0].spin1z, label = "Injections", s = 1)
plt.legend()
plt.title("SNR vs Spin 1")
plt.savefig("SNRvsSpin1.png")
plt.clf()

plt.scatter(train_data_blip[0].snr, train_data_blip[0].spin2z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].snr, train_data_injections[0].spin2z, label = "Injections", s = 1)
plt.legend()
plt.title("SNR vs Spin 2")
plt.savefig("SNRvsSpin2.png")
plt.clf()

plt.scatter(train_data_blip[0].chisq, train_data_blip[0].mass_1, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].chisq, train_data_injections[0].mass_1, label = "Injections", s = 1)
plt.legend()
plt.title("Chisq vs Mass 1")
plt.savefig("ChisqvsMass1.png")
plt.clf()

plt.scatter(train_data_blip[0].chisq, train_data_blip[0].mass_2, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].chisq, train_data_injections[0].mass_2, label = "Injections", s = 1)
plt.legend()
plt.title("Chisq vs Mass 2")
plt.savefig("ChisqvsMass2.png")
plt.clf()

plt.scatter(train_data_blip[0].chisq, train_data_blip[0].spin1z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].chisq, train_data_injections[0].spin1z, label = "Injections", s = 1)
plt.legend()
plt.title("Chisq vs Spin 1")
plt.savefig("ChisqvsSpin1.png")
plt.clf()

plt.scatter(train_data_blip[0].chisq, train_data_blip[0].spin2z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].chisq, train_data_injections[0].spin2z, label = "Injections", s = 1)
plt.legend()
plt.title("Chisq vs Spin 2")
plt.savefig("ChisqvsSpin2.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_1, train_data_blip[0].mass_2, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_1, train_data_injections[0].mass_2, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 1 vs Mass 2")
plt.savefig("Mass1vsMass2.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_1, train_data_blip[0].spin1z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_1, train_data_injections[0].spin1z, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 1 vs Spin 1")
plt.savefig("Mass1vsSpin1.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_1, train_data_blip[0].spin2z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_1, train_data_injections[0].spin2z, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 1 vs Spin 2")
plt.savefig("Mass1vsSpin2.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_2, train_data_blip[0].spin1z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_2, train_data_injections[0].spin1z, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 2 vs Spin 1")
plt.savefig("Mass2vsSpin1.png")
plt.clf()

plt.scatter(train_data_blip[0].mass_2, train_data_blip[0].spin2z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].mass_2, train_data_injections[0].spin2z, label = "Injections", s = 1)
plt.legend()
plt.title("Mass 2 vs Spin 2")
plt.savefig("Mass2vsSpin2.png")
plt.clf()

plt.scatter(train_data_blip[0].spin1z, train_data_blip[0].spin2z, label = "Blip", s = 1)
plt.scatter(train_data_injections[0].spin1z, train_data_injections[0].spin2z, label = "Injections", s = 1)
plt.legend()
plt.title("Spin 1 vs Spin 2")
plt.savefig("Spin1vsSpin2.png")
plt.clf()