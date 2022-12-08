# general modules
import os
import matplotlib.pyplot as plt

#torch
from torch.utils.data import DataLoader

# my modules
from LoadCSV_ana import LoadCSV
import globals_ana as g
from plots import plots
        
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

plots(train_data_blip[0], train_data_injections[0], snr, snr, "SNR", "SNR")
plots(train_data_blip[0], train_data_injections[0], chisq, chisq, "Chisq", "Chisq")
plots(train_data_blip[0], train_data_injections[0], mass_1, mass_1, "Mass 1", "Mass 1")
plots(train_data_blip[0], train_data_injections[0], mass_2, mass_2, "Mass 2", "Mass 2")
plots(train_data_blip[0], train_data_injections[0], spin1z, spin1z, "Spin 1z", "Spin 1z")
plots(train_data_blip[0], train_data_injections[0], spin2z, spin2z, "Spin 2z", "Spin 2z")

plots(train_data_blip[0], train_data_injections[0], snr, chisq, "SNR", "Chisq")
plots(train_data_blip[0], train_data_injections[0], snr, mass_1, "SNR", "Mass 1")
plots(train_data_blip[0], train_data_injections[0], snr, mass_2, "SNR", "Mass 2")
plots(train_data_blip[0], train_data_injections[0], snr, spin1z, "SNR", "Spin 1z")
plots(train_data_blip[0], train_data_injections[0], snr, spin2z, "SNR", "Spin 2z")

plots(train_data_blip[0], train_data_injections[0], chisq, mass_1, "Chisq", "Mass 1")
plots(train_data_blip[0], train_data_injections[0], chisq, mass_2, "Chisq", "Mass 2")
plots(train_data_blip[0], train_data_injections[0], chisq, spin1z, "Chisq", "Spin 1z")
plots(train_data_blip[0], train_data_injections[0], chisq, spin2z, "Chisq", "Spin 2z")

plots(train_data_blip[0], train_data_injections[0], mass_1, mass_2, "Mass 1", "Mass 2")
plots(train_data_blip[0], train_data_injections[0], mass_1, spin1z, "Mass 1", "Spin 1z")
plots(train_data_blip[0], train_data_injections[0], mass_1, spin2z, "Mass 1", "Spin 2z")

plots(train_data_blip[0], train_data_injections[0], mass_2, spin1z, "Mass 2", "Spin 1z")
plots(train_data_blip[0], train_data_injections[0], mass_2, spin2z, "Mass 2", "Spin 2z")

plots(train_data_blip[0], train_data_injections[0], spin1z, spin2z, "Spin 1z", "Spin 2z")