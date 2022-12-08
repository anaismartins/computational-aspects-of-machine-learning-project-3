# general modules
import os

#torch
from torch.utils.data import DataLoader

# my modules
from LoadCSV_ana import LoadCSV
import globals_ana as g
<<<<<<< HEAD
=======
from plots_ana import plots
>>>>>>> e5c1b640bfb5c743792c7887f351de5857f24bda
        
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
<<<<<<< HEAD
=======

# plot

plots(train_data_blip[0].snr, train_data_blip[0].snr, train_data_injections[0].snr, train_data_injections[0].snr, "SNR", "SNR")
plots(train_data_blip[0].chisq, train_data_blip[0].chisq, train_data_injections[0].chisq, train_data_injections[0].chisq, "chisq", "chisq")
plots(train_data_blip[0].mass_1, train_data_blip[0].mass_1, train_data_injections[0].mass_1, train_data_injections[0].mass_1, "mass_1", "mass_1")
plots(train_data_blip[0].mass_2, train_data_blip[0].mass_2, train_data_injections[0].mass_2, train_data_injections[0].mass_2, "mass_2", "mass_2")
plots(train_data_blip[0].spin1z, train_data_blip[0].spin1z, train_data_injections[0].spin1z, train_data_injections[0].spin1z, "spin1z", "spin1z")
plots(train_data_blip[0].spin2z, train_data_blip[0].spin2z, train_data_injections[0].spin2z, train_data_injections[0].spin2z, "spin2z", "spin2z")

plots(train_data_blip[0].snr, train_data_blip[0].chisq, train_data_injections[0].snr, train_data_injections[0].chisq, "SNR", "chisq")
plots(train_data_blip[0].snr, train_data_blip[0].mass_1, train_data_injections[0].snr, train_data_injections[0].mass_1, "SNR", "mass_1")
plots(train_data_blip[0].snr, train_data_blip[0].mass_2, train_data_injections[0].snr, train_data_injections[0].mass_2, "SNR", "mass_2")
plots(train_data_blip[0].snr, train_data_blip[0].spin1z, train_data_injections[0].snr, train_data_injections[0].spin1z, "SNR", "spin1z")
plots(train_data_blip[0].snr, train_data_blip[0].spin2z, train_data_injections[0].snr, train_data_injections[0].spin2z, "SNR", "spin2z")

plots(train_data_blip[0].chisq, train_data_blip[0].mass_1, train_data_injections[0].chisq, train_data_injections[0].mass_1, "chisq", "mass_1")
plots(train_data_blip[0].chisq, train_data_blip[0].mass_2, train_data_injections[0].chisq, train_data_injections[0].mass_2, "chisq", "mass_2")
plots(train_data_blip[0].chisq, train_data_blip[0].spin1z, train_data_injections[0].chisq, train_data_injections[0].spin1z, "chisq", "spin1z")
plots(train_data_blip[0].chisq, train_data_blip[0].spin2z, train_data_injections[0].chisq, train_data_injections[0].spin2z, "chisq", "spin2z")

plots(train_data_blip[0].mass_1, train_data_blip[0].mass_2, train_data_injections[0].mass_1, train_data_injections[0].mass_2, "mass_1", "mass_2")
plots(train_data_blip[0].mass_1, train_data_blip[0].spin1z, train_data_injections[0].mass_1, train_data_injections[0].spin1z, "mass_1", "spin1z")
plots(train_data_blip[0].mass_1, train_data_blip[0].spin2z, train_data_injections[0].mass_1, train_data_injections[0].spin2z, "mass_1", "spin2z")

plots(train_data_blip[0].mass_2, train_data_blip[0].spin1z, train_data_injections[0].mass_2, train_data_injections[0].spin1z, "mass_2", "spin1z")
plots(train_data_blip[0].mass_2, train_data_blip[0].spin2z, train_data_injections[0].mass_2, train_data_injections[0].spin2z, "mass_2", "spin2z")

plots(train_data_blip[0].spin1z, train_data_blip[0].spin2z, train_data_injections[0].spin1z, train_data_injections[0].spin2z, "spin1z", "spin2z")
>>>>>>> e5c1b640bfb5c743792c7887f351de5857f24bda
