from plots_function_ana import plots

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