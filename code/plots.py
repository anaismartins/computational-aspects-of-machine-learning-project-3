def plots(blip, param1, param2, param1_name, param2_name):
    plt.scatter(blip.param1, train_data_blip[0].param2, label = "Blip", s = 1)
    plt.scatter(injections.param1, train_data_injections[0].param2, label = "Injections", s = 1)
    plt.legend()
    plt.title(param1_name + " vs " + param2_name)
    plt.savefig(param1_name + "vs" + param2_name + ".png")
    plt.clf()