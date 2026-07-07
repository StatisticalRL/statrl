import os
import pickle

def dump(values, filename, tag, root_folder):
    filenameM = root_folder + filename + "_" + tag
    file = open(filenameM, 'wb')
    file.truncate(0)
    pickle.dump(values, file)
    file.close()
    return filenameM

def clear_auxiliaryfiles(env, root_folder):
    for file in os.listdir(root_folder):
        if file.startswith("aux_" + env.name):
            os.remove(root_folder + file)
