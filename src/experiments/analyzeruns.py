
import pickle
import time
import numpy as np

def computeScoreDiffs(names, dump_scores, timeHorizon, envName, root_folder):
    """

    :param names: get list of algorithm names
    :param dump_scores: list of filenames, each getting cumulative rewards for multiple runs. Last file of the list is cum reward of Oracle.
    :param timeHorizon:
    :param envName:
    :return: vectors median, quantile0.25, quantile0.75, timesteps, where median[i] is median of expreimnts at time timesteps[i]
    """

    median = []
    mean = []
    quantile1 = []
    quantile2 = []
    nbAlgs = len(dump_scores) - 1

    #Downsample the times, especially in case timeHorizon is huge.
    skip = max(1, (timeHorizon // 1000))
    times = [t for t in range(0,timeHorizon,skip)]

    #file_oracle = open(dump_scores[-1], 'rb')
    #scores_oracle = pickle.load(file_oracle)
    # Comment the following line for BatchMabs:
    #scores_oracle = scores_oracle[0]
    #file_oracle.close()

    data_o = []
    for i in range(len(dump_scores[-1])):
        #print(f"O{dump_scores[-1][i]}")
        file = open(dump_scores[-1][i], 'rb')
        scores_oi = pickle.load(file)
        data_o.append([scores_oi[t] for t in range(0, timeHorizon, skip)])
        file.close()
    scores_oracle = np.mean(data_o, axis=0)

    for j in range(nbAlgs):
        data_j = []
        for i in range(len(dump_scores[j])):
            #print(f"{dump_scores[j][i]}")
            file = open(dump_scores[j][i], 'rb')
            scores_ij = pickle.load(file)
            data_j.append([scores_oracle[t] - scores_ij[t] for t in range(0,timeHorizon,skip)])
            file.close()

        filename = root_folder+"regret_" + envName + "_" + names[j] + "_" + str(timeHorizon) + "_" + str(
            j) + "_" + str(
            time.time())
        file = open(filename, 'wb')
        pickle.dump(data_j, file)
        file.close()

        mean.append(np.mean(data_j, axis=0))
        median.append(np.quantile(data_j, 0.5, axis=0))
        quantile1.append(np.quantile(data_j, 0.25, axis=0))
        quantile2.append(np.quantile(data_j, 0.75, axis=0))

    return mean,median,quantile1,quantile2,times

