
import pylab as pl
import sys
from math import exp
from typing import Any
import numpy as np

ROOT= "results/"
def plotScoreDiffs(learnersName: list[str], envName: str, title: str, mean: list[np.ndarray], median: list[np.ndarray], quantile1: list[np.ndarray], quantile2: list[np.ndarray], times: list[int], timeHorizon: int, logfile: Any='', timestamp: Any=0, root_folder: str=ROOT) -> None:
    if (logfile==''):
        logfile=sys.stdout
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = root_folder+"Regrets_"
    #colors= ['black', 'blue','gray', 'green', 'red']#['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
     '#f781bf', '#a65628', '#984ea3',
     '#999999', '#e41a1c', '#dede00']

    style = ['o','v','s','d','<']
    m,M=0,0
    pl.title(title)
    for i in range(len(median)):
        m=min(m,min(median[i]),min(mean[i]))
        M=1.1*max(M,max(median[i]),max(mean[i]))
        pl.fill_between(
            times,
            quantile1[i],
            quantile2[i],
            color=colors[i% len(colors)],
            alpha=0.18,
            linewidth=0
        )
        pl.plot(
            times,
            median[i],
            color=colors[i% len(colors)],
            alpha=0.6,
            linewidth=1.8,
            linestyle='--'
        )
        pl.plot(
            times,
            mean[i],
            style[i % len(style)],
            markevery=0.15,
            markersize=8,
            color=colors[i% len(colors)],
            linewidth=2.3,
            linestyle='-',
            label=learnersName[i]
        )

        #pl.plot(times, mean[i], style[i% len(style)], label=learnersName[i], color=colors[i % len(colors)], linewidth=2.0, linestyle='-.', markevery=0.05)
        #pl.plot(times, median[i], style[i% len(style)], color=colors[i % len(colors)], linewidth=2.0, linestyle='--', markevery=0.05)
        #pl.plot(times,quantile1[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.6)
        #pl.plot(times,quantile2[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.6)

        textfile += learnersName[i] + "_"
        logfile.write(learnersName[i] + ' has regret ' + str(median[i][-1]) + ' after ' + str(timeHorizon) + ' time steps with quantiles ' +
              str(quantile1[i][-1]) +' and '+ str(quantile2[i][-1])+"\n")

    textfile+="_"+str(timeHorizon)+"_"+envName+"_"+timestamp
    pl.legend(loc=2)
    pl.xlabel("Time steps", fontsize=13, fontname = "Arial")
    pl.ylabel("Regret", fontsize=13, fontname = "Arial")
    #pl.xticks(times)
    pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
    pl.ylim([m,M])
    pl.savefig(textfile+'.png')
    pl.savefig(textfile+ '.pdf')
    # pl.xscale('log')
    # pl.savefig(textfile + '_xlog.png')
    # pl.savefig(textfile + '_xlog.pdf')
    # pl.ylim(1)
    if(timeHorizon>10):
        pl.xlim(10,timeHorizon)
    pl.xscale('linear')
    pl.yscale('log')
    pl.ylim([max(m,1e-0),max(M,2e-0)])
    pl.savefig(textfile + '_ylog.png')
    pl.savefig(textfile + '_ylog.pdf')
    # pl.xscale('log')
    # pl.savefig(textfile + '_loglog.png')
    # pl.savefig(textfile + '_loglog.pdf')
    logfile.write("\nPlots are depicted in files "+textfile + ".pdf/png, etc.")