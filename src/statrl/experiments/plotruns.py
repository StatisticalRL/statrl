
import pylab as pl
import sys
from typing import Any
import numpy as np

ROOT= "results/"
def plotScoreDiffs(learnersName: list[str], envName: str, title: str, mean: list[np.ndarray], median: list[np.ndarray], quantile1: list[np.ndarray], quantile2: list[np.ndarray],quantile3: list[np.ndarray],quantile4: list[np.ndarray], times: list[int], timeHorizon: int, logfile: Any='', timestamp: Any=0, root_folder: str=ROOT) -> None:
    if (logfile==''):
        logfile=sys.stdout
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    fig, ax = pl.subplots(layout="constrained")
    textfile = root_folder+"Regrets_"
    #colors= ['black', 'blue','gray', 'green', 'red']#['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
     '#f781bf', '#a65628', '#984ea3',
     '#999999', '#e41a1c', '#dede00']

    style = ['o','v','s','d','<']
    m,M=0,0

    ax.set_title(title)
    for i in range(len(median)):
        m=min(m,min(quantile1[i]),min(mean[i]))
        M=1.1*max(M,max(quantile4[i]),max(mean[i]))
        ax.fill_between(
            times,
            quantile1[i],
            quantile4[i],
            color=colors[i% len(colors)],
            alpha=0.18,
            linewidth=0
        )
        ax.fill_between(
            times,
            quantile2[i],
            quantile3[i],
            color=colors[i % len(colors)],
            alpha=0.18,
            linewidth=0
        )
        ax.plot(
            times,
            median[i],
            color=colors[i% len(colors)],
            alpha=0.6,
            linewidth=1.8,
            linestyle='--'
        )
        ax.plot(
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
    #fig.tight_layout()
    ax.legend(loc=2)
    ax.set_xlabel("Time steps", fontsize=13, fontname = "Arial")
    ax.set_ylabel("Regret", fontsize=13, fontname = "Arial")
    ax.set_xlim(0,min(timeHorizon,len(mean[0]))-1)
    #pl.xticks(times)
    ax.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
    ax.set_ylim([m,M])
    fig.savefig(textfile+'.png')
    fig.savefig(textfile+ '.pdf')
    # pl.xscale('log')
    # pl.savefig(textfile + '_xlog.png')
    # pl.savefig(textfile + '_xlog.pdf')
    # pl.ylim(1)
    #if(timeHorizon>10):
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_ylim([max(m,1e-0),max(M,2e-0)])
    fig.savefig(textfile + '_ylog.png')
    fig.savefig(textfile + '_ylog.pdf')
    # pl.xscale('log')
    # pl.savefig(textfile + '_loglog.png')
    # pl.savefig(textfile + '_loglog.pdf')
    logfile.write("\nPlots are depicted in files "+textfile + ".pdf/png, etc.")