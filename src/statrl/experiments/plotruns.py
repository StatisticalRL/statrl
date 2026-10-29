
import pylab as pl
import sys
from typing import Any
import numpy as np

ROOT= "results/"
def plotScoreDiffs(learnersName: list[str], envName: str, title, mean: list[np.ndarray], median: list[np.ndarray], quantile1: list[np.ndarray], quantile2: list[np.ndarray],quantile3: list[np.ndarray],quantile4: list[np.ndarray], times: list[int], timeHorizon: int, logfile: Any='', timestamp: Any=0, root_folder: str=ROOT) -> None:
    """Draw the regret figures and record final regrets in the logfile.

    Each agent gets a mean curve with markers, a dashed median, and two nested
    shaded bands (0.1-0.9 and 0.25-0.75).  

    Parameters
    ----------
    learnersName : list of str
        Agent names, used as legend labels and in the output filename.
    envName : str
        Environment name, used in the output filename.
    title : tuple of (str, str, str)
        ``(figure_title, xlabel, ylabel)``. The two labels come from
        :attr:`~statrl.experiments.onerun.Interaction.plotlabels`.
    mean, median : list of ndarray
        Per-agent mean and median regret over time.
    quantile1, quantile2, quantile3, quantile4 : list of ndarray
        Per-agent regret quantiles at levels 0.1, 0.25, 0.75, and 0.9.
    times : list of int
        Sampled time steps, shared by every series.
    timeHorizon : int
        Horizon of the runs, used for the x-limit and in the filename.
    logfile : file-like, default=''
        Where final regrets are written. The default empty string selects
        :data:`sys.stdout`.
    timestamp : str, default=0
        Suffix making the filename unique across runs.
    root_folder : str, default='results/'
        Output directory. Must end with a separator.

    Returns
    -------
    None
        Writes ``Regrets_<agents>_<horizon>_<env>_<timestamp>`` as ``.png``
        and ``.pdf``, plus an ``_ylog`` pair with a logarithmic ``y`` axis.

    Notes
    -----
    Colours and markers cycle after 9 and 5 agents respectively, so beyond
    that two agents become hard to tell apart.
    """
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

    fig_title,fig_xlabel,fig_ylabel=title

    ax.set_title(fig_title)


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

    ax.set_xlabel(fig_xlabel, fontsize=13, fontname = "Arial")
    ax.set_ylabel(fig_ylabel, fontsize=13, fontname = "Arial")

    ax.set_xlim(0, times[-1])
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
    fig.savefig(root_folder+"Regrets" + str(timeHorizon)+"_"+envName+'.pdf')
    # pl.xscale('log')
    # pl.savefig(textfile + '_loglog.png')
    # pl.savefig(textfile + '_loglog.pdf')
    logfile.write("\nPlots are depicted in files "+textfile + ".pdf/png, etc.")