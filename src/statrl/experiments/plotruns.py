
import pylab as pl
import sys
from typing import Any
import numpy as np

ROOT= "results/"


def _style_axes(ax: Any) -> None:
    """Shared cosmetic styling: light grid, slightly larger tick labels."""
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=11)


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
    pl.title(title, fontsize=15, fontweight='bold')
    _style_axes(pl.gca())
    for i in range(len(mean)):
        m=min(m,min(median[i]),min(mean[i]))
        M=1.1*max(M,max(median[i]),max(mean[i]))
        pl.plot(
            times, median[i], color=colors[i % len(colors)],
            alpha=0.6, linewidth=1.8, linestyle='--',
        )
        draw_regret_curve(
            pl.gca(), learnersName[i], colors[i % len(colors)], style[i % len(style)],
            times, mean[i], quantile1[i], quantile2[i],
        )

        textfile += learnersName[i] + "_"
        logfile.write(learnersName[i] + ' has regret ' + str(median[i][-1]) + ' after ' + str(timeHorizon) + ' time steps with quantiles ' +
              str(quantile1[i][-1]) +' and '+ str(quantile2[i][-1])+"\n")

    textfile+="_"+str(timeHorizon)+"_"+envName+"_"+timestamp
    pl.legend(loc=2)
    pl.xlabel("Time steps", fontsize=14)
    pl.ylabel("Regret", fontsize=14)
    #pl.xticks(times)
    pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
    pl.ylim([m,M])
    pl.savefig(textfile+'.png', bbox_inches='tight', dpi=200)
    pl.savefig(textfile+ '.pdf', bbox_inches='tight', dpi=200)
    # pl.xscale('log')
    # pl.savefig(textfile + '_xlog.png')
    # pl.savefig(textfile + '_xlog.pdf')
    # pl.ylim(1)
    if(timeHorizon>10):
        pl.xlim(10,timeHorizon)
    pl.xscale('linear')
    pl.yscale('log')
    pl.ylim([max(m,1e-0),max(M,2e-0)])
    pl.savefig(textfile + '_ylog.png', bbox_inches='tight', dpi=200)
    pl.savefig(textfile + '_ylog.pdf', bbox_inches='tight', dpi=200)
    # pl.xscale('log')
    # pl.savefig(textfile + '_loglog.png')
    # pl.savefig(textfile + '_loglog.pdf')
    logfile.write("\nPlots are depicted in files "+textfile + ".pdf/png, etc.")


def draw_regret_curve(ax: Any, label: str, color: str, marker: str, times: list[int], mean: np.ndarray, quantile1: np.ndarray, quantile2: np.ndarray) -> None:
    """Shared per-agent drawing recipe: shaded quantile band + marker'd mean."""
    ax.fill_between(times, quantile1, quantile2, color=color, alpha=0.18, linewidth=0)
    ax.plot(times, mean, marker, markevery=0.15, markersize=8, color=color, linewidth=2.3, linestyle='-', label=label)


def plotCumulativeRewards(learnersName: list[str], envName: str, title: str, mean: list[np.ndarray], quantile1: list[np.ndarray], quantile2: list[np.ndarray], times: list[int], timeHorizon: int, logfile: Any = '', timestamp: Any = 0, root_folder: str = ROOT) -> None:
    if (logfile == ''):
        logfile = sys.stdout
    nbFigure = pl.gcf().number + 1
    pl.figure(nbFigure)
    textfile = root_folder + "Rewards_"
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
     '#f781bf', '#a65628', '#984ea3',
     '#999999', '#e41a1c', '#dede00']
    style = ['o', 'v', 's', 'd', '<']
    m, M = 0, 0
    pl.title(title, fontsize=15, fontweight='bold')
    _style_axes(pl.gca())
    for i in range(len(mean)):
        m = min(m, min(mean[i]))
        M = 1.1 * max(M, max(mean[i]))
        draw_regret_curve(
            pl.gca(), learnersName[i], colors[i % len(colors)], style[i % len(style)],
            times, mean[i], quantile1[i], quantile2[i],
        )
        textfile += learnersName[i] + "_"
        logfile.write(learnersName[i] + ' has cumulative reward ' + str(mean[i][-1]) + ' after ' + str(timeHorizon) + ' time steps with quantiles ' +
              str(quantile1[i][-1]) + ' and ' + str(quantile2[i][-1]) + "\n")

    textfile += "_" + str(timeHorizon) + "_" + envName + "_" + timestamp
    pl.legend(loc=2)
    pl.xlabel("Time steps", fontsize=14)
    pl.ylabel("Cumulative reward", fontsize=14)
    pl.ticklabel_format(axis='both', useMathText=True, useOffset=True, style='sci', scilimits=(0, 0))
    pl.ylim([m, M])
    pl.savefig(textfile + '.png', bbox_inches='tight', dpi=200)
    pl.savefig(textfile + '.pdf', bbox_inches='tight', dpi=200)
    logfile.write("\nPlots are depicted in files " + textfile + ".pdf/png, etc.")
