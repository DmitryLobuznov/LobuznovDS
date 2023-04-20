"""Plots configuration to matplotlib figures.
"""
import matplotlib.pyplot as plt


# Base constants
TITLESIZE = 16
LABELSIZE = 16
LEGENDSIZE = 16
XTICKSIZE = YTICKSIZE = 16

# the relative size of legend markers vs. original
plt.rcParams['legend.markerscale'] = 1.5
plt.rcParams['legend.handletextpad'] = 0.5
# the vertical space between the legend entries in fraction of fontsize
plt.rcParams['legend.labelspacing'] = 0.4
# border whitespace in fontsize units
plt.rcParams['legend.borderpad'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = LABELSIZE
plt.rcParams['axes.titlesize'] = TITLESIZE
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120

# Grid lines
#plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5

plt.rc('xtick', labelsize=XTICKSIZE)
plt.rc('ytick', labelsize=YTICKSIZE)
plt.rc('legend', fontsize=LEGENDSIZE)
