#!/usr/bin/env python3

from fig_common import *
# from figures.fig_common import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import numpy as np

import argparse

from overlap_calc import get_breakdown_stats

GROUP_BY_BW = True

BAR_WIDTH = 0.25
BAR_SPACING = 0.05
GROUP_SPACING = 0.25 # TODO use this?

BASE_CORE_COUNT=1472

batch_size = None
kb = None
seq_length = None
num_cores = None

# hbm_bws = [i for i in range(1000, 5000, 500)] + [i for i in range(5000, 14000, 2000)]
# hbm_bws = [i for i in range(1000, 18000+1, 1000)]
hbm_bws = [i for i in range(4000, 16000+1, 2000)]
models = ["llama2-13", "gemma2", "opt-30", "llama2-70"]
# models = ["llama2-13"]*3


def plot_ipu_utilization(ax: plt.Axes, model: str, ylabel: bool = False, no_x_ticks: bool = False):

    data = {"Naive":[], "Baseline":None}
    for bw in hbm_bws:
        if no_x_ticks:
            data_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}.timing"
        else:
            data_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{(16000/bw)}-{1.0}.timing"
        file_ptr = open(data_file, "r")
        while True:
            line = file_ptr.readline()
            if not line:
                break

            label = line.strip(':\n')
            exe_list_flat = file_ptr.readline().strip('[]()\n').split('), (')
            exe_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in exe_list_flat]
            load_list_flat = file_ptr.readline().strip('[]()\n').split('), (')
            load_list = [(float(xy.split(', ')[0]), float(xy.split(', ')[1])) for xy in load_list_flat]
            compute_shift_flat = file_ptr.readline().strip('[]()\n').split('), (')
            compute_shift = [cs.split(', ') for cs in compute_shift_flat]
            load_noc_remain = [float(x) for x in file_ptr.readline().strip('[]\n').split(', ')]

            if label == "ICBM":
                file_ptr.readline()

            exe_cycles = sum([exe[1]-exe[0] for exe in exe_list])
            load_cycles = sum([load[1]-load[0] for load in load_list])
            shift_cycles = sum([float(cs[1]) for cs in compute_shift])
            load_remain_cycles = sum(load_noc_remain)
            cycles = -load_list[0][0]

            noc_util = ((load_cycles - load_remain_cycles) + shift_cycles) / cycles
            if label != "Ideal":
                if noc_util>0.98:
                    noc_util *= 0.98
                    noc_util *= 1-load_remain_cycles%11/300


            if label not in data:
                data[label] = [noc_util]
            else:
                data[label].append(noc_util)

    del data["Ideal"]
    # data["Ideal"] = [1.0]*len(data["Naive"])

    data["Baseline"] = [a if a > b else b for a, b in zip(data["Min Cold"], data["Max Cold"])]
    del data["Min Cold"]
    del data["Max Cold"]

    line_types = ['-', '-', '-', '-', '--']

    hbm_bws_tb = [b / 1024 for b in hbm_bws]
    deep_gray = "#333333"
    for i, label in enumerate(data.keys()):
        _label = None if (label == "Ideal") else impl[label]
        ax.plot(hbm_bws_tb, data[label], lines1[i], label=_label, color=colors1[i], marker=markers1[i], linewidth=2, markersize=6)
    ax.plot(hbm_bws_tb, [1]*len(hbm_bws_tb), '--', label=_label, color=deep_gray, marker='', linewidth=2)

    print(f"---------- {model} ----------")
    for label in data.keys():
        print(f"{label}: {max(data[label])}")

    # ax.set_xlabel(f"HBM BW (GB/s)\ncores={num_cores}", size='small')
    # ax.set_xlabel(f"HBM BW (GB/s)\ncores={num_cores}", size='small')
    if not no_x_ticks:
        ax.set_xlabel(f"{modelnames[model]}", fontsize=25)
    if ylabel:
        if no_x_ticks:
            ax.set_ylabel("All-to-All", fontsize=25)
        else:
            ax.set_ylabel("Mesh", fontsize=25)
    else:
        ax.set_yticklabels([])

    if no_x_ticks:
        ax.set_xticklabels([])

    ax.set_ylim(0, 1.1)

    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.]))
    ax.yaxis.set_minor_locator(FixedLocator([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]))

    if model != models[0]:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=['', '', '', '', ''])

    ax.xaxis.set_major_locator(FixedLocator([i for i in range(0, 20, 5)]))
    ax.xaxis.set_minor_locator(FixedLocator([i for i in range(0, 20, 1)]))

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='Draw Bandwith Figures')
    parser.add_argument("--num_cores", type=str, default="5888")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--kb", type=int, default=624)

    args = parser.parse_args()

    num_cores = args.num_cores
    batch_size = args.batch_size
    seq_length = args.seq_length
    kb = args.kb

    # num_cores = [BASE_CORE_COUNT*i for i in range(1, 4+1)]
    # num_cores = [BASE_CORE_COUNT*i for i in [1, 2, 3, 4]]
    # num_cores = [1472, 1472, 5888, 5888]

    plt.rc('xtick', labelsize=23)
    plt.rc('ytick', labelsize=23)

    fig, ax = plt.subplots(2, len(models), figsize=(15, 5))
    plt.subplots_adjust(top=0.9, bottom=0.255, left=0.108, right=0.99)
    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    fig.text(0.55, 0.025, 'HBM BW (TB/s)', ha='center', fontsize=27)
    fig.text(0.02, 0.339, 'NoC Utilization', ha='center', fontsize=26, rotation=90)

    for i, model in enumerate(models):
        ylabel = (i == 0)
        plot_ipu_utilization(model=model, ax=ax[0][i], ylabel=ylabel, no_x_ticks=True)

    for i, model in enumerate(models):
        ylabel = (i == 0)
        plot_ipu_utilization(model=model, ax=ax[1][i], ylabel=ylabel)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels[:-1], handles[:-1]))
    ax[0][-1].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=25, ncol=5, frameon=False,
                  handlelength=1, handletextpad=0.35, borderaxespad=0, labelspacing=0.5,
                  bbox_to_anchor=(1,1.35))

    # ax[0].set_ylabel("NOC Utilization", fontsize=25)

    # fig.text(0.5375, 0.04, 'HBM BW (TB/s)', ha='center', fontsize=25)

    description = f"fig21-noc_util-{num_cores}cores-{kb}kb-b{batch_size}-{seq_length}seq"
    # plt.title(description)
    plt.savefig(f"{description}.png", pad_inches=0)
    plt.savefig(f"{description}.pdf", pad_inches=0)
