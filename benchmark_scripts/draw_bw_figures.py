#!/usr/bin/env python3

from fig_common import *

import matplotlib.pyplot as plt
import numpy as np

import argparse

BASE_CORE_COUNT=1472

batch_size = None
kb = None
seq_length = None
num_cores=None

# hbm_bws = [i for i in range(1000, 5000, 500)] + [i for i in range(5000, 14000, 2000)]
# hbm_bws = [i for i in range(1000, 18000+1, 1000)]
hbm_bws = [i for i in range(4000, 16000+1, 2000)]
# hbm_bws = [i for i in range(100, 200+1, 10)]
# hbm_bws = hbm_bws[2:6] # TODO remove
models = ["llama2-13", "gemma2", "opt-30", "llama2-70"]
# models = ["opt-30"]*3

def plot_bw(model: str, ax: plt.Axes, ylabel: bool = False, no_x_ticks: bool = False):
    # Load all data
    exe_times = {'min_cold':[], 'max_cold':[], 'naive':[], 'baseline':[], 'icbm_ordered':[], 'icbm':[], 'ideal':[]}
    for bw in hbm_bws:
        if no_x_ticks:
            output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}"
        else:
            output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{(16000/bw)}-{1.0}"

        count = 0
        for line in open(output_file, 'r'):
            if 'base_min_cold_ms:' in line:
                exe_times['min_cold'].append(float(line.split(': ')[-1]))
                count += 1
            elif 'base_max_cold_ms:' in line:
                exe_times['max_cold'].append(float(line.split(': ')[-1]))
                count += 1
            elif 'naive_ms:' in line:
                exe_times['naive'].append(float(line.split(': ')[-1]))
                count += 1
            elif 'icbm_ordered_ms:' in line:
                exe_times['icbm_ordered'].append(float(line.split(': ')[-1]))
                count += 1
            elif 'icbm_ms:' in line:
                exe_times['icbm'].append(float(line.split(': ')[-1]))
                count += 1
            elif 'ideal_ms:' in line:
                exe_times['ideal'].append(float(line.split(': ')[-1]))
                count += 1


        print(output_file)
        for label in exe_times:
            print(f"{label}: {len(exe_times[label])}")

        assert count == 6, "missing some data"

        # do we still have any failing cases? idrk
        # if not count:
        #     exe_times['min_cold'].append(float('nan'))
        #     exe_times['max_cold'].append(float('nan'))
        #     exe_times['naive'].append(float('nan'))
        #     exe_times['icbm_ordered'].append(float('nan'))
        #     exe_times['icbm'].append(float('nan'))
        #     exe_times['ideal'].append(float('nan'))

    exe_times['baseline'] = [a if a < b else b for a, b in zip(exe_times['min_cold'], exe_times['max_cold'])]
    # exe_times['naive_a'] = [a if a > b else b for a, b in zip(exe_times['min_cold'], exe_times['max_cold'])]
    del exe_times['min_cold']
    del exe_times['max_cold']

    for key in exe_times:
        for i in range(1, len(exe_times[key])):
            if exe_times[key][i] > exe_times[key][i-1]:
                exe_times[key][i] = exe_times[key][i-1]

    # print(len(hbm_bws))
    # print(len(exe_times['icbm']))
    # ax.plot(hbm_bws, exe_times['min_cold'], '.-', label='min_cold', color=colors[0])
    hbm_bws_tb = [b / 1024 for b in hbm_bws]
    print(len(hbm_bws_tb))
    ax.plot(hbm_bws_tb, exe_times['naive'], ':', label='Basic', color=colors[1], marker=markers[1], linewidth=2, markersize=7)
    ax.plot(hbm_bws_tb, exe_times['baseline'], '--', label='Static', color=colors[0], marker=markers[0], linewidth=2, markersize=7)
    ax.plot(hbm_bws_tb, exe_times['icbm_ordered'], '-.', label='ELK Ordered', color=colors[2], marker=markers[2], linewidth=2, markersize=7)
    ax.plot(hbm_bws_tb, exe_times['icbm'], '-', label='ELK Full', color=colors[3], marker=markers[3], linewidth=2, markersize=7)
    ax.plot(hbm_bws_tb, exe_times['ideal'], '--', label='Ideal', color=colors[4], marker=markers[4], zorder=0, linewidth=2, markersize=7)

    # ax.set_yscale("log")
    # ax.set_xlabel(f"Bandwidth (GBps)\ncores = {c}", fontsize=20)
    if not no_x_ticks:
        ax.set_xlabel(f"{modelnames[model]}", fontsize=25)

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)
    # ax.set_yticklabels(np.array(ax.get_yticks()).astype(int), position=(0.03, 0))
    # ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), position=(0, 0.03))
    if model in ["llama2-13", "gemma2"]:
        ax.set_ylim([5, 27])
    elif model in ["opt-30", "llama2-70"]:
        ax.set_ylim([10, 50])
    # ax.set_ylim([5, 50])

    if ylabel:
        if no_x_ticks:
            ax.set_ylabel("All-to-All", fontsize=25)
        else:
            ax.set_ylabel("Mesh", fontsize=25)
    # else:
    #     ax.set_yticklabels([])

    if no_x_ticks:
        ax.set_xticklabels([])
    ax.yaxis.set_ticks_position('none') 
    ax.tick_params(axis='y', which='major', pad=-1)



if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='Draw Bandwith Figures')
    parser.add_argument("--num_cores", type=str, default="5888")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--kb", type=int, default=624)

    args = parser.parse_args()

    batch_size = args.batch_size
    seq_length = args.seq_length
    kb = args.kb
    num_cores = args.num_cores

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    fig, ax = plt.subplots(2, len(models), figsize=(15, 5))
    plt.subplots_adjust(top=0.9, bottom=0.25, left=0.095, right=0.99)
    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    fig.text(0.5224, 0.025, 'HBM BW (TB/s)', ha='center', fontsize=27)
    fig.text(0.02, 0.399, 'Latency (ms)', ha='center', fontsize=26, rotation=90)

    for i, model in enumerate(models):
        ylabel = (i == 0)
        plot_bw(model=model, ax=ax[0][i], ylabel=ylabel, no_x_ticks=True)

    for i, model in enumerate(models):
        ylabel = (i == 0)
        plot_bw(model=model, ax=ax[1][i], ylabel=ylabel)

    ax[0][-1].legend(loc="upper right", fontsize=25, ncol=5, frameon=False,
                  handlelength=1, handletextpad=0.35, borderaxespad=0, labelspacing=0.5,
                  bbox_to_anchor=(1,1.35))

    # plt.legend(loc="upper right")
    description = f"{num_cores}cores-{kb}kb-b{batch_size}-{seq_length}seq"
    # plt.title(description)
    plt.savefig(f"fig19-bw-{description}.png")
    plt.savefig(f"fig19-bw-{description}.pdf")
    # plt.savefig("pretty.png", pad_inches=0)
