#!/usr/bin/env python3

from fig_common import *

import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

BASE_CORE_COUNT=1472

SKETCHY_OPT = False
AUTO_BATCH_PAD = True

batch_size = None
kb = None
seq_length = None

#models = ["llama2-13", "gemma2", "opt-30", "vit-huge"]
# models = ["llama2-13", "gemma2", "opt-30", "llama2-70"]
# models = ["llama2-13", "gemma2", "opt-30", ""]
# models = ["vit-huge"]*4
#models = ["llama2-13", "gemma2", "opt-30", "vit-huge"]
models = ["llama2-13", "gemma2", "opt-30", "llama2-70", "vit-huge"]

per_core_bw = 4000/1472

vs_naive = []
vs_baseline = []

def plot_bw(model: str, ax: plt.Axes, ylabel: bool = False, mesh: bool = False):
    # Load all data
    # exe_times = {'min_cold':[], 'max_cold':[], 'naive':[], 'baseline':[], 'icbm_ordered':[], 'icbm':[], 'ideal':[]}
    exe_times = {label: [] for label in designs}

    # num_cores = [BASE_CORE_COUNT*i for i in range(1, 4+1)]
    if model in ["llama2-13", "gemma2", "opt-30", "llama2-70"]:
        num_cores = [(1472//2)*i for i in range(3, 8, 1)]
        per_core_bw = 4000/1472
    elif model in ["vit-huge"]:
        num_cores = [(1472//8)*i for i in range(3, 9, 1)]
        per_core_bw = (500//8)/(1472//8)
        per_core_bw = 1000/1472
    else:
        print("Unrecognized Model")
        exit(-1)

    for n in num_cores:
        bw = int(n*per_core_bw)
        if not mesh:
            output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{n}-{kb}-0.0-1.0"
        else:
            output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{n}-{kb}-1.0-1.0"
            if model in ["vit-huge"]:
                output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{n}-{kb}-{(1472/n)**0.5}-1.0"
            else:
                output_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{n}-{kb}-{(5888/n)**0.5}-1.0"

        count = 0
        for line in open(output_file, 'r'):
            if 'base_min_cold_ms:' in line:
                exe_times['Min Cold'].append(float(line.split(': ')[-1]))
                count += 1
            elif 'base_max_cold_ms:' in line:
                exe_times['Max Cold'].append(float(line.split(': ')[-1]))
            elif 'naive_ms:' in line:
                exe_times['Naive'].append(float(line.split(': ')[-1]))
            elif 'icbm_ordered_ms:' in line:
                exe_times['ICBM Ordered'].append(float(line.split(': ')[-1]))
            elif 'icbm_ms:' in line:
                exe_times['ICBM'].append(float(line.split(': ')[-1]))
            elif 'ideal_ms:' in line:
                exe_times['Ideal'].append(float(line.split(': ')[-1]))

        # for label in exe_times:
        #     print(f"{label}: {len(exe_times[label])}")

    exe_times['Baseline'] = [a if a < b else b for a, b in zip(exe_times['Min Cold'], exe_times['Max Cold'])]

    avg_vs_naive = sum([naive / icbm for icbm, naive in zip(exe_times['ICBM'], exe_times['Naive'])]) / len(exe_times['ICBM'])
    avg_vs_baseline = sum([base / icbm for icbm, base in zip(exe_times['ICBM'], exe_times['Baseline'])]) / len(exe_times['ICBM'])

    print(f"---------- {model} ----------")
    print([naive / icbm for icbm, naive in zip(exe_times['ICBM'], exe_times['Naive'])])
    print(f"avg_vs_naive: {avg_vs_naive}")
    print(f"avg_vs_baseline: {avg_vs_baseline}")

    vs_naive.append(avg_vs_naive)
    vs_baseline.append(avg_vs_baseline)

    # percent_ideal = max([1 - (ideal / icbm) for icbm, ideal in zip(exe_times['ICBM'], exe_times['Ideal'])])
    # print(f"percent of ideal: {percent_ideal}")



    # if SKETCHY_OPT: # we don't actually use this
    #     for key in exe_times:
    #         s, e = 0, 1
    #         while e < len(exe_times[key]):
    #             if exe_times[key][s] >= exe_times[key][e]:
    #                 step = (exe_times[key][s] - exe_times[key][e]) / (e - s)
    #                 for i in range(1, e - s):
    #                     exe_times[key][s+i] = exe_times[key][s] - i*step

    #                 s = e
    #                 e = e + 1
    #             else:
    #                 e += 1

    #                 if e == len(exe_times[key]):
    #                     for i in range(1, e - s):
    #                         exe_times[key][s+i] = exe_times[key][s]

    if AUTO_BATCH_PAD:
        for key in exe_times:
            cur_min = exe_times[key][0]
            for i in range(len(exe_times[key])):
                exe_times[key][i] = min(exe_times[key][i], cur_min)
                cur_min = exe_times[key][i]

    exe_times = {impl[label]: exe_times[label] for label in impl}
    for i, label in enumerate(exe_times):
        # print(model, i, label)
        ax.plot(num_cores, exe_times[label], lines1[i], marker=markers1[i], color=colors1[i], label=label)
    # # ax.plot(num_cores, exe_times['min_cold'], '.-', label='min_cold', color=colors[0])
    # # ax.plot(num_cores, exe_times['max_cold'], '.--', label='max_cold', color=colors[1])
    # ax.plot(num_cores, exe_times['naive'], '.:', label='Naive', color=colors[2])
    # ax.plot(num_cores, exe_times['baseline'], '.--', label='Baseline', color=colors[1])
    # ax.plot(num_cores, exe_times['icbm_ordered'], '.-.', label='ICBM Ordered', color=colors[3])
    # ax.plot(num_cores, exe_times['icbm'], '.-', label='ICBM Full', color=colors[4])
    # ax.plot(num_cores, exe_times['ideal'], '.--', label='Ideal', color=colors[5])

    # ax.set_yscale("log")
    # ax.set_ylim(0, 35)

    if mesh:
        ax.set_xlabel(f"{modelnames[model]}", fontsize=23)
    else:
        ax.set_xticklabels("")

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)
    # ax.set_yticklabels(np.array(ax.get_yticks()).astype(int), position=(0.03, 0))
    # ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), position=(0, 0.03))
    # ax.set_ylim([10, 220])

    if model in ["llama2-13", "gemma2", "opt-30", "llama2-70"]:
        ax.xaxis.set_major_locator(FixedLocator([i for i in range(0, 7500, 2500)]))
        ax.xaxis.set_minor_locator(FixedLocator([i for i in range(0, 7500, 500)]))
    elif model in ["vit-huge"]:
        pass
        # ax.xaxis.set_major_locator(FixedLocator([i for i in range(0, 7500, 2500)]))
        # ax.xaxis.set_minor_locator(FixedLocator([i for i in range(0, 7500, 500)]))
    else:
        print("Unrecognized Model")
        exit(-1)

    ax.yaxis.set_ticks_position('none') 
    ax.tick_params(axis='y', which='major', pad=-1)

    if ylabel:
        if mesh:
            ax.set_ylabel("Mesh", fontsize=25)
        else:
            ax.set_ylabel("All-to-All", fontsize=25)
        
    if model in ["llama2-13", "gemma2"]:
        ax.set_ylim([6, 27])
            
    # else:
    #     yticks = ax.get_yticks()
    #     ax.set_yticklabels(['' for t in yticks])

if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='Draw HBM BW Utilization Figures')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--kb", type=int, default=624)
    # parser.add_argument("--bandwidth", type=int, required=True)

    args = parser.parse_args()

    seq_length = args.seq_length
    kb = args.kb
    batch_size = args.batch_size
    # hbm_bw = args.hbm_bw # this is PER CORE -> probably need to rerun expierements for this

    # hbm_bws = [5000, 7000, 10000, 15000]
    # hbm_bws = [3000, 5000, 11000, 13000]

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    fig, ax = plt.subplots(2, len(models), figsize=(15, 4.5))
    plt.subplots_adjust(top=0.9, bottom=0.25, left=0.085, right=0.979)
    # plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(wspace=0.3, hspace=0.15)

    for i, model in enumerate(models):
        ylabel = (i == 0)
        plot_bw(model=model, ax=ax[0][i], ylabel=ylabel, mesh=False)
        plot_bw(model=model, ax=ax[1][i], ylabel=ylabel, mesh=True)
        ylim0 = ax[0][i].get_ylim()
        ylim1 = ax[1][i].get_ylim()
        ylim = (min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1]))
        ax[0][i].set_ylim(ylim)
        ax[1][i].set_ylim(ylim)

    print("---------- AVG ---------")
    print("vs naive", sum(vs_naive) / len(vs_naive))
    print("vs baseline", sum(vs_baseline) / len(vs_baseline))

    ax[0][-1].legend(loc="upper right", fontsize=25, ncol=5, frameon=False,
                  handlelength=1, handletextpad=0.35, borderaxespad=0, labelspacing=0.5,
                  bbox_to_anchor=(1,1.38))
    fig.text(0.5375, 0.02, 'Number of Cores', ha='center', fontsize=27)
    fig.text(0.013, 0.38, 'Latency (ms)', ha='center', fontsize=26, rotation=90)

    description = f"{kb}kb-b{batch_size}-{seq_length}seq"
    # plt.legend(loc="upper right")
    plt.savefig(f"fig23-corelines-{description}.png")
    plt.savefig(f"fig23-corelines-{description}.pdf")
    # plt.savefig("draw-core-lines.png", pad_inches=0)
