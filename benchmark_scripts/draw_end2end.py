#!/usr/bin/env python3

from fig_common import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import numpy as np

from overlap_calc import get_breakdown_stats, get_breakdown_stats_simple, get_breakdown_stats_percentage

import argparse

BAR_WIDTH = 1.
BAR_SPACING = 0.1
GROUP_SPACING = 4. # TODO use this?
SUBGROUP_SPACING = 1

BASE_CORE_COUNT=1472

num_cores = None
kb = None

hbm_bw_cases = [i for i in range(1000, 15000+1, 1000)]

models = ["llama2-13", "gemma2", "opt-30", "llama2-70"]
# models = ["llama2-13", "gemma2", "opt-30"]
batch_sizes = [16, 32, 64] #, 32, 16, 32]
# batch_sizes = [] #, 32, 16, 32]
# seq_lengths = [1024, 2048]
seq_lengths = [2048, 4096]
# seq_lengths = [2048]

stat_labels = ["Naive", "Baseline", "ICBM Ordered", "ICBM", "Ideal", "min_cold", "max_cold"]

# del impl["Ideal"] # toggle showing 'ideal' bar

def min_append(arr: list, data):
    if arr is None:
        return [data]

    offset = 3*(len(arr) // 3)
    for i in range(offset, len(arr)):
        if arr[i] > data:
            arr[i] = data
    arr.append(data)
    return arr

def max_append():
    return

def plot_bw(
        idx: int,
        ax: plt.Axes,
        data: dict,
        label: str,
        num_bars_per_group: int,
        num_impl: int,
        ylabel: bool = False):

    num_subgroups = num_bars_per_group // num_impl
    model_width = BAR_WIDTH*num_bars_per_group + BAR_SPACING*(num_bars_per_group-1) + \
        SUBGROUP_SPACING*num_subgroups + GROUP_SPACING
    model_offset = (BAR_WIDTH + BAR_SPACING)*idx

    for i, model in enumerate(data):
        offset = model_offset + i*model_width
        batch_offset = [offset+((BAR_WIDTH+BAR_SPACING)*num_impl+SUBGROUP_SPACING)*x for x in list(range(len(data[model])))]

        bar_container = ax.bar(x=batch_offset, height=data[model], label=label, \
               width=BAR_WIDTH, color=colors1[idx], hatch=bar_hatches1[idx], edgecolor='black')
        for bc in bar_container:
            bc._hatch_color = mpl.colors.to_rgba('white')

    if ylabel:
        ax.set_ylabel("Latency (ms)", fontsize=17)

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='Draw Bandwith Figures')
    parser.add_argument("--num_cores", type=int, default=5888)
    parser.add_argument("--kb", type=int, default=624)
    parser.add_argument("--bandwidth", type=int, default=16000)

    args = parser.parse_args()

    num_cores = args.num_cores
    kb = args.kb
    bw = args.bandwidth

    num_groups = len(models)
    num_bars_per_group = len(batch_sizes)*len(seq_lengths)*len(impl)
    group_width = BAR_WIDTH*num_bars_per_group + BAR_SPACING*(num_bars_per_group-2)
    stats = {label: {model : [] for model in models} for label in stat_labels + ["base_min_cold_ms", "base_max_cold_ms"]}
    for model in models:
        for seq_length in seq_lengths:
            for batch_size in batch_sizes:
                result_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-{0.0}-{1.0}"
                with open(result_file, 'r') as f:
                    for line in f:
                        if 'base_min_cold_ms:' in line:
                            #stats["min_cold"][model].append(float(line.split(': ')[-1]))
                            stats["min_cold"][model] = min_append(stats["min_cold"][model], float(line.split(': ')[-1]))
                        elif 'base_max_cold_ms:' in line:
                            #stats["max_cold"][model].append(float(line.split(': ')[-1]))
                            stats["max_cold"][model] = min_append(stats["max_cold"][model], float(line.split(': ')[-1]))
                        elif "naive_ms:" in line:
                            #stats["Naive"][model].append(float(line.split(': ')[-1]))
                            stats["Naive"][model] = min_append(stats["Naive"][model], float(line.split(': ')[-1]))
                        elif 'icbm_ordered_ms:' in line:
                            stats["ICBM Ordered"][model].append(float(line.split(': ')[-1]))
                        elif "icbm_ms:" in line:
                            stats["ICBM"][model].append(float(line.split(': ')[-1]))
                        elif 'ideal_ms:' in line:
                            stats["Ideal"][model].append(float(line.split(': ')[-1]))

        stats["Baseline"][model] = \
            [a if a < b else b for a, b in zip(stats["min_cold"][model], stats["max_cold"][model])]

    # for label in ["min_cold", "max_cold"]:
    #     print(f"----- {label} -----")
    #     print(f"{label}: {stats[label]['llama2-13']}")
    stats = {impl[label]: {model: stats[label][model] for model in models} for label in impl}

    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=14)
    plt.rc('hatch', color='white')

    fig, ax = plt.subplots()
    fig.set_figheight(2.2)
    fig.set_figwidth(17)

    for i, data in enumerate(stats.keys()):
        ylabel = (i == 0)
        plot_bw(i, ax, data=stats[data], label=data, num_bars_per_group=num_bars_per_group,
                num_impl=len(impl), ylabel=ylabel)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = plt.legend(by_label.values(), by_label.keys(), ncol=len(impl), fontsize=15, columnspacing=0.5,\
                            loc='upper left', handlelength=0.7, handletextpad=0.35, borderaxespad=0, frameon=True, fancybox=False,
                            bbox_to_anchor=(0.+0.005, 1-0.01))

        frame = legend.get_frame()
        frame.set_linewidth(0)

        speedup_vs = [sum([d / elk for elk, d in zip(stats[impl["ICBM"]][model], stats[data][model])])
                      / len(stats[data][model]) for model in stats[data]]
        print(f"max. speedup vs {data}: {max(speedup_vs)}", end='\t\t\t')
        speedup_vs = sum(speedup_vs) / len(speedup_vs)
        # print(f"avg. speedup vs {data}: {speedup_vs}", end='\t\t\t')


    speedup_vs = [sum([d / elk for elk, d in zip(stats[impl["ICBM Ordered"]][model], stats["Ideal"][model])])
                    / len(stats["Ideal"][model]) for model in stats["Ideal"]]
    speedup_vs = sum(speedup_vs) / len(speedup_vs)
    print(f"Elk-fixed. speedup vs Ideal: {speedup_vs}", end='\t\t\t')

    xticks = [((BAR_WIDTH+BAR_SPACING)*len(impl) + SUBGROUP_SPACING)*i for i in range(len(batch_sizes)*len(seq_lengths))]

    num_bars_per_group = len(batch_sizes)*len(seq_lengths)*len(impl)
    num_subgroups = num_bars_per_group // len(impl)
    subgroup_width = BAR_WIDTH*(len(impl)-1) + BAR_SPACING*(len(impl)-1)
    group_width = BAR_WIDTH*num_bars_per_group + BAR_SPACING*(num_bars_per_group-1) + SUBGROUP_SPACING*num_subgroups
    xticks = [[x+(group_width + GROUP_SPACING)*i + subgroup_width/2 \
               for x in xticks] for i in range(num_groups)]
    ticks = []
    for group in xticks:
        ticks.extend(group)


    xtick_labels = [f"b{str(bs)}" for bs in batch_sizes]
    tick_labels = []
    for seq_length in seq_lengths:
        base_label = xtick_labels[len(xtick_labels) // 2]
        xtick_labels[len(xtick_labels) // 2] += f"\nseq_length: {seq_length}"
        tick_labels.extend(xtick_labels)
        xtick_labels[len(xtick_labels) // 2] = base_label

    ax.set_xticks(ticks, tick_labels*num_groups, ha='center') # TODO

    # left_lim = (0 - BAR_WIDTH/2 - BAR_SPACING)
    # right_lim = (group_width*num_groups + GROUP_SPACING*(num_groups-1))
    # ax.set_xlim(left=left_lim, right=right_lim)


    left, right = ax.get_xlim()
    ax.set_xlim(left=left+1.5*GROUP_SPACING, right=right-1.5*GROUP_SPACING)
    left, right = ax.get_xlim()


    scale_factor = (1 / (right - left)) * (0.98-0.075)
    labels = [group_width*i+GROUP_SPACING*i for i in range(len(models))]
    for l, model in zip(labels, models):
            xcoord = (l + (group_width / 2))*scale_factor + 0.115 - 5*scale_factor
            # TODO normalize this
            # xticks.append(t+0.3)
            # label_text = "ICBM Full" if (k == "ICBM") else k
            fig.text(xcoord, 0.02, modelnames[model], ha='center', fontsize=15)

    # ax.xaxis.set_major_locator(LogLocator(10, subs=(1.0,), numticks=8))
    # ax.yaxis.set_major_locator(FixedLocator([i for i in range(0, 25, 10)]))
    # # ax.xaxis.set_minor_locator(LogLocator(10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=8))
    # ax.yaxis.set_minor_locator(FixedLocator([i for i in range(0, 25, 5)]))

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)
    ax.grid(which="minor", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    plt.subplots_adjust(top=0.95, bottom=0.335, left=0.075, right=0.98)
    description = f"fig17-end2end-{num_cores}cores-{kb}kb"
    # plt.title(description)
    plt.savefig(f"{description}.pdf", pad_inches=0)
    plt.savefig(f"{description}.png", pad_inches=0)
