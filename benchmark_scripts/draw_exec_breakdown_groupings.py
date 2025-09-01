#!/usr/bin/env python3

from fig_common import *

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import numpy as np

from overlap_calc import get_breakdown_stats, get_breakdown_stats_simple, get_breakdown_stats_percentage

import argparse

BAR_WIDTH = 1.
BAR_SPACING = 0.25
GROUP_SPACING = 1 # TODO use this?

BASE_CORE_COUNT=1472

batch_size = None
num_cores = None
kb = None
seq_length = None

# hbm_bw_cases = [i for i in range(2000, 10000+1, 1000)]
# hbm_bw_cases = [i for i in range(2000, 16000+1, 2000)]
hbm_bw_cases = [i for i in range(6000, 16000+1, 2000)]
bar_hatches: list = ['//', '\\\\', '..', '', '+', 'x', 'o', 'O', '.', '*']
colors = ["#ec6632", "#126b91", "#93bc38", "#252422", "#c45ab3"]*2

model = None

def plot_bw(c: int, ax: plt.Axes, data, num_bars_per_group, ylabel: bool = False):
    # Load all data
    weights = {"Overlapped Preload & Execute": [d[2] for d in data],
                "Preload": [d[0] for d in data], "Execute": [d[1] for d in data],
                "Interconnect Contention": [d[3] for d in data]}
    # weights = {"overlapped": [(d[2]) for d in data], "load only": [(d[0]) for d in data],
    #             "compute_only": [(d[1]) for d in data]}

    bottom = np.zeros(len(weights["Overlapped Preload & Execute"]))

    # width = 0.25*(len(weights["overlapped"])+1) + 0.05*((len(weights["overlapped"])+1)-1)
    # num_bars_per_group = len(weights["overlapped"])
    # num_bars_per_group = len(hbm_bw_cases)

    group_width = BAR_WIDTH*num_bars_per_group + BAR_SPACING*(num_bars_per_group-1)
    multiplier = 0

    rects = []

    for i, (label, value) in enumerate(weights.items()):
        offset = (group_width + GROUP_SPACING)*c - GROUP_SPACING
        rects.append(ax.bar(x=[offset + (BAR_WIDTH+BAR_SPACING)*i for i in range(len(weights["Overlapped Preload & Execute"]))], height=value, label=label, \
                       bottom=bottom, width=BAR_WIDTH, color=colors[i], hatch=bar_hatches[i], edgecolor='black')
        )

        for bc in rects[-1]:
            bc._hatch_color = mpl.colors.to_rgba('white') 
        bottom += value
        multiplier += 1

    # for i in range(min(num_bars_per_group, len(weights["overlapped"]))):
    #     # ax.text(offset + 0.3*i - BAR_WIDTH/2, bottom[i], "  " + str(int(bottom[i])), rotation='vertical', size='x-small')
    #     ax.text(offset + 0.3*i - BAR_WIDTH/2, bottom[i], "  {:.4}".format(float(bottom[i])), rotation='vertical', size='x-small')

    # ax.set_ylim(0, 100)
    ax.set_ylabel("Latency (ms)", fontsize=13)

    # ax.set_xlabel("HBM BW (TBps)", fontsize=22)

    if c == 0:
        legend = ax.legend(
            [rects[0],rects[3],rects[1],rects[2]],
            ["Overlapped Preload & Execute", "Interconnect Contention","Preload", "Execute"],
            ncol=2, fontsize=12, columnspacing=0.5, loc='upper right',\
                  handlelength=0.7, handletextpad=0.35, borderaxespad=0, frameon=False, fancybox=False, bbox_to_anchor=(1, 1-0.005))

        # frame = legend.get_frame()
        # frame.set_linewidth(0)


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='Draw Bandwith Figures')
    parser.add_argument("--model", type=str, default="llama2-13")
    parser.add_argument("--num_cores", type=int, default=5888)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--kb", type=int, default=624)

    args = parser.parse_args()
    model = args.model

    num_cores = args.num_cores
    seq_length = args.seq_length
    kb = args.kb
    batch_size = args.batch_size

    stats = {"Naive": [], "Baseline": [], "ICBM Ordered": [], "ICBM": [], "Ideal": []} # needed for ordering, the rest will just get populated
    for bw in hbm_bw_cases:
        timing_file = f"benchmark_scripts/outputs/{model}/test-bw-{batch_size}-{seq_length}-{bw}-{num_cores}-{kb}-0.0-1.0.timing"
        # more_stats = get_breakdown_stats(timing_file, model, batch_size, num_cores, seq_length, bw)
        # more_stats = get_breakdown_stats_simple(timing_file, model, batch_size, num_cores, seq_length, bw)
        more_stats = get_breakdown_stats_percentage(timing_file, model, batch_size, num_cores, seq_length, bw, mod=True)
        # more_stats = get_breakdown_stats(timing_file)
        for data in more_stats:
            if data in stats:
                stats[data] += more_stats[data]
            else:
                stats[data] = more_stats[data]

    stats["Baseline"] = [(a if sum(a) < sum(b) else b) for a, b in zip(stats["Max Cold"], stats["Min Cold"])]
    stats = {impl[label]: stats[label] for label in impl}

    # for GROUP_BY_BW in [False, True]:
    for GROUP_BY_BW in [False]:
        num_bars_per_group = len(stats[impl["Naive"]])
        # num_bars_per_group = len(hbm_bw_cases)
        if GROUP_BY_BW:
            bw_stats = {}
            for i, bw in enumerate(hbm_bw_cases):
                bw_stats[bw] = []
                for key in stats.keys():
                    try:
                        bw_stats[bw].append(stats[key][i])
                    except:
                        bw_stats[bw].append((0, 0, 0))

            stats = bw_stats
            num_bars_per_group = len(bw_stats[hbm_bw_cases[0]])

        num_groups = len(stats)

        group_width = BAR_WIDTH*num_bars_per_group + BAR_SPACING*(num_bars_per_group-1)

        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('hatch', color='white')

        fig, ax = plt.subplots()
        fig.set_figheight(1.75)
        fig.set_figwidth(7)
        ax.set_ylabel("Latency (ms)", fontsize=13)


        for i, data in enumerate(stats.keys()):
            ylabel = (i == 0)
            plot_bw(i, ax, data=stats[data], num_bars_per_group=num_bars_per_group, ylabel=ylabel)


        comp_overhead = sum([full[1]/fixed[1] for full, fixed in zip(stats["ELK-Full"], stats["ELK-Dyn"])]) / len(stats["ELK-Full"])
        print(f"comp_overhead: {comp_overhead}")

        xticks = [(BAR_WIDTH+BAR_SPACING)*i for i in range(len(hbm_bw_cases))]
        xticks = [[x+(group_width + GROUP_SPACING)*i-GROUP_SPACING for x in xticks] for i in range(num_groups)]
        ticks = []
        for group in xticks:
            ticks.extend(group)

        # ax.set_xticks(ticks, [str(bw) + "GBps" for bw in hbm_bw_cases]*num_groups, rotation=90, ha='right')
        ax.set_xticks(ticks, [str(bw // 1000) for bw in hbm_bw_cases]*num_groups, rotation=45, ha='center', fontsize=10, position=(0,0.035))
        # TODO rerun so it's real TB

        # left_lim = (0 - BAR_WIDTH/2 - BAR_SPACING)
        # right_lim = (group_width*num_groups + GROUP_SPACING*(num_groups-1))

        # ax.set_yticks([i for i in range(0, 20, 10)])
        # ax.set_yticklabels([str(i) for i in range(0, 20, 10)])
        

        left_lim, right_lim = ax.get_xlim()
        ax.set_xlim(left=left_lim+GROUP_SPACING, right=right_lim-GROUP_SPACING)
        down_lim, up_lim = ax.get_ylim()
        ax.set_ylim(bottom=down_lim, top=1.2*up_lim)


        left_lim, right_lim = ax.get_xlim()

        # scale_factor = (1 / (right_lim - left_lim)) # * (0.98-0.045)
        scale_factor = (0.99-0.08) / (right_lim - left_lim)
        labels = [group_width*i+GROUP_SPACING*i for i in range(len(stats))]
        for l, k in zip(labels, stats.keys()):
            print("l:", l)
            xcoord = (l + (group_width / 2) + (0-left_lim))*scale_factor + scale_factor*2
            print("xc",xcoord)
            # TODO normalize this
            # xticks.append(t+0.3)
            label_text = "ICBM Full" if (k == "ICBM") else k
            fig.text(xcoord, 0.109, label_text, ha='center', fontsize=12)
            if label_text == "ELK-Dyn":
                fig.text(xcoord, 0.00, "HBM BW (TB/s)", ha='center', fontsize=12)
        # if GROUP_BY_BW:
        #     ax.set_xlabel("HBM BW (GB/s)", fontsize=20)

        # ax.xaxis.set_major_locator(LogLocator(10, subs=(1.0,), numticks=8))
        ax.yaxis.set_major_locator(FixedLocator([i for i in range(0, 100+1, 20)]))
        # ax.xaxis.set_minor_locator(LogLocator(10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=8))
        ax.yaxis.set_minor_locator(FixedLocator([i for i in range(0, 100+1, 10)]))

        ax.set_axisbelow(True)
        ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="grey", zorder=1)
        ax.grid(which="minor", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

        plt.subplots_adjust(top=1, bottom=0.335, left=0.08, right=1)
        grouping = "bw_grouped" if GROUP_BY_BW else "impl_grouped"
        description = f"fig20-breakdown-{grouping}-{model}-{num_cores}cores-{kb}kb-b{batch_size}-{seq_length}seq"
        # plt.title(description)
        plt.savefig(f"{description}.pdf", pad_inches=0)
        plt.savefig(f"{description}.png", pad_inches=0)
