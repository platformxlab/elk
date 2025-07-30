#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
out_location = ""

import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams['pdf.fonttype'] = 42

designs = [
    "Min Cold",
    "Max Cold",
    "Naive",
    "Baseline",
    "ICBM Ordered",
    "ICBM",
    "Ideal"
]

impl = {
    "Naive": "Basic",
    "Baseline": "Static",
    "ICBM Ordered": "ELK-Dyn",
    "ICBM": "ELK-Full",
    "Ideal": "Ideal"
    }

modelnames = {
    "llama2-13": "Llama-2-13B",
    "llama2-70": "Llama-2-70B",
    "gemma2": "Gemma-2-27B",
    "opt-30": "OPT-30B",
    "vit-huge": "DiT-XL"
}

# colors = ["brown", "royalblue", "peru", "forestgreen", "gray", "black", "red"] # ['#ff796c', '#a4d46c', '#fca45c', '#95dbd0']
# colors = ["#5c95ff", "#ed9b40", "#393e41", "#548c2f", "#ba3b46"]
# colors = ["#ffb563", "#ba324f", "#175676", "#ff70a2", "#270722"]

# colors = ["#ea591f", "#126b91", "#9bc53d", "#252422", "#c45ab3"]
colors = ["#ec6632", "#126b91", "#93bc38", "#252422", "#c45ab3"]*2
colors1 = ["#126b91", "#ec6632", "#93bc38", "#252422", "#c45ab3"]*2
dark_colors = ["#95340e", "#072836", "#495e1c", "#6a6762", "#873179"]*2
dark_colors1 = ["#072836", "#95340e", "#495e1c", "#6a6762", "#873179"]*2
gradient_colors = ["#6ec7ed", "#49b9e9", "#25abe4", "#1993c8", "#126b91", "#0b435b"]
markers = ["o", "v", "^", "*", ""]
markers1 = ["v", "o", "^", "*", ""]

bar_hatches: list = ['', '\\\\', '..', '//', '+', 'x', 'o', 'O', '.', '*']
bar_hatches1: list = ['\\\\', '', '..', '//', '+', 'x', 'o', 'O', '.', '*']
lines: list = ['-', '-', '-', '-', '--']
lines1: list = [':', '--', '-.', '-', '--']


def IPU_Mk2_cycle_to_ms(cycles) -> float:
    return cycles / 1.325e6

def get_mesh_length_width_ratio(bw_case, noc_bw, c, mesh_calc, default_core_count) -> float:
    if mesh_calc == "bw": 
        mesh_length_width_ratio = 16000/bw_case
    elif mesh_calc == "noc": 
        mesh_length_width_ratio = max((16000/bw_case)**0.5*noc_bw, 1.0)
    elif mesh_calc == "core": 
        mesh_length_width_ratio = (default_core_count/c)**0.5
    return mesh_length_width_ratio
