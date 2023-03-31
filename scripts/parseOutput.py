import re
import argparse
import numpy as np

def parse_output(filename, num_nodes, num_ranks):
    pattern_init = re.compile("Initialized in \d+\.\d+s")
    pattern_load = re.compile("Loaded in \d+\.\d+s")
    pattern_prep = re.compile("Images prepared in \d+\.\d+s")
    pattern_nufft = re.compile("Initialized NUFFT in \d+\.\d+s")
    pattern_merge = re.compile("AC recovered in \d+\.\d+s")
    pattern_phase = re.compile("Problem phased in \d+\.\d+s")
    pattern_slice = re.compile("slice=\d+\.\d+s")
    pattern_slice_oh = re.compile("slice_oh=\d+\.\d+s")
    pattern_match = re.compile("match=\d+\.\d+s")
    pattern_match_oh = re.compile("match_oh=\d+\.\d+s")
    pattern_conv = re.compile("Check convergence done in \d+\.\d+s")
    pattern_free = re.compile("Free memory done in \d+\.\d+s")
    pattern_completed = re.compile("completed in \d+\.\d+s")

    init = []
    load = []
    prep = []
    nufft = []
    merge = []
    phase = []
    slices = []
    slice_oh = []
    ori_match = []
    ori_match_oh = []
    conv = []
    free = []
    completed = []

    for i, line in enumerate(open(filename)):
        for match in re.findall(pattern_init, line):
            init.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_load, line):
            load.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_prep, line):
            prep.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_nufft, line):
            nufft.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_merge, line):
            merge.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_phase, line):
            phase.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_slice, line):
            slices.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_slice_oh, line):
            slice_oh.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_match, line):
            ori_match.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_match_oh, line):
            ori_match_oh.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_conv, line):
            conv.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_free, line):
            free.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_completed, line):
            completed_time = float(re.findall("\d+\.\d+", match)[0])

    # merging & phasing are performed independently by each rank
    # monitor the longest time
    init_max, init_min, init_mean, init_std = np.max(init), np.min(init), np.mean(init), np.std(init)
    load_max, load_min, load_mean, load_std = np.max(load), np.min(load), np.mean(load), np.std(load)
    prep_max, prep_min, prep_mean, prep_std = np.max(prep), np.min(prep), np.mean(prep), np.std(prep)
    nufft_max, nufft_min, nufft_mean, nufft_std = np.max(nufft), np.min(nufft), np.mean(nufft), np.std(nufft)
    merging_max, merging_min, merging_mean, merging_std = np.max(merge), np.min(merge), np.mean(merge), np.std(merge)
    phasing_max, phasing_min, phasing_mean, phasing_std = np.max(phase), np.min(phase), np.mean(phase), np.std(phase)
    conv_max, conv_min, conv_mean, conv_std = np.max(conv), np.min(conv), np.mean(conv), np.std(conv)
    free_max, free_min, free_mean, free_std = np.max(free), np.min(free), np.mean(free), np.std(free)
    # slicing & orientation matching perform a subset of the task
    tot_ranks = int(num_nodes * num_ranks)
    num_gen = int(len(slices) / tot_ranks)
    num_times = tot_ranks * num_gen
    slicing_max, slicing_min, slicing_mean, slicing_std = np.max(slice_oh)+np.max(slices), \
                                             np.min(slice_oh)+np.min(slices), \
                                             (np.sum(slice_oh) + np.sum(slices))/num_times, \
                                             (np.std(slice_oh) + np.std(slices))/num_times
    ori_match_max, ori_match_min, ori_match_mean, ori_match_std = np.max(ori_match_oh)+np.max(ori_match), \
                                             np.min(ori_match_oh)+np.min(ori_match), \
                                             (np.sum(ori_match_oh) + np.sum(ori_match))/num_times,\
                                             (np.std(ori_match_oh) + np.std(ori_match))/num_times

    sum_manual = sum(init)/tot_ranks +                          \
            sum(load)/tot_ranks +                               \
            sum(prep)/tot_ranks +                               \
            sum(nufft)/tot_ranks +                              \
            ((sum(slice_oh)+sum(slices))/tot_ranks) +           \
            ((sum(ori_match_oh)+sum(ori_match))/tot_ranks) +    \
            sum(merge)/tot_ranks +                              \
            sum(phase)/tot_ranks +                              \
            sum(conv)/tot_ranks +                               \
            sum(free)/tot_ranks             
    print(f"Max/min/mean/std time per generation and total time for {num_gen} generations in seconds")
    print(f"Init time: {init_max:.3f}/{init_min:.3f}/{init_mean:.3f}/{init_std:.3f}/{sum(init)/tot_ranks:.3f}")
    print(f"Loading time: {load_max:.3f}/{load_min:.3f}/{load_mean:.3f}/{load_std:.3f}/{sum(load)/tot_ranks:.3f}")
    print(f"Prep time: {prep_max:.3f}/{prep_min:.3f}/{prep_mean:.3f}/{prep_std:.3f}/{sum(prep)/tot_ranks:.3f}")
    print(f"Initialize NUFFT time: {nufft_max:.3f}/{nufft_min:.3f}/{nufft_mean:.3f}/{nufft_std:.3f}/{sum(nufft)/tot_ranks:.3f}")
    print(f"Slicing time: {slicing_max:.3f}/{slicing_min:.3f}/{slicing_mean:.3f}/{slicing_std:.3f}/{(sum(slice_oh)+sum(slices))/tot_ranks:.3f}")
    print(f"Orientation matching time: {ori_match_max:.3f}/{ori_match_min:.3f}/{ori_match_mean:.3f}/{ori_match_std:.3f}/{(sum(ori_match_oh)+sum(ori_match))/tot_ranks:.3f}")
    print(f"Merging time: {merging_max:.3f}/{merging_min:.3f}/{merging_mean:.3f}/{merging_std:.3f}/{sum(merge)/tot_ranks:.3f}")
    print(f"Phasing time: {phasing_max:.3f}/{phasing_min:.3f}/{phasing_mean:.3f}/{phasing_std:.3f}/{sum(phase)/tot_ranks:.3f}")
    print(f"Convergence time: {conv_max:.3f}/{conv_min:.3f}/{conv_mean:.3f}/{conv_std:.3f}/{sum(conv)/tot_ranks:.3f}")
    print(f"Free gpu memory time: {free_max:.3f}/{free_min:.3f}/{free_mean:.3f}/{free_std:.3f}/{sum(free)/tot_ranks:.3f}")
    print(f"Total time from start to finish for {num_gen} generations: {completed_time:.3f}/ sum above: {sum_manual:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments to parseOutput')
    #positional arguments
    parser.add_argument('--fname', help='name of output file', type=str)
    parser.add_argument('--nodes', help='number of nodes', type=int)
    parser.add_argument('--ranks', help='number of ranks per node', type=int)
    args = parser.parse_args()
    parse_output(args.fname, args.nodes, args.ranks)
