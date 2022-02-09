import re
import argparse
import numpy as np

def parse_output(filename, num_nodes, num_ranks):
    pattern_load = re.compile("Loaded in \d+\.\d+s")
    pattern_merge = re.compile("AC recovered in \d+\.\d+s")
    pattern_phase = re.compile("Problem phased in \d+\.\d+s")
    pattern_slice = re.compile("slice=\d+\.\d+s")
    pattern_slice_oh = re.compile("slice_oh=\d+\.\d+s")
    pattern_match = re.compile("match=\d+\.\d+s")
    pattern_match_oh = re.compile("match_oh=\d+\.\d+s")
    pattern_completed = re.compile("completed in \d+\.\d+s")

    merge = []
    phase = []
    slices = []
    slice_oh = []
    ori_match = []
    ori_match_oh = []
    completed = []

    for i, line in enumerate(open(filename)):
        for match in re.findall(pattern_load, line):
            loading_time = float(re.findall("\d+\.\d+", match)[0])
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
        for match in re.findall(pattern_completed, line):
            completed_time = float(re.findall("\d+\.\d+", match)[0])

    # merging & phasing are performed independently by each rank
    # monitor the longest time
    merging_max, merging_mean, merging_std = np.max(merge), np.mean(merge), np.std(merge)
    phasing_max, phasing_mean, phasing_std = np.max(phase), np.mean(phase), np.std(phase)
    # slicing & orientation matching perform a subset of the task
    tot_ranks = int(num_nodes * num_ranks)
    num_gen = int(len(slices) / tot_ranks)
    num_times = tot_ranks * num_gen
    slicing_max, slicing_mean, slicing_std = np.max(slice_oh)+np.max(slices), \
                                             (np.sum(slice_oh) + np.sum(slices))/num_times, \
                                             (np.std(slice_oh) + np.std(slices))/num_times
    ori_match_max, ori_match_mean, ori_match_std = np.max(ori_match_oh)+np.max(ori_match), \
                                             (np.sum(ori_match_oh) + np.sum(ori_match))/num_times,\
                                             (np.std(ori_match_oh) + np.std(ori_match))/num_times

    print("Max/mean/std time per generation in seconds")
    print(f"Loading time: {loading_time:.3f}")
    print(f"Phasing time: {phasing_max:.3f}/{phasing_mean:.3f}/{phasing_std:.3f}")
    print(f"Merging time: {merging_max:.3f}/{merging_mean:.3f}/{merging_std:.3f}")
    print(f"Slicing time: {slicing_max:.3f}/{slicing_mean:.3f}/{slicing_std:.3f}")
    print(f"Orientation matching time: {ori_match_max:.3f}/{ori_match_mean:.3f}/{ori_match_std:.3f}")
    print(f"Total time for {num_gen} generations: {completed_time:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments to parseOutput')
    #positional arguments
    parser.add_argument('--fname', help='name of output file', type=str)
    parser.add_argument('--nodes', help='number of nodes', type=int)
    parser.add_argument('--ranks', help='number of ranks per node', type=int)
    args = parser.parse_args()
    parse_output(args.fname, args.nodes, args.ranks)

