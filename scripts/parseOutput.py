import re
import argparse

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
    slice = []
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
            slice.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_slice_oh, line):
            slice_oh.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_match, line):
            ori_match.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_match_oh, line):
            ori_match_oh.append(float(re.findall("\d+\.\d+", match)[0]))
        for match in re.findall(pattern_completed, line):
            completed.append(float(re.findall("\d+\.\d+", match)[0]))

    merging_time = sum(merge)
    phasing_time = sum(phase)
    slicing_time = (sum(slice_oh) + sum(slice)) / num_nodes / num_ranks
    ori_matching_time = (sum(ori_match) + sum(ori_match_oh)) / num_nodes / num_ranks
    completed_time = sum(completed)

    print("Loading time = ", loading_time)
    print("Phasing time = ", phasing_time)
    print("Merging time = ", merging_time)
    print("Slicing time = ", slicing_time)
    print("Orientation matching time = ", ori_matching_time)
    print("Total time = ", completed_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments to parseOutput')
    #positional arguments
    parser.add_argument('--fname', help='name of output file', type=str)
    parser.add_argument('--nodes', help='number of nodes', type=int)
    parser.add_argument('--ranks', help='number of ranks per node', type=int)
    args = parser.parse_args()
    parse_output(args.fname, args.nodes, args.ranks)

