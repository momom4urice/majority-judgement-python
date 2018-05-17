# -- coding: utf-8 --

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click

from collections import OrderedDict
from bidict import bidict
from majorityjudgement import MajorityJudgement
from pprint import pprint
from matplotlib import cm


class Votes(object):
    def __init__(self, name, tally):
        assert (type(tally) is list)
        self.name = name
        self.data = tally
        self.n_votes = sum(self.data)
        self.n_bins = len(self.data)
        self.mj = MajorityJudgement(self.data)

    def histogram(self):
        # Normalize data
        data_sum = float(sum(self.data))
        data_normalized = [float(x) / float(self.n_votes) for x in self.data]

        # Plot histogram
        hist = []
        for i in range(len(self.data)):
            x = self.data[i]
            x_normalized = data_normalized[i]
            hist.append("[{:4d} - {:5.1f}%] {}".format(x, x_normalized * 100, "∎" * int(x_normalized * 50)))

        hist.append("[{:4d} - 100.0%] total".format(int(data_sum)))
        return "\n".join(hist)

    def __str__(self):
        return ("[" + self.name + "] ").ljust(66, "-") + "\n" + self.histogram()

    def __repr__(self):
        return "Votes(\"{}\", n_bins={}, n_votes={})".format(self.name, self.n_bins, self.n_votes)

    @staticmethod
    def compare(votes_a, votes_b):
        return votes_a.mj < votes_b.mj


def test():
    n_voters = 100
    n_bins = 7

    votes_alice = Votes("Alice", [1, 9, 10, 42, 37, 1, 0])
    print(votes_alice)
    votes_bob = Votes("Bob", [0, 1, 19, 38, 28, 13, 1])
    print(votes_bob)
    votes_charlie = Votes("Charlie", [12, 24, 35, 16, 13, 0, 0])
    print(votes_charlie)
    votes_dennis = Votes("Dennis", [21, 10, 14, 8, 17, 20, 10])
    print(votes_dennis)

    list_votes = [votes_alice, votes_bob, votes_charlie, votes_dennis]
    pprint(sorted(list_votes, key=lambda votes: votes.mj, reverse=True))


# Output plot
def plot_mj(output_filename, mention_map, votes_dict_mention, candidate_map, candidate_order=None):
    # Check arguments
    if candidate_order is None:
        candidate_order = sorted(candidate_map.keys())
    else:
        if len(candidate_order) != len(candidate_map):
            raise ValueError("candidate_order list must have the same length as candidate_map")

    # Prepare data and plot
    n_votes = sum([votes_mention[0] for votes_mention in votes_dict_mention.values()])
    ind = range(len(candidate_map.keys()))
    cumulated = np.zeros(len(ind), dtype=int)

    plt.figure()
    plot_list = []
    lgd_labels = []

    # Build stacked bar plot
    for mention_id in sorted(mention_map.keys()):
        # -- Rearrange candidate order if necessary
        votes = np.array([votes_dict_mention[mention_id][candidate_id]
                          for candidate_id in candidate_order],
                         dtype=int)
        plot_list.append(plt.bar(ind, votes, bottom=cumulated,
                                 color=cm.RdYlGn(float(mention_id) / len(mention_map))))
        lgd_labels.append(mention_map[mention_id])
        cumulated += votes

    # Add median line
    plt.plot((ind[0] - 1, ind[-1] + 1), (n_votes / 2., n_votes / 2.))

    # Set plot details
    plt.title("RTM Name Vote Analysis")
    plt.ylabel("Votes")
    plt.xticks(ind, [candidate_map[candidate_id]
                     for candidate_id in candidate_order],
               rotation="60")
    plt.xlim((ind[0] - 0.5, ind[-1] + 0.5))
    plt.legend([p[0] for p in plot_list[::-1]], lgd_labels[::-1],
               bbox_to_anchor=(1.01, 1.05))

    plt.tight_layout()
    print("Saving plot to {}".format(output_filename))
    plt.savefig(output_filename)
    plt.close()


@click.command(help="Process a CSV file for majority judgement voting.")
@click.argument("filename")
def cli(filename):
    # Load CSV data
    with open(filename, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        csv_lines = [line for line in csv_reader]

    # Prepare data structures
    # -- Map: mention_name <=> mention_id
    mention_map = bidict({0: "To Reject",
                          1: "Poor",
                          2: "Acceptable",
                          3: "Good",
                          4: "Very Good",
                          5: "Excellent"})

    print("Variable: mention_map ".ljust(66, "-"))
    pprint(dict(mention_map))

    # -- Map: candidate_name <=> candidate_id
    candidate_map = bidict()
    for candidate_id, idx in zip(csv_lines[0][1:], range(len(csv_lines[0][1:]))):
        candidate_map[idx] = candidate_id
    print("Variable: candidate_map ".ljust(66, "-"))
    pprint(dict(candidate_map))

    # -- Dict: candidate_id => per-mention breakdown
    init_list = [0 for i in range(len(mention_map))]
    votes_dict_candidate = OrderedDict([(candidate_id, init_list[:])
                                        for candidate_id in range(len(candidate_map))])
    # pprint(dict(votes_dict_candidate))

    # -- Dict: mention_id => per-candidate breakdown
    init_list = [0 for i in range(len(candidate_map))]
    votes_dict_mention = OrderedDict([(mention_id, init_list[:])
                                      for mention_id in range(len(mention_map))])
    # pprint(dict(votes_dict_mention))

    # Process data
    print("Vote processing ".ljust(66, "-"))
    line_id = 0
    for line in csv_lines[1:]:
        line_id += 1
        print("Ballot #{}".format(line_id))

        for candidate_id in range(len(line) - 1):
            candidate_name = candidate_map[candidate_id]

            mention_name = line[candidate_id + 1]
            try:
                mention_id = mention_map.inv[mention_name]
                print("    {}: {}".format(candidate_name, mention_name))
            except KeyError:
                mention_name = "To Reject"
                mention_id = mention_map.inv[mention_name]
                print("    {}: {}*".format(candidate_name, mention_name))

            votes_dict_candidate[candidate_id][mention_id] += 1
            votes_dict_mention[mention_id][candidate_id] += 1
    print("Variable: votes_dict_candidate ".ljust(66, "-"))
    pprint(dict(votes_dict_candidate))
    print("Variable: votes_dict_mention ".ljust(66, "-"))
    pprint(dict(votes_dict_mention))

    # Cast results to Votes
    votes_list = [Votes(candidate_map[candidate_id], votes_dict_candidate[candidate_id])
                  for candidate_id in sorted(candidate_map.keys())]

    # Perform MJ sorting
    votes_list = sorted(votes_list, key=lambda votes: votes.mj, reverse=True)
    for votes in votes_list:
        print(votes)

    # Make plots
    sns.set()
    outfilename_prefix = filename.replace(".csv", "")

    # Plot in alphabetical order
    plot_mj("{}_alpha.png".format(outfilename_prefix),
            mention_map, votes_dict_mention, candidate_map)

    # Plot in MJ order
    candidate_order_mj = [candidate_map.inv[votes.name]
                          for votes in votes_list]
    plot_mj("{}_mj.png".format(outfilename_prefix),
            mention_map, votes_dict_mention, candidate_map, candidate_order=candidate_order_mj)


if __name__ == "__main__":
    cli()
