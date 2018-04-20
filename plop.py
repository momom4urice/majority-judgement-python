# -- coding: utf-8 --

from majorityjudgement import MajorityJudgement
from pprint import pprint


class Votes(object):
	def __init__(self, name, tally):
		assert(type(tally) is list)
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
		return  ("[" + self.name + "] ").ljust(66, "-") + "\n" + self.histogram()

	def __repr__(self):
		return "Votes(\"{}\", n_bins={}, n_votes={})".format(self.name, self.n_bins, self.n_votes)

	@staticmethod
	def compare(votes_a, votes_b):
		return votes_a.mj < votes_b.mj


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
