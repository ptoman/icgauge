import math
import numpy as np
import json
from matplotlib import pyplot as plt

hists = np.zeros((3,7))

for row, fn in enumerate(['train_rev.json', 'dev_rev.json', 'test.json']):
  scores = []
  with open(fn) as f:
    items = json.load(f)
    for i in items:
      score = int(math.floor(i['score'] + 0.5))
      scores.append(score)
  hists[row] = np.bincount(scores)[1:]


print hists

# Original:
#array([[ 202.,  328.,  204.,   37.,   20.,   13.,    2.],
#       [  43.,   59.,   35.,    6.,    9.,    4.,    6.],
#       [  59.,   75.,   44.,    7.,    5.,    1.,    1.]])
# Revised:
#array([[ 199.,  309.,  177.,   15.,    9.,    4.,    6.],
#       [  46.,   78.,   62.,   28.,   20.,   13.,    2.],
#       [  59.,   75.,   44.,    7.,    5.,    1.,    1.]])

num_per_set = np.sum(hists, axis=1)
print num_per_set
print np.round(num_per_set / np.sum(num_per_set),2)

# Original:
# [ 806.  162.  192.]
# [ 0.69,  0.14,  0.17]
# Revised:
# [ 719.  249.  192.]
# [ 0.62,  0.21,  0.17]


normed_hist = hists / num_per_set[:,np.newaxis]
print np.round(normed_hist,3)

# Original:
#[[ 0.251  0.407  0.253  0.046  0.025  0.016  0.002]
# [ 0.265  0.364  0.216  0.037  0.056  0.025  0.037]
# [ 0.307  0.391  0.229  0.036  0.026  0.005  0.005]]
# Revised:
#[[ 0.277  0.43   0.246  0.021  0.013  0.006  0.008]
# [ 0.185  0.313  0.249  0.112  0.08   0.052  0.008]
# [ 0.307  0.391  0.229  0.036  0.026  0.005  0.005]]

labels = ["Train (n=719)", "Dev (n=249)", "Test (n=192)"]

fig = plt.figure()
fig.patch.set_facecolor('white')
handles = []
for i,row in enumerate(normed_hist):
  plt.plot(range(1,8), row, "-D", label=labels[i])

plt.xlabel("Integrative Complexity Score")
plt.ylabel("Proportion of Data")
plt.title("Distribution of Complexity Scores")
plt.legend()
plt.show()

