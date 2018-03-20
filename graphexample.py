import numpy as np
from numpy import random, cos, sin
from numpy.random import uniform
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
x = 2
y = 2
color_map = []
vec = range(1, 60)
sample = [-1, 1]
G.add_node(0, pos=(2,2))
for i in vec:
    angle = uniform(0, 2 * np.pi)
    xcoord = x + cos(angle) * uniform(0.8, 1.2)
    ycoord = y + sin(angle) * uniform(0.8, 1.2)
    G.add_node(i, pos=(xcoord, ycoord))
    G.add_edge(0, i)

for j in range(100):
    G.add_edge(np.random.choice(vec), np.random.choice(vec))

for node in G:
    if node == 0:
        color_map.append('red')
    else:
        color_map.append('blue')

pos = nx.get_node_attributes(G, 'pos')
# plt.Circle((2, 2), 0.2, color='g', clip_on=False)
nx.draw(G, pos, edge_color='grey', node_color = color_map, node_size = 105)
plt.draw()
plt.savefig('plots/withcenter.png', transparent = True)
plt.show()
plt.clf()

color_map = color_map[1:len(color_map) - 1]
G.remove_node(0)
nx.draw(G, pos, edge_color = 'grey', node_color = color_map, node_size = 105)
plt.draw()
plt.savefig('plots/withoutcenter.png', transparent = True)
plt.show()
plt.clf()
