from math import e
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

url = 'ttl/starwars_events_20.ttl'

g = rdflib.Graph()
result = g.parse(url, format='turtle')
# result = remove_namespaces(g)


def transform(s):
    return s
    if isinstance(s, rdflib.URIRef):
        s = str(s).split('/')[-1]
    return s


G = rdflib_to_networkx_multidigraph(
    result, transform_s=transform, transform_o=transform)

# # print edges
# for e in G.edges(keys=True):
#     print(e)
#     print("---")
# # Draw the graph using matplotlib
# plt.figure(figsize=(12, 12))
# plt.axis('off')
# pos = nx.spring_layout(G, seed=42)
# nx.draw_networkx_nodes(G, pos, node_size=50)
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
# plt.title("Star Wars Events Graph")
# plt.show()

# Plot with pyvis
net = Network(
    directed=True,
    select_menu=True,
    filter_menu=True,
    notebook=True,
)
net.show_buttons()
net.from_nx(G)

net.show("graph.html", notebook=True)
