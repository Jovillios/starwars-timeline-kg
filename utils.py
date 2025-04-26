

from rdflib import Namespace, Graph
from rdflib.namespace import RDF
SW = Namespace("https://starwars.fandom.com/")

def get_descriptions(g):
        return [str(g.value(subject=event, predicate=SW.description))
                        for event in g.subjects(RDF.type, SW.Event)]


def load_graph(filename):
    g = Graph()
    g.parse(filename, format='turtle')
    return g
