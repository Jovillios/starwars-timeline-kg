

from rdflib import Namespace
from rdflib.namespace import RDF
SW = Namespace("https://starwars.fandom.com/")

def get_descriptions(g):
        return [str(g.value(subject=event, predicate=SW.description))
                        for event in g.subjects(RDF.type, SW.Event)]
