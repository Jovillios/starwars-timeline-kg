
from rdflib import Graph, Namespace
from rdflib.namespace import RDF


class PruneAlgorithm:
    """
    Base class for pruning algorithms.
    """

    def __init__(self, name):
        self.name = name
        self.namespace = Namespace("https://starwars.fandom.com/")

    def __call__(self, g):
        print(f"Pruning with {self.name} algorithm")
        print("Removed 0 events out of 0 total events.")
        return g

    def __str__(self):
        return self.name


class PruneTTL(PruneAlgorithm):
    """
    Prune events based on their year.
    """

    def __init__(self, min_year=None, max_year=None):
        super().__init__(name="PruneTTL")
        self.min_year = min_year
        self.max_year = max_year

    def __call__(self, g):
        print(f"Pruning with {self.name} algorithm")

        SW = self.namespace
        events_to_remove = []

        n_events = len(list(g.subjects(RDF.type, SW.Event)))
        for event in g.subjects(RDF.type, SW.Event):
            year_literal = g.value(subject=event, predicate=SW.eventYear)
            if year_literal is None:
                continue
            year = int(str(year_literal))

            if (self.min_year is not None and year < self.min_year) or (self.max_year is not None and year > self.max_year):
                events_to_remove.append(event)

        n_events_to_remove = len(events_to_remove)
        for event in events_to_remove:
            # Remove all triples about this event
            for triple in g.triples((event, None, None)):
                g.remove(triple)
            # Optionally: remove Thing nodes if they're no longer linked
            for related_thing in g.objects(event, SW.relatedTo):
                # Remove the thing only if no other events refer to it
                if not list(g.subjects(SW.relatedTo, related_thing)):
                    for triple in g.triples((related_thing, None, None)):
                        g.remove(triple)

        print(
            f"Removed {n_events_to_remove} events out of {n_events} total events.")

        return g


class PruneSummarize(PruneAlgorithm):
    """
    Summarize events using a LLM algorithm.
    """

    def __init__(self, llm=None):
        super().__init__(name="PruneSummarize")
        self.llm = llm

    def __call__(self, g):
        print(f"Pruning with {self.name} algorithm")
        # Implement the summarization logic here
        # For now, just return the graph unchanged
        return g


def load_graph(filename):
    g = Graph()
    g.parse(filename, format='turtle')
    return g


if __name__ == "__main__":
    g = load_graph("ttl/starwars_events.ttl")
    pruneTTl = PruneTTL(min_year=0)
    g = pruneTTl(g)
    g.serialize(destination="ttl/starwars_events_pruned.ttl", format='turtle')
    print("Graph saved to ttl/starwars_events_pruned.ttl")
