
from rdflib import Graph, Namespace
from rdflib.namespace import RDF
import time
import matplotlib.pyplot as plt


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

class PruneBACKPERF(PruneAlgorithm):
    """
    Prune events based on a benchmark
    """
    def __init__(self, benchmark):
        super().__init__(name="PruneBACKPERF")
        self.benchmark = benchmark

    def __call__(self, g):
        print(f"Pruning with {self.name} algorithm")

        SW = self.namespace
        n_triples_removed = 0
        n_triples = len(list(g))
        
        for event in g.subjects(RDF.type, SW.Event):
            # Remove the useless triples about this event
            for triple in g.triples((event, None, None)):  
                g_copy = copy.deepcopy(g)
                g_copy.remove(triple)
                if self.benchmark(g_copy) > self.benchmark(g):
                    g.remove(triple)
                    n_triples_removed += 1
                
        print(
            f"Removed {n_triples_removed} events out of {n_triples} total events.")

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

class PruningTester:
    """To adapt to see the influence of the number of triple for example"""
    def __init__(self, graphs, pruning_algorithms, benchmark_function):
        self.dict_graph = graphs
        self.pruning_algorithms = pruning_algorithms
        self.benchmark_function = benchmark_function
        self.results = {{}}
        
    def test_all_graph(self):
        for graph_name, graph in self.dict_graph.items():
            self.test_graph(graph, graph_name)

    def test_graph(self, graph, grah_name = "default"):
        results = {}
        initial_triples = len(graph)
        initial_performance = self.benchmark_function(graph)

        for algo_name, algorithm in self.pruning_algorithms.items():
            start_time = time.time()
            pruned_graph = algorithm(graph)
            end_time = time.time()

            final_triples = len(pruned_graph)
            final_performance = self.benchmark_function(pruned_graph)

            self.results[algo_name][grah_name] = {
                'time_to_prune': end_time - start_time,
                'initial_performance': initial_performance,
                'final_performance': final_performance,
                'initial_triples': initial_triples,
                'final_triples': final_triples,
                'triples_removed': initial_triples - final_triples
            }

    def display_results(self, results):
        for algo_name, result_by_algo in results.items():
            for graph_name, result in result_by_algo.items():
                print(f"Algorithm: {algo_name}")
                print(f"Graph : {graph_name}")
                print(f"Time to prune: {result['time_to_prune']} seconds")
                print(f"Initial performance: {result['initial_performance']}")
                print(f"Final performance: {result['final_performance']}")
                print(f"Initial triples: {result['initial_triples']}")
                print(f"Final triples: {result['final_triples']}")
                print(f"Triples removed: {result['triples_removed']}")
                print("-" * 40)
                
    def plot_results(self, metrics = ['time_to_prune', 'final_performance', 'final_triples', 'triples_removed']):
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for algo_name, result_by_algo in self.results.items():
                x = list(result_by_algo.keys())
                y = [result[metric] for result in result_by_algo.values()]
                plt.plot(x, y, label=algo_name)
            plt.xlabel('Graph Name')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} vs Graph Name')
            plt.legend()
            plt.grid(True)
            plt.show()
            
if __name__ == "__main__":
    g = load_graph("ttl/starwars_events.ttl")
    pruneTTl = PruneTTL(min_year=0)
    g = pruneTTl(g)
    g.serialize(destination="ttl/starwars_events_pruned.ttl", format='turtle')
    print("Graph saved to ttl/starwars_events_pruned.ttl")
