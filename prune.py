
from rdflib import Graph, Namespace, Literal, XSD
from rdflib.namespace import RDF
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from torch import threshold
from llm import Embedding, LLM
from benchmark import EmbeddingBenchmark, Benchmark
import numpy as np
from tqdm import tqdm
from utils import get_descriptions


def remove_events(g, SW, events_to_remove: list):
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
        remove_events(g, SW, events_to_remove)

        print(
            f"Removed {n_events_to_remove} events out of {n_events} total events.")

        return g


class PruneBackperf(PruneAlgorithm):
    """
    Prune events based on a benchmark
    """

    def __init__(self, threshold, embed_model, questions, topk):
        super().__init__(name="PruneBackperf")
        # benchmark is a class that give a performance score between 0 and 1
        # without any pruning the performance score should be 1
        # and the more we prune the less the performance score
        self.benchmark = EmbeddingBenchmark(embed_model, questions, topk) 
        self.threshold = threshold


    def get_event_id(self, event):
        return int(event.split("_")[-1])

    def __call__(self, g):
        print(f"Pruning with {self.name} algorithm")

        SW = self.namespace
        events = list(g.subjects(RDF.type, SW.Event))
        n_events = len(events)

        events_to_remove = []

        for event in events:
            # Register to remove the events that doesn't degrade the benchmark given the threshold when removed
            if self.benchmark(self.get_event_id(event)) < self.threshold:
                events_to_remove.append(event)

        n_events_to_remove = len(events_to_remove)
        remove_events(g, SW, events_to_remove)
        print(
            f"Removed {n_events_to_remove} events out of {n_events} total events.")

        return g


class PruneSummarize(PruneAlgorithm):
    """
    Summarize events using a LLM algorithm.
    """

    def __init__(self, llm, embed_model):
        super().__init__(name="PruneSummarize")
        self.llm = llm
        self.embed_model = embed_model

    def embed_descriptions(self, g):
        SW = self.namespace
        descriptions = [g.value(subject=event, predicate=SW.description)
                        for event in g.subjects(RDF.type, SW.Event)]
        embeddings = self.embed_model.embed(descriptions)
        return embeddings, np.array(descriptions)

    def clusterize(self, embeddings):
        clustering = DBSCAN(eps=0.7, min_samples=2).fit(embeddings)
        return clustering.labels_

    def __call__(self, g):
        print(f"Pruning with {self.name} algorithm")
        SW = self.namespace

        # Embed the descriptions of the events
        print("embedding ...")
        embeddings, descriptions = self.embed_descriptions(g)

        # Cluster the embeddings
        print("clustering ...")
        clustering = self.clusterize(embeddings)
        clusters = np.unique(clustering)

        # Summarize the events in each cluster
        events = [event for event in g.subjects(RDF.type, SW.Event)]
        n_events = len(events)
        events_to_remove = []

        event_to_create = []
        # Iterate over the clusters and summarize the events
        for cluster in tqdm(clusters):
            # Skip the noise points
            if cluster < 0:
                continue

            # Get the indices of the events in this cluster
            idx = np.where(clustering == cluster)[0]

            # Initialize the variables for the new event
            event_year = g.value(
                subject=events[idx[0]], predicate=SW.eventYear)
            things = []

            # Iterate over the events in this cluster and register them for removal
            for i in idx:
                event = events[i]
                events_to_remove.append(event)

                # event year should be the max of all events in the cluster
                event_year = max(int(event_year), int(g.value(
                    subject=event, predicate=SW.eventYear)))

                # Get the related things for this event
                for thing in g.objects(event, SW.relatedTo):
                    if thing not in things:
                        things.append(thing)

            # Use the LLM to summarize the events in this cluster
            descriptions_cluster = descriptions[idx]
            summary = self.llm.summarize(descriptions_cluster)

            # Register the new event to create
            event_to_create.append((
                summary,
                event_year,
                things
            ))

        # Create the new events
        for summary, event_year, things in event_to_create:
            new_event = SW[f"event_{len(list(g.subjects(RDF.type, SW.Event)))}"]
            g.add((new_event, RDF.type, SW.Event))
            g.add((new_event, SW.eventYear, Literal(
                event_year, datatype=XSD.integer)))
            g.add((new_event, SW.description, Literal(
                summary, datatype=XSD.string)))
            for thing in things:
                # Add the related things to the new event
                g.add((new_event, SW.relatedTo, thing))

        n_events = len(events)
        n_events_to_remove = len(events_to_remove)
        n_events_to_create = len(event_to_create)
        remove_events(g, SW, events_to_remove)
        print(
            f"Removed {n_events_to_remove} events out of {n_events} total events and created {n_events_to_create} new events.")
        return g


class PruningTester:
    """To adapt to see the influence of the number of triple for example"""

    def __init__(self, graphs, pruning_algorithms, benchmark_function):
        self.dict_graph = graphs
        self.pruning_algorithms = pruning_algorithms
        self.benchmark_function = benchmark_function
        self.results = {}

    def test_all_graph(self):
        for graph_name, graph in self.dict_graph.items():
            self.test_graph(graph, graph_name)

    def test_graph(self, graph, grah_name="default"):
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

    def plot_results(self, metrics=['time_to_prune', 'final_performance', 'final_triples', 'triples_removed']):
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


