import argparse
from utils import load_graph, get_descriptions
from llm import Embedding, LLM
from benchmark import Benchmark
from prune import PruneTTL, PruneBackperf, PruneSummarize


def main(args):
    print("load graph ...")
    g = load_graph(args.graph_path)

    print("load embed model ...")
    embed_model = Embedding(args.embed_model_name)

    print("load llm ...")
    llm = LLM(args.llm_model_name)

    print("load benchmark ...")
    benchmark = Benchmark(args.benchmark_file, llm, embed_model, topk=args.topk)

    questions = benchmark.get_questions()

    if not args.pre_pruned:
        # Choose the pruning method based on the argument
        if args.pruning_method == "PruneBackperf":
            prune = PruneBackperf(
                args.threshold, embed_model, questions, topk=args.prune_topk
            )
        elif args.pruning_method == "PruneSummarize":
            prune = PruneSummarize(llm, embed_model)
        elif args.pruning_method == "PruneTTL":
            prune = PruneTTL(min_year=args.min_year)
        else:
            raise ValueError(f"Unknown pruning method: {args.pruning_method}")
        g = prune(g)
        if args.save:
            g.serialize(destination=args.save, format="turtle")

    descriptions = get_descriptions(g)
    print(benchmark(descriptions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--pruning_method",
        type=str,
        default="PruneTTL",
        choices=["PruneTTL", "PruneBackperf", "PruneSummarize"],
        help="Pruning method to use.",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="ttl/starwars_events.ttl",
        help="Path to the graph file.",
    )
    parser.add_argument(
        "--embed_model_name",
        type=str,
        default="nomic-embed-text",
        help="Name of the embedding model.",
    )
    parser.add_argument(
        "--llm_model_name", type=str, default="gemma3:4b", help="Name of the LLM model."
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="docs/benchmark.csv",
        help="Path to the benchmark file.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Threshold for pruning."
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="Top k value for benchmark."
    )
    parser.add_argument(
        "--prune_topk", type=int, default=100, help="Top k value for pruning."
    )
    parser.add_argument(
        "--pre_pruned",
        action="store_true",
        help="Flag indicating that the graph is already pruned.",
    )
    parser.add_argument(
        "--min_year", type=int, default=0, help="Minimum year for PruneTTL pruning."
    )
    parser.add_argument(
        "--save", type=str, help="Save the pruned graph to the specified file"
    )

    args = parser.parse_args()
    main(args)
