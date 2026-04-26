<p align="center">
	<img src="assets/sun.jpeg" height="400" />
</p>

# 😶 Wordcel

`wordcel` is a library of functions that provides a set of common tools for working with large language models.

Candidly, it is mostly a set of functions that I myself use on a regular basis — my own personal Swiss army knife. 

## Installation

You can simply `pip install wordcel`.

## Documentation

- [LLM APIs](docs/llms.md): Wrapper functions over the most common LLM APIs.
- [featurize](docs/featurize.md): Helper functions for multithreaded inference over text columns in pandas DataFrames.
- [DAG](docs/dag.md): WordcelDAG is a flexible and extensible framework for defining and executing Directed Acyclic Graphs (DAGs) of data processing tasks, particularly involving LLMs and dataframes. 

There is also a nascent CLI. `wordcel --help`:

```
Usage: wordcel [OPTIONS] COMMAND [ARGS]...

  Wordcel CLI.

Options:
  --help  Show this message and exit.

Commands:
  dag  WordcelDAG commands.
```