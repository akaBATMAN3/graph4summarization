# Graph4Summarization

Abstractive summarization realized by [graph4nlp](https://github.com/graph4ai/graph4nlp), which includes the implementation of GCN, GAT, GraphSAGE, GGNN, Graph2Seq, attention mechanism, coverage mechanism, copy mechanism, etc. This repo provides detailed description.

## What does this repo done

1. Run the Summarization example from [graph4nlp](https://github.com/graph4ai/graph4nlp) successfully in both local and [AutoDL](https://www.autodl.com/register?code=ed898b66-46ba-45a2-8ef8-1b4d6127f8ae), and give detailed guide about [usage](#Usage)

2. Annotate the code (mostly in Chinese) and give details of implementation in [report](./report.md) (also written in Chinese)

3. Have [experiments](#Experiment) on four aspects which enriches the results of original example. To be specific, I tested the performance of different graph construction method, different graph encoder, different graph embedding generation method and different decoder mechanism.

   Graph construction: dependency graph, constituency graph, IE graph

   Graph encoder: GCN, GAT, GraphSAGE, GGNN

   Graph embedding: max-pooling, mean-pooling

   Decoder mechanism: attention, coverage, copy

## Dependency

Here I only give the dependency I used in cloud server ([AutoDL](https://www.autodl.com/register?code=ed898b66-46ba-45a2-8ef8-1b4d6127f8ae)) to avoid misguide:

- pytorch 1.7.1+cu110
- torchtext 0.8.1
- graph4nlp 0.5.5

## Usage

1. Download needed data from [preprocess.md](./examples/pytorch/summarization/cnn/raw/prerpocess.md) and run `preprocess.py` under `examples/pytorch/summarization/cnn` to preprocess the downloaded data

2. Start [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) server with cmd:

   ```
   cd stanford-corenlp-4.4.0
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
   ```

   About how to download and configure CoreNLP please click [here](https://stanfordnlp.github.io/CoreNLP/), I used CoreNLP 4.4.0 when running. This step is for graph construction, graph construction should be finished locally (I didn't test it in cloud server). **Remember to leave at least 4g memory when running and don't close the command line window unless you can see data.pt and vocab.pt under `examples/pytorch/summarization/cnn/processed` directory.**

3. Run the following code to start constructing graph, building model, training and testing model:

   ```
   python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_construction_cp/gcn_dependency.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml
   ```

   You can check processed graph data and vocab model under `examples/pytorch/summarization/cnn/processed` directory. I created more config file than the original example, you can replace the `gcn_dependency.yaml` with any file except `cnn.yaml` under `examples/pytorch/summarization/cnn/config` directory. When above steps have been done, check the results in `out` directory.

4. Replace `main.py` with `inference.py` in step3 to start inference

5. More information in [graph4nlp](https://github.com/graph4ai/graph4nlp) and [report](./report.md). **If there's any discrepancy between this repo and [graph4nlp](https://github.com/graph4ai/graph4nlp), the later should prevail.**

## Experiment

Here's part of experimental results, more analysis about the experiment in [report](./report.md).

| Compare graph construction methods | ROUGE-L(%) |
| ---------------------------------- | :--------: |
| **dependency**-GCN-attn-cov        |   29.23    |
| **constituency**-GCN-attn-cov      | **29.98**  |
| **IE**-GCN-attn-cov                |   27.51    |

| Compare graph encoders    | ROUGE-L(%) |
| ------------------------- | :--------: |
| IE-**GCN**-attn-cov       | **27.51**  |
| IE-**GAT**-attn-cov       |   27.10    |
| IE-**GraphSAGE**-attn-cov |   27.10    |
| IE-**GGNN**-attn-cov      |   27.40    |