# Embedding Model with Instructions

Do natural-language instructions improve the quality of text embeddings for downstream classification? This project builds a small experimental pipeline to compare embeddings from several pretrained language models — with and without instruction/prefix text prepended to the input — by embedding text, feeding the embeddings to a simple classifier, and measuring classification accuracy. The target use case is domain-specific semantic search / text classification (movie reviews, restaurant reviews, and research-paper abstracts) evaluated as a proxy task.

## Approach

**Embedding models compared** (`Research_embed.ipynb`):
- `google-bert/bert-base-uncased` (CLS-token pooling)
- `google-bert/bert-large-uncased` (mean pooling)
- `hkunlp/instructor-large` via `sentence-transformers` (instruction-conditioned encoder — instructions are passed as a `[instruction, text]` pair rather than concatenated text)
- `t5-base` (mean-pooled encoder hidden states)
- `openai-community/gpt2` (mean pooling)

A separate, simplified reference implementation of the pipeline (`TextEmbeddingPipeline.py`, the `EmbedFlow` class) supports BERT, T5, Instructor, and `sentence-transformers/all-MiniLM-L12-v2` for standalone use outside the notebook.

**Datasets** (loaded via Hugging Face `datasets`):
- [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) — binary movie-review sentiment (25k train / 25k test in the full dataset)
- [`yelp_review_full`](https://huggingface.co/datasets/yelp_review_full) — 1-5 star restaurant/business review ratings (650k train / 50k test in the full dataset)
- [`Voice49/arXiv-Abstract-Label-20k`](https://huggingface.co/datasets/Voice49/arXiv-Abstract-Label-20k) — arXiv abstracts labeled by primary category (10,000 train / 10,000 test). This dataset was built and published from scratch for this project (`arXiv.ipynb`), by pulling abstracts via the `arxiv` API across 8 primary categories (`cs`, `econ`, `eess`, `math`, `physics`, `q-bio`, `q-fin`, `stat`) and pushing the result to the Hugging Face Hub.

For the experiments actually run, each dataset was subsampled (e.g. 1,000 train / 1,000 test examples for IMDB) rather than trained on the full split, to keep iteration fast.

**Instruction augmentation**: for BERT, BERT-large, T5, and GPT-2, an instruction string is prepended as a text prefix before embedding (e.g. `"Sentiment Analysis: " + review_text`). For Instructor, the instruction is passed alongside the text as an `(instruction, text)` pair, per that model's intended usage. Each dataset has its own set of candidate instructions (general-purpose phrasings plus dataset-specific phrasings, both short and long forms) that are compared against a no-instruction baseline.

**Evaluation**: embeddings are used as fixed features for two simple classifiers — a linear SVM (`sklearn.svm.SVC`) and an MLP (`sklearn.neural_network.MLPClassifier`) — scored with `sklearn.metrics.classification_report` / `accuracy_score`.

## Results

The full model x instruction x dataset sweep in `Research_embed.ipynb` was interrupted (`KeyboardInterrupt`) partway through the first dataset, so no complete comparison table or plot was produced — the notebook cells that aggregate and plot results (mean/std accuracy per instruction group) were written but never executed to completion.

One concrete data point was recorded before the interruption: BERT-base embeddings (CLS pooling) with the instruction prefix `"Sentiment Analysis: "`, evaluated with a linear SVM on a 1,000-example IMDB train/test split, achieved:

| Metric | Class 0 (neg) | Class 1 (pos) | Overall accuracy |
|---|---|---|---|
| Precision | 0.7960 | 0.7720 | |
| Recall | 0.7773 | 0.7910 | |
| F1 | 0.7866 | 0.7814 | |
| - | | | **0.7840** |

This project should be read as a working experimental harness for the instruction-sensitivity question rather than a finished benchmark study — the infrastructure (data loading, five embedding backends, instruction augmentation, SVM/MLP evaluation) is implemented and runs end-to-end for at least one configuration, but the broader sweep across models, instructions, and datasets was not completed in the committed notebook. An experiment log intended to track additional runs is linked from the notebook.

## How to Run

```bash
git clone https://github.com/ZikunFu/Embedding-Model-with-Instructions.git
cd Embedding-Model-with-Instructions
```

Conda:
```bash
conda create --name embed --file environment.yml
conda activate embed
```

Pip:
```bash
pip install -r requirements.txt
```

Then open `Research_embed.ipynb` in Jupyter to run the embedding/instruction experiments, or `arXiv.ipynb` to see how the arXiv abstract dataset was built. `TextEmbeddingPipeline.py` can also be imported directly (`from TextEmbeddingPipeline import EmbedFlow`) for a scripted BERT/T5/Instructor/MiniLM pipeline on the IMDB dataset.

## Acknowledgments

- Hugging Face for the pretrained models and `datasets` library.

---
Graduate coursework project, Ontario Tech University (2024).
