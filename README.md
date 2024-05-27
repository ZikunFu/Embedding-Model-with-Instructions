
# Embedding Model with Instructions

## Current Progress

- **Find or Build Datasets for Evaluation:**
    - Working datasets
        - [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
            - 25k Train / 25k Test
            - labels (0 - neg; 1 - pos)
        - [yelp_review_full](https://huggingface.co/datasets/yelp_review_full)
            - 650k Train / 50k Test
            - labels (1 star ... 5 star)
    - TODO: Challenging datasets
        - [DBLP](https://github.com/angelosalatino/dblp-parser)
        - [Amazon reviews](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
    - TODO: keyword prediction from arXiv abstracts
    - TODO: adding noise to inputs (e.g., spelling errors)
    - TODO: Identify datasets where the selection of instructions is straightforward and beneficial to the model.
    - TODO: Continuously looking for more datasets
- **Generate Text Embedding**
    - Currently using Hugging face Transformers [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
    - Testing on BERT
    - TODO: Planning to implement BERT-large, Sentence Transformer, T5, Instructor, GPT2 and other models.
- **Enhance Text Embedding with Instructions:**
    - TODO: Experimenting with domain-specific instructions to improve embeddings
    - TODO: Explore other kinds of instructions to improve embeddings
- **Evaluate the Quality of Original and Augmented Text Embeddings:**
    - Currently evalute original text embeddings using SVM, MLP model from SKLearn
    - TODO: Exploring other evaluation method and classification models for linear seperability.
    - TODO: Utilizing Hugging Face [MTEB scoreboard](https://github.com/embeddings-benchmark/mteb) (Maybe)
- **Pipeline**
    - Inprogress: EmbedFlow for batch processing and experimenting on models.

## Future Research Questions:
- Does the inclusion of instructions improve the embeddings? (HOPEFULLY)
- How can prompt engineering be used to further enhance embedding models?

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/ZikunFu/Embedding-Model-with-Instructions.git
cd Embedding-Model-with-Instructions
```

### For Conda

```bash
conda create --name embed --file environment.yml
conda activate embed
```

### For Pip

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository.
2. Create and activate the environment.
3. Run the Jupyter Notebook.


## Acknowledgments

- Hugging Face for providing the pre-trained models.