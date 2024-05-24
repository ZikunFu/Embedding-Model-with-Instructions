
# Embedding Model with Instructions

## Current Progress

- **Find or Build Datasets for Evaluation:**
    - Currently using Hugging face dataset [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
    - TODO: DBLP and Amazon reviews
    - TODO: Exploring new datasets
- **Generate Text Embedding**
    - Currently using Hugging face Transformers [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
    - Testing on BERT
    - TODO: Implement Sentence Transformer, T5, Instructor, and more
- **Enhance Text Embedding with Instructions:**
    - TODO: Experimenting with domain-specific instructions to improve embeddings
- **Evaluate the Quality of Original and Augmented Text Embeddings:**
    - Currently evalute original text embeddings using SVM, MLP model from SKLearn
    - TODO: Exploring other evaluation method and classification models for linear seperability.
    - TODO: Utilizing Hugging Face [MTEB scoreboard](https://github.com/embeddings-benchmark/mteb)

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