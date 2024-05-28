
# Embedding Model with Instructions

## Current Progress

- **Datasets**
    - Implemented
        - [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
            - 25k Train / 25k Test
            - labels (0 - neg; 1 - pos)
        - [yelp_review_full](https://huggingface.co/datasets/yelp_review_full)
            - 650k Train / 50k Test
            - labels (1 star ... 5 star)
    - TODO: 
        - Add Challenging datasets
            - [DBLP](https://github.com/angelosalatino/dblp-parser)
            - [Amazon reviews](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
            - ArXiv abstracts (keyword prediction)
        - Adding noise to inputs (e.g., spelling errors)
        - Identify datasets where the selection of instructions is straightforward and beneficial to the model.
        - Exploring more datasets
- **Text Embedding Model**
    - Implemented:
        - Bert, Bert-Large, Instructor, T5
    - TODO: 
        - GPT2 and other models
- **Enhance Text Embedding with Instructions:**
    - [Experiment Log]()
    - TODO: 
        - Experimenting with domain-specific instructions to improve embeddings
        - Explore other kinds of instructions to improve embeddings
- **Evaluation**
    - SVM (Linear)
    - MLP
        - ReLu
    - TODO: 
        - Exploring other methods.
        - [MTEB scoreboard](https://github.com/embeddings-benchmark/mteb)

## Research Questions:
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