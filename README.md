# SciConcepts

Code for creating a scientific concept network from arXiv research papers.

![Concept network of significant links](network.png)

## Quick Start

```bash
git clone https://github.com/singularity4/SciConcepts.git
cd SciConcepts
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tested with Python 3.13. A GPU is used if available.

## Data

The scripts use the arXiv metadata snapshot `arxiv-metadata-oai-snapshot.json`, provided by Cornell University [Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).

## Usage

```bash
python sci_concepts.py --snapshot arxiv-metadata-oai-snapshot.json --out sci_concepts
python concept_network.py --input sci_concepts --out networks --all
python topics.py --input sci_concepts
python figures.py --input sci_concepts --networks networks
```

| Script | Description |
|---|---|
| `sci_concepts.py` | arXiv corpus, MultipartiteRank candidate concepts and concept list |
| `concept_network.py` | concept co-occurrence network and networks of significant links |
| `topics.py` | topic model (KNSC-Rcut) for validating the field categories |
| `figures.py` | figures of the corpus and the concept networks |



## License

MIT
