# CCTS
If using this work, please cite fellow bibtex:

@article{SUN2025116198,
title = {Explainable community detection},
journal = {Chaos, Solitons & Fractals},
volume = {194},
pages = {116198},
year = {2025},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2025.116198},
url = {https://www.sciencedirect.com/science/article/pii/S0960077925002115},
author = {Xiaoxuan Sun and Lianyu Hu and Xinying Liu and Mudi Jiang and Yan Liu and Zengyou He}
}


This project implements community detection algorithms and analyzes the detected communities by finding optimal center nodes and threshold distances to explain the communities.

## Features

- Implements multiple community detection methods:
  - Louvain
  - Girvan-Newman (GN)
  - Label Propagation (LPA)
  - Fast Newman (FN)
- Analyzes detected communities by:
  - Finding optimal center nodes
  - Determining threshold distances
  - Calculating precision and error rates
- Provides flexible configuration options:
  - Partial community analysis
  - Threshold range limiting
  - Threshold pruning
  - BFS optimization

## Requirements

- Python 3.x
- Required packages:
  - networkx
  - python-louvain
  - matplotlib
  - numpy
  - scipy
  - alphashape
  - pickle

## Usage

1. Place your graph data in `data/graph/` directory as `.txt` files

2. Configure the parameters at the top of the script:
   - `method`: Choose from ['Louvain','GN','LPA','FN']
   - `names`: List of dataset names to analyze
   - `index`: Index of dataset to analyze
   - Various boolean flags to control analysis behavior
tag=True is the fast algorithm
tag=False is the naive algorithm

3. Run the script:
   ```bash
   python CCTS.py

## baseline
baseline.py is the comparative test baseline in the article
