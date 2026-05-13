# Cross Lingual Technical Knowledge Graph Completion via Relational GNN

Connects what SAP knows in German to what NTT knows in Japanese.

## Overview

This project aligns multilingual knowledge graphs, trains a Relational Graph Convolutional Network to predict missing entity relations, and exposes inference through a local search interface. It addresses the core unsolved problem of inferring missing facts across massive internal knowledge bases in multiple languages, making it highly relevant for global technology firms.

## Architecture

The end to end system consists of several distinct stages

* KG Construction extracting Wikidata SPARQL dumps for German, Japanese, and Dutch subsets
* Entity Alignment using multilingual embeddings and cosine similarity to generate anchor pairs
* Unified Graph merging entity nodes and language tagged edges into a single data structure
* Graph Neural Network Training learning entity and relation embeddings using a RotatE objective
* Link Prediction generating top k tail predictions for a given head and relation query
* Evaluation calculating standard research metrics like Hits@1, Hits@10, and Mean Reciprocal Rank on held out triples
* Local Search UI querying the knowledge graph and viewing predicted missing links via a web application

## Technology Stack

* Python
* PyTorch Geometric
* DGL
* HuggingFace 
* FastAPI
* React
* NetworkX
* SPARQLWrapper

## Implementation Details

The codebase is structured into distinct modular blocks to ensure maintainability and separation of concerns

* Utilities contains configuration loaders, logging setup, and text normalization tools.
* Data Pipeline handles robust SPARQL extraction with retry and backoff logic, triple preprocessing, cross lingual entity alignment, and PyTorch Geometric graph construction.
* Models implements the R GCN architecture with basis decomposition and the RotatE link prediction scorer.
* Training contains the optimization loop with early stopping, margin ranking loss, and a robust evaluator that implements filtered ranking to prevent penalizing the model for predicting known true facts.
* Inference provides an optimized predictor engine with cold start fallback using embedding similarity.
* Web Application serves the interactive search user interface, graph explorer, and statistics dashboard.

## Getting Started

First install Python 3.10 and recreate the virtual environment. Install PyTorch CPU wheels and PyTorch Geometric followed by the remaining dependencies. 

Run the pipeline script to extract data and build the unified graph. Train the model using the provided Jupyter notebook on a GPU instance. Download the best model checkpoint to the local checkpoints directory. Finally start the FastAPI server to access the user interface.
