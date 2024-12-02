# Compositionality and Ambiguity: Latent Co-occurrence and Interpretable Subspaces
Matthew A. Clarke, Hardik Bhatnagar, Joseph Bloom

This repository contains the code for the LessWrong "Compositionality and Ambiguity: Latent Co-occurrence and Interpretable Subspaces" (in progress).

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install dependencies, run:

```bash
poetry install
```

## Project Structure

### Pre-requisites

Access to HuggingFace via API token for base models.

### Usage

In `src/cooc`:

#### Script 1: Generate Normalized Features
`1_generate_normalised_features_loop.py` generates normalised feature co-occurrence data according to a config file specified by the user. These are saved as npz compressed numpy arrays. For each model and SAE layer this generates:

- A co-occurrence matrix of shape `[n_features, n_features]` (e.g., `feature_acts_cooc_total_threshold_1_5.npz`)
- A normalised co-occurrence matrix of shape `[n_features, n_features]` (e.g., `feature_acts_cooc_jaccard_threshold_1_5.npz`)
- A list of overall feature occurrences of shape `[n_features]` (e.g., `feature_acts_total_threshold_1_5.npz`)

#### Script 2: Generate Graphs
`2_generate_graphs_loop.py` calculates a threshold such that the largest connected component of the normalised feature co-occurrence graph is below a threshold. It then generates a dataframe of all the nodes, their subgraphs, and neuronpedia links for said subgraphs.

Output:
- A datatable of all nodes, subgraphs, and neuronpedia links (e.g., `dataframes/node_info_df_1_5.csv`)

#### Script 3: Analyze Subspaces
`3_analyse_subspaces.ipynb` - Template notebook to analyse a subspace using PCA.

#### Script 5: PCA for Streamlit
`5_pca_for_streamlit.py` generates PCA data for a set of example graphs.

Output:
- An h5 file containing:
  - PCA data (pca_df)
  - Results from pca.py/ProcessedResults class (includes tokens, context, etc.)

### Configuration Options

#### Generation
- `n_batches`: Number of batches of the activation store to cycle through (Default: 1000 for gpt2-small, 500 for feature splitting)
- `model_name`: Name of the model to use
- `sae_release_short`: Short name of the SAE release (e.g., 'res-jb' or 'res-jb-feature-splitting')
- `sae_ids`: List of SAE IDs to use
- `activation_thresholds`: List of activation thresholds for feature activation counting

#### Analysis
- `random_seed`: Random seed
- `min_subgraph_size`: Minimum size of largest connected component (default: 150)
- `max_subgraph_size`: Maximum size of largest connected component (default: 200)
- `min_subgraph_size_to_plot`: Minimum size for HTML/pyvis visualization
- `skip_subgraph_plots`: Toggle subgraph plotting
- `skip_subgraph_pickles`: Toggle saving subgraphs as pickle files
- `include_metrics`: Toggle inclusion of hubness metrics

#### PCA
- `candidate_sizes`: List of candidate sizes for PCA analysis
- `candidates_per_size`: Number of candidates per size
- `n_batches_reconstruction`: Number of batches for PCA reconstruction
- `recalculate_results`: Deprecated

## Development

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, [Pyright](https://github.com/microsoft/pyright) for type checking, and [Pytest](https://docs.pytest.org/en/stable/) for testing.

To run all checks:
```bash
make check-ci
```

### IDE Setup
In VSCode, install the Ruff extension for automatic linting and formatting. Enable formatting on save for best results.

Install pre-commit hook for automatic linting and type-checking:
```bash
poetry run pre-commit install
```

### Poetry Tips

Common Poetry commands:
- Install main dependency: `poetry add <package>`
- Install development dependency: `poetry add --dev <package>`
- Update lockfile: `poetry lock`
- Run command in virtual environment: `poetry run <command>`
- Run Python file as module: `poetry run python -m sae_coocurrence.path.to.file`

# Figure Reproduction

## SAE latent co-occurrence heatmaps (Figure 3A)
See `src/size_effects/gpt2_768_heatmap_and_cluster_stats.ipynb`

## SAE latent co-occurrence Jaccard normalisation histogram (Figure 4A): 
See `src/size_effects/gpt2_768_heatmap_and_cluster_stats.ipynb`

## Effect of Jaccard normalisation on subgraph size and degree (Figure 5):
See `src/size_effects/gpt2_768_heatmap_and_cluster_stats.ipynb`

## SAE latent occurrence per token with width and L0 (Figure 3B and Appendix Figure \app_fig_gemma_features_vs_lo): 
See `src/size_effects/features_active_per_token.py`