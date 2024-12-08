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

## Methods

#### SAE latent co-occurrence heatmaps (Figure 3A)
See `src/size_effects/gpt2_768_heatmap_and_cluster_stats.ipynb`

#### SAE latent co-occurrence Jaccard normalisation histogram (Figure 4A): 
See `src/size_effects/gpt2_768_heatmap_and_cluster_stats.ipynb`

#### Effect of Jaccard normalisation on subgraph size and degree (Figure 5):
See `src/size_effects/gpt2_768_heatmap_and_cluster_stats.ipynb`

#### SAE latent occurrence per token with width and L0 (Figure 3B and Appendix Figure \app_fig_gemma_features_vs_l0): 
See `src/size_effects/features_active_per_token.py`

#### SAE latent co-occurrence boxplots and density plots (Figure 7 and Appendix Figure \gemma-2-2b-boxplots-density-vs-width and \gemma-2-2b-boxplots-density-vs-l0):
See `src/size_effects/feature_distribution_data.py` and `src/size_effects/feature_distribution_plots.py`

#### SAE latent cluster size with width and L0 (Figure 9, Appendix Figure \app_subgraph_size_vs_width,  \app_subgraph_size_vs_l0):
See `src/size_effects/subgraph_size_vs_width.ipynb`

#### SAE latent and subgraph sparsity with width and L0 (Figure 9, Appendix Figure \app_subgraph_l0_width and \app_subgraph_l0_l0):
See `src/size_effects/feature_graph_l0.py`

## Results
#### SAEs trained on Gemma-2-2b (Gemma Scope) encode qualitative statements about the number of items compositionally (Figure \latent_mix_vs_semantic_mix, \one_of_mean_activation_per_group, \one_of_mean_activation_per_group_zoomin):
See `src/example_clusters/gemma_one_of_4740_layer_12_1_5_activation_100_batch_100_pca.ipynb`. This relies on loading the h5 file generated by `5_pca_for_streamlit.py` for this cluster. 
<!-- See also `src/example_clusters/gemma_one_of_59_layer_18_100_batch_1_5.ipynb`. -->

####  Encoding of continuous properties in feature strength without compositionality (Figure \counting_overview,  \counting_feature_strength, Appendix Figure \app_first_second_third, \app_first_second_third_custom):
See `src/example_clusters/gemma_counting_1370_layer_0_1_5_100_100.ipynb` and `src/example_clusters/gemma_first_second_layer_21_1_5_511.ipynb`.

####  Encoding of continuous properties in feature strength without compositionality (Appendix Figure \app_first_second_third_custom):
See `src/example_clusters/gemma_apostrophe_4334_layer_12_1_5_100_100.ipynb`

####  Distinguishing between uses of the word 'how' in GPT2-Small:
See `src/example_clusters/gpt2_layer8_24k_1_5_787_how.ipynb`

## Streamlit App

Streamlit app uses data generated by `5_pca_for_streamlit.py` which depends on the dataframes generated by `2_generate_graphs_loop.py`. 

In cases where the data files are too large to be stored on a github repository, the data files can be split with `src/datahandling/split_h5.py`. When running locally this is not necessary.

To run the streamlit app:
```bash
poetry run streamlit run sae_cooccurrence/general_streamlit.py
```

### Streamlit App Configuration

The streamlit app configuration is stored in `sae_cooccurrence/config_pca_streamlit_maxexamples.toml`. This allows the user to specify the models, SAE releases, and SAE IDs to use in the app.
