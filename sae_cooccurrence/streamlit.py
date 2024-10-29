from os.path import join as pj

import toml

from sae_cooccurrence.utils.set_paths import get_git_root


def load_streamlit_config(filename):
    config_path = pj(get_git_root(), "src", filename)
    with open(config_path) as f:
        return toml.load(f)
