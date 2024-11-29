import json

# from webbrowser import open as open_browser
import urllib.parse
import webbrowser

from sae_lens import SAE

NEURONPEDIA_DOMAIN = "https://neuronpedia.org"


def get_neuronpedia_feature_dashboard_no_open(sae: SAE, index: int, open: bool = False):
    sae_id = sae.cfg.neuronpedia_id
    if sae_id is None:
        print(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )
        return ""
    else:
        url = f"{NEURONPEDIA_DOMAIN}/{sae_id}/{index}"
        if open:
            webbrowser.open(url)
        return url


def get_neuronpedia_quick_list_no_open(
    sae: SAE,
    features: list[int],
    name: str = "temporary_list",
    open: bool = False,
):
    sae_id = sae.cfg.neuronpedia_id
    if sae_id is None:
        print(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )
        return ""

    url = NEURONPEDIA_DOMAIN + "/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {
            "modelId": sae.cfg.model_name,
            "layer": sae_id.split("/")[1],
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    if open:
        webbrowser.open(url)

    return url


# def warn_if_ends_with_digit(input_str: str) -> None:
#     if input_str[-1].isdigit():
#         warnings.warn(
#             "SAE ID ends with a digit, do you mean to be looking at a feature splitting dataset?",
#             UserWarning,
#         )


# def get_layer_from_id(sae_id: str) -> int:
#     if "layer" in sae_id.lower():
#         layer_match = re.search(r"layer_(\d+)", sae_id)
#         if layer_match:
#             layer = int(layer_match.group(1))
#             return layer
#         raise ValueError(
#             "sae_id does not match the expected format when 'layer' is in the sae_id"
#         )

#     layer_match = re.search(r"blocks\.(\d+)", sae_id)
#     if layer_match:
#         layer = int(layer_match.group(1))
#         return layer

#     raise ValueError(f"Unable to extract layer from sae_id: {sae_id}")


# def get_sae_size_from_id(sae_id: str) -> int:
#     match = re.search(r"_([0-9]+)$", sae_id)

#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError("sae_id not in feature splitting format")


# def mc_neuronpedia_link(
#     feature: int,
#     sae_id: str,
#     model: str = "gpt2-small",
#     dataset: str = "res-jb",
#     NEURONPEDIA_DOMAIN="https://neuronpedia.org",
# ):
#     if dataset == "res-jb":
#         warn_if_ends_with_digit(sae_id)
#         layer = get_layer_from_id(sae_id)
#         url = f"{NEURONPEDIA_DOMAIN}/{model}/{layer}-{dataset}/{feature}"
#     elif dataset == "res-jb-feature-splitting":
#         layer = get_layer_from_id(sae_id)
#         sae_size = get_sae_size_from_id(sae_id)
#         # https://www.neuronpedia.org/gpt2-small/8-res_fs768-jb/0
#         url = f"{NEURONPEDIA_DOMAIN}/{model}/{layer}-res_fs{sae_size}-jb/{feature}"
#     elif dataset == "gemma-scope-2b-pt-res-canonical":
#         layer = get_layer_from_id(sae_id)
#         url = f"{NEURONPEDIA_DOMAIN}/{model}/{layer}-{dataset}/{feature}"
#     return url


# def mc_quicklist(
#     features: list[int],
#     sae_id: str,
#     model: str = "gpt2-small",
#     dataset: str = "res-jb",
#     name: str = "temporary_list",
#     NEURONPEDIA_DOMAIN="https://neuronpedia.org",
# ):
#     url = NEURONPEDIA_DOMAIN + "/quick-list/"
#     name = urllib.parse.quote(name)
#     url = url + "?name=" + name

#     if dataset == "res-jb":
#         warn_if_ends_with_digit(sae_id)
#         layer = get_layer_from_id(sae_id)
#         layer_for_quicklist = f"{layer}-{dataset}"
#     elif dataset == "res-jb-feature-splitting":
#         layer = get_layer_from_id(sae_id)
#         sae_size = get_sae_size_from_id(sae_id)
#         layer_for_quicklist = f"{layer}-res_fs{sae_size}-jb"
#         # https://www.neuronpedia.org/gpt2-small/8-res_fs768-jb/0
#     elif dataset == "gemma-scope-2b-pt-res-canonical":
#         layer = get_layer_from_id(sae_id)
#         layer_for_quicklist = f"{layer}-{dataset}"

#     list_feature = [
#         {
#             "modelId": model,
#             "layer": layer_for_quicklist,
#             "index": str(feature),
#         }
#         for feature in features
#     ]
#     url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
#     # open(url)

#     return url
