from datetime import datetime
import json
import logging
import os
import requests


logger = logging.getLogger(__name__)

API_BASE_URL = "https://huggingface.co/api/"
HF_CO_URL_TEMPLATE = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def get_datasets():
    response = requests.get(API_BASE_URL + "datasets")
    response.raise_for_status()
    return response.json()


def get_adapters():
    params = {"filter": "adapter-transformers", "full": True}
    response = requests.get(API_BASE_URL + "models", params=params)
    response.raise_for_status()
    return response.json()


def _has_sibling(adapter_info, sibling):
    files = [sibling["rfilename"] for sibling in adapter_info["siblings"]]
    return sibling in files


def build_adapter_entries():
    for adapter_info in get_adapters():
        adapterhub_tag = next((t for t in adapter_info["tags"] if t.startswith("adapterhub:")), None)
        # TODO also allow adapters without this tag
        if adapterhub_tag is None:
            logger.warning(f"[{adapter_info['modelId']}] No adapterhub tag.")
            continue
        task_splits = adapterhub_tag.split(":")[-1].split("/")
        if len(task_splits) != 2:
            logger.warning(f"Invalid adapterhub tag {adapterhub_tag}")
            continue
        task, subtask = task_splits
        groupname, filename = adapter_info["modelId"].split("/")
        last_update = datetime.strptime(adapter_info["lastModified"].split(".")[0], "%Y-%m-%dT%H:%M:%S")
        # Get values from adapter_config.json
        response = requests.get(
            HF_CO_URL_TEMPLATE.format(repo_id=adapter_info["modelId"], revision="main", filename="adapter_config.json")
        )
        if response.status_code != 200:
            logger.warning(f"[{adapter_info['modelId']}] Could not load adapter_config.json.")
            continue
        adapter_config = response.json()
        # Get README.md
        response = requests.get(
            HF_CO_URL_TEMPLATE.format(repo_id=adapter_info["modelId"], revision="main", filename="README.md")
        )
        if response.status_code != 200:
            logger.warning(f"[{adapter_info['modelId']}] Could not load README.md.")
            continue
        description = response.text.split("---")[-1].strip()

        yield {
            "source": "hf",
            "groupname": groupname,
            "filename": filename,
            "task": task,
            "subtask": subtask,
            "model_type": adapter_config.get("model_type"),
            "model_name": adapter_config.get("model_name"),
            "model_class": adapter_config.get("model_class"),
            "prediction_head": _has_sibling(adapter_info, "head_config.json"),
            "config_string": json.dumps(adapter_config["config"]),
            "config_non_linearity": adapter_config["config"].get("non_linearity"),
            "config_reduction_factor": adapter_config["config"].get("reduction_factor"),
            "description": description,
            # ... others not available ...
            "last_update": last_update,
            "default_version": "main",
        }


def check_last_modified(cache_file):
    # read last modification time from cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_last_modified = datetime.strptime(f.read(), "%Y-%m-%dT%H:%M:%S")
    else:
        cached_last_modified = datetime.fromtimestamp(0)

    has_new_modifications = False
    for adapter_info in get_adapters():
        adapter_last_modified = datetime.strptime(adapter_info["lastModified"].split(".")[0], "%Y-%m-%dT%H:%M:%S")
        if adapter_last_modified > cached_last_modified:
            has_new_modifications = True
            break

    # write current time to cache
    with open(cache_file, "w") as f:
        f.write(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"))

    return has_new_modifications


# When calling as script, returns 0 if there have been modifications and 1 otherwise.
if __name__ == "__main__":
    import sys

    cache_file = sys.argv[1]

    if check_last_modified(cache_file):
        exit(0)
    else:
        exit(1)
