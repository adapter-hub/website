from datetime import datetime
import json
import logging
import os
import requests


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


API_BASE_URL = "https://huggingface.co/api/"
HF_CO_URL_TEMPLATE = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

# TODO extend this list
HF_TASK_TO_AH_TASK_MAP = {
    "question-answering": "qa",
    "natural-language-inference": "nli",
    "translation": "mt",
    "sentiment-analysis": "sentiment",
    "sequence-modeling": "lm",
    "summarization": "sum",
    "part-of-speech-tagging": "pos",
    "named-entity-recognition": "ner",
}
HF_DATASET_TO_AH_TASK_MAP = {
    "emo": "emotion",
    "emotion": "emotion",
    "anli": "nli",
    "art": "comsense",
    "com_qa": "qa",
    "hotpot_qa": "qa",
    "quail": "rc",
    "quoref": "qa",
    "yelp_polarity": "sentiment",
}


def get_datasets():
    params = {"full": True}
    response = requests.get(API_BASE_URL + "datasets", params=params)
    response.raise_for_status()
    dataset_list = response.json()
    dataset_dict = {item["id"]: item for item in dataset_list}
    return dataset_dict


def convert_hf_dataset_to_subtask(dataset_info):
    task = None
    if dataset_info["id"] in HF_DATASET_TO_AH_TASK_MAP:
        task = HF_DATASET_TO_AH_TASK_MAP[dataset_info["id"]]
    if task is None and isinstance(dataset_info["cardData"].get("task_categories", None), list):
        for task_category in dataset_info["cardData"]["task_categories"]:
            task = HF_TASK_TO_AH_TASK_MAP.get(task_category)
            if task is not None:
                break
    if task is None and isinstance(dataset_info["cardData"].get("task_ids", None), list):
        for task_id in dataset_info["cardData"]["task_ids"]:
            task = HF_TASK_TO_AH_TASK_MAP.get(task_id)
            if task is not None:
                break
    if task is None:
        task = "other"
    subtask = dataset_info["id"]
    displayname = subtask.replace("_", " ").title()
    description = dataset_info["description"]
    citation = dataset_info["citation"]
    # TODO
    language = None
    if isinstance(dataset_info["cardData"]["languages"], list):
        language = dataset_info["cardData"]["languages"][0]
    return {
        "task": task,
        "subtask": subtask,
        "displayname": displayname,
        "description": description,
        "citation": citation,
        # TODO
        "task_type": "text_task",
        "language": language,
        "hf_datasets_id": dataset_info["id"],
    }


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
        dataset_tag = next((t for t in adapter_info["tags"] if t.startswith("dataset:")), None)
        if adapterhub_tag is not None:
            task_splits = adapterhub_tag.split(":")[-1].split("/")
            if len(task_splits) != 2:
                logger.warning(f"Invalid adapterhub tag {adapterhub_tag}")
                continue
            task, subtask = task_splits
            hf_dataset = dataset_tag.split(":")[-1] if dataset_tag else None
        elif dataset_tag is not None:
            task, subtask = None, None
            hf_dataset = dataset_tag.split(":")[-1]
        else:
            logger.warning(f"[{adapter_info['modelId']}] No adapterhub or dataset tag found.")
            continue
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
        }, {
            "hf_dataset": hf_dataset,
        }


def check_last_modified(cache_file):
    # read last modification time from cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_last_modified = datetime.strptime(f.read(), "%Y-%m-%dT%H:%M:%S")
    else:
        logger.warning("No cache file found. Re-initializing.")
        cached_last_modified = datetime.fromtimestamp(0)
    logger.info("Cached last modification date: %s", cached_last_modified)

    has_new_modifications = False
    for adapter_info in get_adapters():
        adapter_last_modified = datetime.strptime(adapter_info["lastModified"].split(".")[0], "%Y-%m-%dT%H:%M:%S")
        if adapter_last_modified > cached_last_modified:
            logger.info("Found newer modification date: %s", adapter_last_modified)
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
