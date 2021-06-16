import logging

from .hf_hub import get_datasets, build_adapter_entries, convert_hf_dataset_to_subtask
from .models import db, Adapter, Subtask, Task


logger = logging.getLogger(__name__)


def pull_hf_hub_entries():
    # add the fallback task for all HF adapters
    fallback_task = Task(task="other", displayname="[ Uncategorized/ Other ]", task_type="text_task")
    db.session.add(fallback_task)
    # get list of datasets from HF hub
    hf_datasets = get_datasets()
    # add adapters from HF hub
    for adapter_data, meta_data in build_adapter_entries():
        if adapter_data["task"] is not None:
            adapter_obj = Adapter(**adapter_data)
            db.session.add(adapter_obj)
        elif meta_data["hf_dataset"] in hf_datasets:
            dataset_info = hf_datasets[meta_data["hf_dataset"]]
            subtask_data = convert_hf_dataset_to_subtask(dataset_info)
            if subtask_data is not None:
                adapter_data["task"] = subtask_data["task"]
                adapter_data["subtask"] = subtask_data["subtask"]
                adapter_obj = Adapter(**adapter_data)
                db.session.add(adapter_obj)
                # a subtask with the inferred id might exist, in that case we assume it's the same task
                if not Subtask.query.get((subtask_data["task"], subtask_data["subtask"])):
                    subtask_obj = Subtask(**subtask_data)
                    db.session.add(subtask_obj)
            else:
                logger.warning("Could not convert HF dataset '%s' to a subtask", meta_data["hf_dataset"])
