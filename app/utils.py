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
            # First, try to find a matching subtask in the database
            subtasks = Subtask.query.filter_by(hf_datasets_id=meta_data["hf_dataset"]).all()
            if len(subtasks) == 1:
                adapter_data["task"] = subtasks[0].task
                adapter_data["subtask"] = subtasks[0].subtask
                adapter_obj = Adapter(**adapter_data)
                db.session.add(adapter_obj)
            # If the information is unclear or not available, try to rebuild from HF info
            else:
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
