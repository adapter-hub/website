import json

from flask import current_app
from flask.cli import AppGroup
from flask_frozen import Freezer
from flask_filealchemy import FileAlchemy
from .models import db, Adapter, Model, Subtask


freeze_cli = AppGroup("freeze")

@freeze_cli.command('build')
def freeze_build():
    freezer = Freezer(current_app)
    for l in freezer.freeze_yield():
        print(l)


db_cli = AppGroup("db")

@db_cli.command('init')
def db_init():
    db.drop_all()
    db.create_all()
    FileAlchemy(current_app, db).load_tables()
    # fill list of models based on adapters
    models = db.session.query(Adapter.model_name, Adapter.model_type).distinct().all()
    for model_args in models:
        model_obj = Model(name=model_args[0], model_type=model_args[1])
        db.session.add(model_obj)
    # fill subtasks
    subtasks = db.session.query(Adapter.task, Adapter.subtask, Adapter.type).distinct().all()
    for subtask_args in subtasks:
        kwargs = {'task': subtask_args[0], 'subtask': subtask_args[1]}
        if not db.session.query(Subtask).filter_by(**kwargs).first():
            subtask_obj = Subtask(task_type=subtask_args[2], **kwargs)
            db.session.add(subtask_obj)
    # hack: fix all configs, this should be replaced with something better
    for adapter in db.session.query(Adapter).all():
        config = json.loads(adapter.config.replace("\'", "\""))
        adapter.config = config["using"]
        adapter.config_non_linearity = config.get("non_linearity", None)
        adapter.config_reduction_factor = config.get("reduction_factor", None)
    db.session.commit()
