from itertools import groupby

from flask import Blueprint
from flask import render_template
from flask_flatpages import FlatPages

from .models import Adapter, AdapterType, Model, Task, Subtask

bp = Blueprint('main', __name__)

pages = FlatPages()


@bp.route('/')
def index():
    return render_template('index.html', posts=[])


@bp.route('/explore/')
@bp.route('/explore/<task_type>/')
def explore_tasks(task_type='text_task'):
    tasks = Task.query.filter(
        Task.task_type == task_type if task_type else True
    ).order_by(Task.displayname).all()
    subtask_query = Subtask.query.filter(
        Subtask.task_type == task_type if task_type else True
    ).order_by(Subtask.task)
    task_type_ref = AdapterType.query.get(task_type)
    subtasks = {k: list(g) for k, g in groupby(subtask_query, lambda t: t.task)}
    return render_template(
        'explore_tasks.html',
        tasks=tasks, subtasks=subtasks, task_type=task_type_ref
    )


@bp.route('/explore/<task>/<subtask>/')
@bp.route('/explore/<task>/<subtask>/<model_type>/')
@bp.route('/explore/<task>/<subtask>/<model_type>/<model>/')
def explore_adapters(task, subtask, model_type=None, model=None):
    all_model_types = [m.model_type for m in Model.query.distinct(Model.model_type)]
    all_models = Model.query.filter(Model.model_type == model_type if model_type else True)
    items = Adapter.query.filter(
        Adapter.task == task,
        Adapter.subtask == subtask,
        Adapter.model_type == model_type if model_type else True,
        Adapter.model_name == model if model else True
    ).all()
    subtask_obj = Subtask.query.get((task, subtask))
    return render_template(
        'explore_adapters.html',
        subtask=subtask_obj,
        model_type=model_type,
        all_model_types=all_model_types,
        model=model,
        all_models=all_models,
        items=items
    )


@bp.route('/adapters/<groupname>/<filename>/')
def adapter_details(groupname, filename):
    adapter = Adapter.query.get((groupname, filename))
    return render_template('adapter.html', adapter=adapter)


@bp.app_errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404


@bp.app_errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500
