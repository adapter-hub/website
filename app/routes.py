from flask import Blueprint
from flask import render_template, request, url_for, flash, redirect, abort
from flask_flatpages import FlatPages
from .models import Adapter, Model, Task, Subtask
from itertools import groupby


bp = Blueprint('main', __name__)

pages = FlatPages()


@bp.route('/')
@bp.route('/index/')
def index():
    return render_template('index.html', posts=[])


@bp.route('/explore/')
@bp.route('/explore/<task_type>/')
def explore_tasks(task_type=None):
    tasks = Task.query.filter(
        Task.task_type==task_type if task_type else True
    ).all()
    subtask_query = Subtask.query.filter(
        Subtask.task_type==task_type if task_type else True
    ).order_by(Subtask.task)
    subtasks = {k: list(g) for k, g in groupby(subtask_query, lambda t: t.task)}
    return render_template(
        'explore_tasks.html',
        tasks=tasks, subtasks=subtasks, task_type=task_type
    )


@bp.route('/explore/<task>/<subtask>/')
@bp.route('/explore/<task>/<subtask>/<model_type>/')
@bp.route('/explore/<task>/<subtask>/<model_type>/<model>/')
def explore_adapters(task, subtask, model_type=None, model=None):
    models = Model.query.filter(Model.model_type==model_type if model_type else True)
    items = Adapter.query.filter(
                Adapter.task==task,
                Adapter.subtask==subtask,
                Adapter.model_type==model_type if model_type else True,
                Adapter.model_name==model if model else True
            ).all()
    subtask_obj = Subtask.query.get((task, subtask))
    return render_template(
        'explore_adapters.html',
        subtask=subtask_obj,
        model_type=model_type, model=model,
        all_models=models, items=items
    )


@bp.route('/adapters/<groupname>/<filename>/')
def adapter_details(groupname, filename):
    adapter = Adapter.query.get((groupname, filename))
    return render_template('adapter.html', adapter=adapter)

@bp.route('/upload/', methods=('GET', 'POST'))
def upload():
    page = pages.get_or_404('contributing')
    return render_template('flatpage.html', page=page, active_page='upload')


@bp.app_errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404


@bp.app_errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500
