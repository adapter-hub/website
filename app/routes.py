from datetime import datetime, timezone
from itertools import groupby
import json

from flask import Blueprint, abort
from flask import render_template, current_app
from flask_flatpages import FlatPages, pygments_style_defs
from feedgen.feed import FeedGenerator

from .models import Adapter, AdapterType, Model, Task, Subtask

bp = Blueprint('main', __name__)

blog_posts = FlatPages()


@bp.app_template_filter("jsondumps")
def json_dumps(o):
    return json.dumps(o, indent=2)


@bp.route('/')
def index():
    n_subtasks = Subtask.query.filter(
        Subtask.task_type == 'text_task'
    ).count()
    n_languages = Task.query.filter(
        Task.task_type == 'text_lang'
    ).count()
    n_adapters = Adapter.query.count()
    num_posts_on_home = current_app.config["NUM_POSTS_ON_HOME"]
    posts = sorted(blog_posts, key=lambda p: p["date"])[-num_posts_on_home:]
    return render_template('index.html', posts=posts, n_subtasks=n_subtasks, n_languages=n_languages,n_adapters=n_adapters)


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
    all_model_types = [m.model_type for m in Model.query.with_entities(Model.model_type).distinct()]
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
    if adapter:
        # Recreate the full adapter_config dict
        if adapter.config_ref:
            adapter_config = json.loads(adapter.config_ref.config)
            adapter_config["non_linearity"] = adapter.config_non_linearity
            adapter_config["reduction_factor"] = adapter.config_reduction_factor
        elif adapter.config_string:
            adapter_config = json.loads(adapter.config_string)
        file = current_app.config["HUB_URL"]+"/"+groupname+"/"+filename+".yaml"
        return render_template('adapter.html', adapter=adapter, adapter_config=adapter_config, file=file)
    else:
        return abort(404)


@bp.route('/blog/')
def blog():
    posts = sorted(blog_posts, key=lambda p: p["date"])
    return render_template('blog.html', posts=posts)


@bp.route('/blog/<path:path>/')
def blog_post(path):
    post = blog_posts.get_or_404(path)
    return render_template('blog_post.html', post=post)


@bp.route('/blog/atom.xml')
def blog_feed():
    feed = FeedGenerator()
    feed.title("The AdapterHub Blog")
    feed.subtitle("The latest news from AdapterHub")
    feed.id("https://adapterhub.ml/blog/")
    feed.link(href="https://adapterhub.ml/blog/")
    for post in blog_posts:
        url = "https://adapterhub.ml/blog/"+post.path
        entry = feed.add_entry()
        entry.id(url)
        entry.link(href=url)
        entry.title(post['title'])
        entry.summary(post['summary'])
        entry.content(post.html, type='html')
        entry.author([{'name': a['name']} for a in post['authors']])
        dt = post['date']
        post_time = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        entry.pubDate(post_time)
        entry.updated(post_time)
    return feed.atom_str(pretty=True), 200, {'Content-Type': 'application/atom+xml'}


@bp.route('/pygments.css')
def pygments_css():
    return pygments_style_defs("tango"), 200, {'Content-Type': 'text/css'}


@bp.route('/imprint-privacy/')
def imprint_privacy():
    return render_template('imprint_privacy.html')


@bp.route('/alps2022/')
def emnlp():
    return render_template('alps2022.html')


@bp.route('/adapters/')
def adapters_lib():
    return render_template('adapters_lib.html')


@bp.app_errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404


@bp.app_errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500
