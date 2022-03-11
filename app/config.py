import os
from flask_filealchemy import ColumnMapping
from .models import (
    Architecture,
    Task,
    Subtask,
    SubtaskMetric,
    AdapterType,
    Adapter,
    AdapterFile,
    AdapterDependency,
)

basedir = os.getcwd()

class Config(object):
    ### some external urls
    HUB_URL = "https://github.com/Adapter-Hub/Hub/blob/master/adapters/"
    DOCUMENTATION_URL = "https://docs.adapterhub.ml/"
    CONTRIBUTING_URL = DOCUMENTATION_URL+"contributing.html"

    ### db
    SECRET_KEY = os.environ.get('SECRET_KEY') or "default_secret_key"
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    ### flat-pages
    FLATPAGES_ROOT = os.path.join(basedir, 'posts')
    FLATPAGES_EXTENSION = '.md'
    FLATPAGES_MARKDOWN_EXTENSIONS = [
        'codehilite',
        'fenced_code',
        'caption',
        'tables',
        'footnotes',
        'mdx_math',
    ]
    FLATPAGES_EXTENSION_CONFIGS = {
        'caption': {
            'captionNumbering': True
        },
        'mdx_math': {
            'enable_dollar_delimiter': True,
        },
    }

    ### frozen flask
    # FREEZER_BASE_URL = "http://127.0.0.1:5000/"
    FREEZER_BASE_URL = os.environ.get("FREEZER_BASE_URL") or "http://adapterhub.ml/"
    FREEZER_DESTINATION = os.path.join(basedir, 'build')
    # FREEZER_RELATIVE_URLS = True
    FREEZER_STATIC_IGNORE = [
        'node_modules',
        '.webassets-cache',
        'package*.json',
        '*.scss'
    ]
    FREEZER_IGNORE_404_NOT_FOUND = True

    ### utterances
    COMMENTS_REPO = os.environ.get("COMMENTS_REPO") or "Adapter-Hub/website"

    ### filealchemy
    FILEALCHEMY_DATA_DIR = os.path.join(basedir, 'data')
    FILEALCHEMY_MODELS = [
        Architecture,
        (Task, {'task_type': ColumnMapping.FOLDER_NAME}),
        (Subtask, {'task_type': ColumnMapping.FOLDER_NAME}),
        SubtaskMetric,
        AdapterType,
        (Adapter, {'groupname': ColumnMapping.FOLDER_NAME, 'filename': ColumnMapping.FILE_NAME}),
        AdapterFile,
        AdapterDependency,
    ]
    FILEALCHEMY_SKIP_NO_MODEL = True
    FILEALCHEMY_MAP_NESTED = True
