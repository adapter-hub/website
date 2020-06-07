import os
from flask_filealchemy import ColumnMapping
from .models import Architecture, Task, Subtask, AdapterType, Adapter, AdapterFile

basedir = os.getcwd()

class Config(object):
    DOCUMENTATION_URL = "http://adapter-hub.webredirect.org/docs/"

    ### db
    SECRET_KEY = os.environ.get('SECRET_KEY') or "default_secret_key"
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    ### flat-pages
    FLATPAGES_ROOT = os.path.join(basedir, 'data')
    FLATPAGES_EXTENSION = '.md'

    ### frozen flask
    # FREEZER_BASE_URL = "http://127.0.0.1:5000/"
    FREEZER_BASE_URL="http://localhost/website/"
    FREEZER_DESTINATION = os.path.join(basedir, 'build')
    # FREEZER_RELATIVE_URLS = True
    FREEZER_STATIC_IGNORE = [
        'node_modules',
        '.webassets-cache',
        'package*.json',
        '*.scss'
    ]
    FREEZER_IGNORE_404_NOT_FOUND = True

    ### filealchemy
    FILEALCHEMY_DATA_DIR = os.path.join(basedir, 'data')
    FILEALCHEMY_MODELS = [
        Architecture,
        (Task, {'task_type': ColumnMapping.FOLDER_NAME}),
        (Subtask, {'task_type': ColumnMapping.FOLDER_NAME}),
        AdapterType,
        (Adapter, {'groupname': ColumnMapping.FOLDER_NAME, 'filename': ColumnMapping.FILE_NAME}),
        AdapterFile
    ]
    FILEALCHEMY_SKIP_NO_MODEL = True
    FILEALCHEMY_MAP_NESTED = True
