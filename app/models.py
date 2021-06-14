from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy.orm import backref


db = SQLAlchemy()


class Architecture(db.Model):
    __tablename__ = "architectures"

    name = db.Column(db.String(30), primary_key=True)
    displayname = db.Column(db.String(80), nullable=False)
    description = db.Column(db.Text)
    citation = db.Column(db.Text)
    config = db.Column(db.Text)
    config_non_linearity = db.Column(db.Text)
    config_reduction_factor = db.Column(db.Integer)

    def __repr__(self):
        return self.name


class Task(db.Model):
    __tablename__ = "tasks"
    
    task = db.Column(db.String(30), primary_key=True)
    displayname = db.Column(db.Text)
    description = db.Column(db.Text)

    task_type = db.Column(db.String(20), db.ForeignKey('adapter_types.id'))
    task_type_ref = db.relationship('AdapterType', backref='tasks')

    def __repr__(self):
        return self.task


class Subtask(db.Model):
    __tablename__ = "subtasks"

    task = db.Column(db.String(30), db.ForeignKey('tasks.task'), primary_key=True)
    subtask = db.Column(db.String(30), primary_key=True)
    displayname = db.Column(db.String(80))
    description = db.Column(db.Text)
    url = db.Column(db.String(150))
    citation = db.Column(db.Text)

    task_ref = db.relationship('Task', backref='subtasks')

    task_type = db.Column(db.String(20), db.ForeignKey('adapter_types.id'))
    task_type_ref = db.relationship('AdapterType', backref='subtasks')

    language = db.Column(db.String(30))

    def __repr__(self):
        return '{}/{}'.format(self.task, self.subtask)


class SubtaskMetric(db.Model):
    __tablename__ = "metric"

    task = db.Column(db.String(30), primary_key=True)
    subtask = db.Column(db.String(30), primary_key=True)
    name = db.Column(db.Text, primary_key=True)
    higher_is_better = db.Column(db.Boolean, nullable=False)

    subtask_ref = db.relationship('Subtask', backref=backref('metric', uselist=False))

    __table_args__ = (
        ForeignKeyConstraint([task, subtask], [Subtask.task, Subtask.subtask]),
    )

    def __repr__(self):
        return str(self.name).title()


class AdapterType(db.Model):
    __tablename__ = "adapter_types"

    id = db.Column(db.String(20), primary_key=True)
    displayname = db.Column(db.String(20), unique=True)
    description = db.Column(db.Text)

    def __repr__(self):
        return self.id


class Model(db.Model):
    __tablename__ = "models"

    name = db.Column(db.Text, primary_key=True)
    model_type = db.Column(db.String(30), nullable=False)

    def __repr__(self):
        return self.name


class Adapter(db.Model):
    __tablename__ = "adapters"
    # source
    source = db.Column(db.String(30), default="ah")
    # name
    groupname = db.Column(db.String(30), primary_key=True)
    filename = db.Column(db.String(80), primary_key=True)

    # type & task
    task = db.Column(db.String(30), nullable=False)
    task_ref = db.relationship('Task', viewonly=True)
    subtask = db.Column(db.String(30), nullable=False)
    subtask_ref = db.relationship('Subtask', backref='adapters')

    # model & config
    model_type = db.Column(db.String(30), nullable=False)
    model_name = db.Column(db.Text, nullable=False)
    model_class = db.Column(db.Text)
    prediction_head = db.Column(db.Boolean)
    config = db.Column(db.Text, db.ForeignKey('architectures.name'))
    config_ref = db.relationship('Architecture', backref='adapters')
    config_non_linearity = db.Column(db.Text)
    config_reduction_factor = db.Column(db.Integer)

    # meta
    description = db.Column(db.Text)
    author = db.Column(db.String(80))
    email = db.Column(db.String(80))
    url = db.Column(db.String(150))
    github = db.Column(db.String(30))
    twitter = db.Column(db.String(30))
    citation = db.Column(db.Text)
    
    # files
    default_version = db.Column(db.String(10), nullable=False)

    last_update = db.Column(db.DateTime)

    __table_args__ = (
        ForeignKeyConstraint([task], [Task.task]),
        ForeignKeyConstraint([task, subtask], [Subtask.task, Subtask.subtask]),
    )

    def __repr__(self):
        return '{}/{}'.format(self.groupname, self.filename)


class AdapterDependency(db.Model):
    __tablename__ = "dependencies"

    adapter_groupname = db.Column(db.String(30), primary_key=True)
    adapter_filename = db.Column(db.String(24), primary_key=True)
    key = db.Column(db.String(100), primary_key=True)
    description = db.Column(db.Text)

    adapter = db.relationship('Adapter', backref='dependencies')

    __table_args__ = (
        ForeignKeyConstraint([adapter_groupname, adapter_filename], [Adapter.groupname, Adapter.filename]),
    )

    def __repr__(self):
        return '<AdapterDependency {}/{} {}>'.format(self.adapter_groupname, self.adapter_filename, self.key)


class AdapterFile(db.Model):
    __tablename__ = "files"

    adapter_groupname = db.Column(db.String(30), primary_key=True)
    adapter_filename = db.Column(db.String(24), primary_key=True)
    version = db.Column(db.String(10), primary_key=True)
    url = db.Column(db.String(200), nullable=False)
    sha1 = db.Column(db.String(40))
    sha256 = db.Column(db.String(64))
    description = db.Column(db.Text)
    score = db.Column(db.Float)

    adapter = db.relationship('Adapter', backref='files')

    __table_args__ = (
        ForeignKeyConstraint([adapter_groupname, adapter_filename], [Adapter.groupname, Adapter.filename]),
    )

    def __repr__(self):
        return '<AdapterFile {}/{} {}>'.format(self.adapter_groupname, self.adapter_id, self.version)
