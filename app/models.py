from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKeyConstraint


db = SQLAlchemy()


class Architecture(db.Model):
    __tablename__ = "architectures"

    name = db.Column(db.String(30), primary_key=True)
    displayname = db.Column(db.String(80), nullable=False)
    description = db.Column(db.Text)
    citation = db.Column(db.Text)
    config = db.Column(db.Text)

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

    task = db.Column(db.String(30), primary_key=True)
    subtask = db.Column(db.String(30), primary_key=True)
    displayname = db.Column(db.String(80))
    description = db.Column(db.Text)
    url = db.Column(db.String(150))
    citation = db.Column(db.Text)

    task_type = db.Column(db.String(20), db.ForeignKey('adapter_types.id'))
    task_type_ref = db.relationship('AdapterType', backref='subtasks')

    def __repr__(self):
        return '{}/{}'.format(self.task, self.subtask)


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

    groupname = db.Column(db.String(30), primary_key=True)
    filename = db.Column(db.String(80), primary_key=True)

    model_name = db.Column(db.Text, nullable=False)
    config_id = db.Column(db.String(24), nullable=False)
    task = db.Column(db.String(30), nullable=False)
    subtask = db.Column(db.String(30), nullable=False)
    description = db.Column(db.Text)
    author = db.Column(db.String(80))
    email = db.Column(db.String(80))
    url = db.Column(db.String(150))
    citation = db.Column(db.Text)
    default_version = db.Column(db.String(10), nullable=False)
    # score = db.Column(db.Float)

    config = db.Column(db.Text, nullable=False)
    hidden_size = db.Column(db.Integer, nullable=False)
    model_type = db.Column(db.String(30), nullable=False)

    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    type = db.Column(db.String(20), db.ForeignKey('adapter_types.id'))
    adapter_type = db.relationship('AdapterType', backref='adapters')

    subtask_ref = db.relationship('Subtask', backref='adapters')

    __table_args__ = (
        ForeignKeyConstraint([task, subtask], [Subtask.task, Subtask.subtask]),
    )

    def __repr__(self):
        return '{}/{}'.format(self.groupname, self.filename)


class AdapterFile(db.Model):
    __tablename__ = "files"

    adapter_groupname = db.Column(db.String(30), primary_key=True)
    adapter_filename = db.Column(db.String(24), primary_key=True)
    version = db.Column(db.String(10), primary_key=True)
    url = db.Column(db.String(200), nullable=False)
    sha1 = db.Column(db.String(40))
    sha256 = db.Column(db.String(64))

    adapter = db.relationship('Adapter', backref='files')

    __table_args__ = (
        ForeignKeyConstraint([adapter_groupname, adapter_filename], [Adapter.groupname, Adapter.filename]),
    )

    def __repr__(self):
        return '<AdapterFile {}/{} {}>'.format(self.adapter_groupname, self.adapter_id, self.version)
