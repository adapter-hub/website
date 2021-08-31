import os
from flask import Flask
from flaskext.markdown import Markdown
from .config import Config
from .models import db, Adapter
from .assets import assets
from .routes import bp as main_bp, blog_posts
from .cli import freeze_cli, db_cli


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    Markdown(app, extensions=["codehilite", "fenced_code", "caption"])
    # init modules
    db.init_app(app)
    assets.init_app(app)
    blog_posts.init_app(app)
    # blueprints
    app.register_blueprint(main_bp)
    # cli
    app.cli.add_command(freeze_cli)
    app.cli.add_command(db_cli)

    @app.shell_context_processor
    def make_shell_context():
        return {"db": db, "Adapter": Adapter}

    return app
