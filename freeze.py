from flask_frozen import Freezer
from app import create_app

freezer = Freezer(create_app())
freezer.run(debug=True)
