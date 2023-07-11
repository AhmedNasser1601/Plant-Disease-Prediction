# from app import app
# from config import Config
# app.config.from_object(Config)
from app import create_app
from flask_cors import CORS

app = create_app()
CORS(app)

if __name__ == "__main__":
    app.run(debug=True)

