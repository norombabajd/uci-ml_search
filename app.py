from flask import Flask
from apis import api
from waitress import serve

app = Flask(__name__)
api.init_app(app)
app.run(debug=True, port=9494, use_reloader=False)
#serve(app, host='0.0.0.0', port=9494)