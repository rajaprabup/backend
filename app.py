from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "World")
    return jsonify({"message": f"Hello, {name}!"})

# Required for local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
