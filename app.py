from flask import Flask, request, jsonify

app = Flask(__name__)  # ✅ Define app FIRST

# Optional: Check API Key or Token before every request
@app.before_request
def check_api_key():
    api_key = request.args.get("key")
    if api_key != "prp-secret-key":
        return jsonify({"error": "Forbidden"}), 403

@app.route("/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "World")
    return jsonify({"message": f"Hello, {name}!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
