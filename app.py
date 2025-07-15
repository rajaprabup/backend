import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# API Key (use env var in production)
VALID_API_KEY = os.environ.get("API_KEY", "prp-secret-key")

# =========================
# API Key Authentication
# =========================
@app.before_request
def check_api_key():
    api_key = request.headers.get("x-api-key")  # More secure than query param
    if api_key != VALID_API_KEY:
        return jsonify({"error": "Forbidden - Invalid or missing API Key"}), 403
    
# Load the trained model (includes preprocessor + classifier)
MODEL_PATH = "shipping_return_prediction_model_v1_0.joblib"
model = joblib.load(MODEL_PATH)

# Extract preprocessor and model
preprocessor = model.named_steps['columntransformer']
classifier = model.named_steps['xgbclassifier']

# ===============================
# Initialize SHAP explainer once
# ===============================
# explainer = shap.TreeExplainer(classifier)

# Define a route for the home page
@app.get('/')
def home():
    return "🚚 Shipping Return Prediction API is live!"

# Define an endpoint to predict shipping return for a single store product
@app.post('/v1/shippingreturn')
def predict_shipping_return():
    # Get JSON data from the request
    product_data = request.get_json()
    try:

      # Extract relevant product features from the input data
      sample = {
          'ServiceType': product_data['ServiceType'],
          'FragilePerishable': product_data['FragilePerishable'],
          'Value': product_data['Value'],
          'Weight': product_data['Weight'],
          'InsuranceCoverage': product_data['InsuranceCoverage'],
          'ShipperCity': product_data['ShipperCity'],
          'ConsigneeCity': product_data['ConsigneeCity'],
          'ConsigneeCountry': product_data['ConsigneeCountry']
      }

      # Convert the extracted data into a DataFrame
      input_data = pd.DataFrame([sample])

      # Make prediction (get shipping return)
      prediction = model.predict(input_data).tolist()[0]
      prediction_proba = model.predict_proba(input_data)[0][1]

      # Map prediction result to a human-readable label
      prediction_label = "True" if prediction == 1 else "False"

      # SHAP processing
      # input_transformed = preprocessor.transform(input_data)
      # if hasattr(input_transformed, "toarray"):
      #     input_transformed = input_transformed.toarray()

      # shap_values = explainer(input_transformed)
      # shap_vals = shap_values.values[0]

      # feature_names = preprocessor.get_feature_names_out()
      # shap_impact = list(zip(feature_names, shap_vals))
      # top_features = sorted(shap_impact, key=lambda x: abs(x[1]), reverse=True)[:3]
      # top_feature_list = [{'feature': k, 'impact': round(float(v), 4)} for k, v in top_features]

      # Return the prediction as a JSON response
      return jsonify({
          'prediction': prediction,
          'Prediction_Label': prediction_label,
          'Prediction_Probability': format(prediction_proba, '.2f')
          # 'Top_Influencing_Features': top_feature_list
          })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define an endpoint to predict shipping return for a batch of products
@app.post('/v1/shippingreturnbatch')
def predict_shipping_return_batch():
  try:
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_shipping_returns = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_revenues = [round(float(shipping_return), 2) for shipping_return in predicted_shipping_returns]

    # Create a dictionary of predictions with property IDs as keys
    product_ids = input_data['Product_Id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(product_ids, predicted_revenues))  # Use actual prices

    return output_dict

  except Exception as e:
    return jsonify({"error": str(e)}), 500

# =========================
# Entry Point (Local Only)
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
