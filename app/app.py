from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('models/xgb_model.sav')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        product_category = request.form['product_category']
        cpu = float(request.form['cpu'])
        irv = float(request.form['irv'])
        supplier_country = request.form['supplier_country']
        month = int(request.form['month'])
        day = int(request.form['day'])
        day_week = int(request.form['day_week'])

        # Create DataFrame for model input
        data_dict = {
            'date_month': month,
            'date_day': day,
            'date_day_of_week': day_week,
            'Cost Price Per Unit': cpu,
            'Item Retail Value': irv,
            'Product Category': product_category,
            'Supplier Country': supplier_country
        }
        X_input = pd.DataFrame(data_dict, index=[0])

        # Predict
        prediction = model.predict(X_input)
        
        # Convert prediction to a readable format if necessary
        prediction_result = round(prediction[0], 2)

        return render_template('index.html', prediction=prediction_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
