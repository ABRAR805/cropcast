from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
import sqlite3
import random



app = Flask(__name__)

# Load trained models and label encoders
MODEL_PATH = "models/crop_prediction_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("\u274C ERROR: Model files not found! Train the model first using train_model.py.")
    exit(1)

with open(MODEL_PATH, "rb") as f:
    crop_model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoders = pickle.load(f)

DATABASE = "database.db"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crops_for_sale (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                price REAL NOT NULL CHECK(price > 0)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vendor_requirements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                max_price REAL NOT NULL CHECK(max_price > 0)
            )
        """)
        conn.commit()

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/farmer")
def farmer():
    return render_template("farmer.html")

@app.route("/vendor")
def vendor():
    return render_template("vendor.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    import pandas as pd
    if request.method == "POST":
        try:
            print("🚀 Form received!")  # Debugging

            # Collect form inputs
            land = float(request.form["land"])
            water = request.form["water"]
            soil = request.form["soil"]
            state = request.form["state"]
            district = request.form["district"]
            country = request.form["country"]

            print(f"🌱 Inputs: Land={land}, Water={water}, Soil={soil}, State={state}, District={district}")

            # Encode categorical values safely
            def safe_encode(column, value):
                if value in label_encoders[column].classes_:
                    return label_encoders[column].transform([value])[0]
                return 0

            water_encoded = safe_encode("water", water)
            soil_encoded = safe_encode("soil", soil)
            state_encoded = safe_encode("state", state)
            district_encoded = safe_encode("district", district)
            country_encoded = safe_encode("country", country)

            # Prepare input array with correct column names
            feature_columns = ["land", "water", "soil", "state", "district", "country"]
            X_input = pd.DataFrame([[land, water_encoded, soil_encoded, state_encoded, district_encoded, country_encoded]], columns=feature_columns)

            # Predict crops
            crop_predictions = crop_model.predict(X_input).round().astype(int)[0]
            print(f"✅ Predicted Crops: {crop_predictions}")

            # Decode crop predictions safely
            def safe_decode(column, value):
                if 0 <= value < len(label_encoders[column].classes_):
                    return label_encoders[column].inverse_transform([value])[0]
                return "Unknown"

            predicted_crops = [
                safe_decode("crop1", crop_predictions[0]),
                safe_decode("crop2", crop_predictions[1]),
                safe_decode("crop3", crop_predictions[2]),
            ]
            print(f"📊 Decoded Crops: {predicted_crops}")

            # Generate Crop Plans
            def generate_plan(land, crops, risk_factor):
                allocation = [0.4, 0.4, 0.2]  # 40%-40%-20% land split
                costs = [land * alloc * 5000 for alloc in allocation]  # Cost per acre ₹5000
                earnings = [cost * 2.0 for cost in costs]  # Estimated earnings: 2x cost
                total_cost = sum(costs)
                total_earning = sum(earnings)
                net_profit = total_earning - total_cost
                adjusted_profit = net_profit - (net_profit * risk_factor / 100)

                return {
                    "crops": crops,
                    "land": [round(land * alloc, 1) for alloc in allocation],
                    "costs": [round(c, 2) for c in costs],
                    "earnings": [round(e, 2) for e in earnings],
                    "total_cost": round(total_cost, 2),
                    "total_earning": round(total_earning, 2),
                    "net_profit": round(net_profit, 2),
                    "adjusted_profit": round(adjusted_profit, 2),
                    "risk": risk_factor
                }

            plans = [
                generate_plan(land, predicted_crops, random.randint(10, 30)),
                generate_plan(land, [predicted_crops[1], predicted_crops[2], "Sunflower"], random.randint(15, 35)),
                generate_plan(land, [predicted_crops[0], predicted_crops[1], predicted_crops[2]], random.randint(5, 25))
            ]

            print(f"📋 Generated Plans: {plans}")

            return render_template("predict.html", plans=plans, land=land)

        except Exception as e:
            print("❌ Error in prediction:", e)
            return render_template("predict.html", error=f"❌ Error: {e}")

    print("🔍 GET request received")
    return render_template("predict.html") 
@app.route("/sell", methods=["GET", "POST"])
def sell():
    if request.method == "POST":
        farmer_name = request.form.get("farmer_name")  # Use .get() to avoid KeyError
        crop_name = request.form.get("crop_name")
        quantity = request.form.get("quantity", type=int)
        price = request.form.get("price", type=float)

        if not farmer_name or not crop_name or quantity is None or price is None:
            return "Missing required fields", 400  # Handle missing input

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO crops_for_sale (farmer_name, crop_name, quantity, price) VALUES (?, ?, ?, ?)",
                (farmer_name, crop_name, quantity, price)
            )
            conn.commit()

        return redirect(url_for("view_crops"))

    # Fetch vendor requirements to show on the sell page
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT vendor_name, crop_name, quantity, max_price FROM vendor_requirements")
        vendor_requirements = cursor.fetchall()

    # Convert fetched data into a list of dictionaries
    requirements_list = [
        {"vendor_name": row[0], "crop_name": row[1], "quantity": row[2], "max_price": row[3]}
        for row in vendor_requirements
    ]

    return render_template("sell.html", vendor_requirements=requirements_list)


@app.route("/view_crops")
def view_crops():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT farmer_name, crop_name, quantity, price FROM crops_for_sale")
        crops = cursor.fetchall()

    # Convert fetched data into a list of dictionaries for easy template access
    crop_list = [
        {"farmer_name": row[0], "name": row[1], "quantity": row[2], "price": row[3]}
        for row in crops
    ]

    return render_template("view_crops.html", crops=crop_list)

@app.route("/post_requirements", methods=["GET", "POST"])
def post_requirements():
    if request.method == "POST":
        print("🚀 Form Data Received:", request.form)  # Debugging output

        if "vendor_name" not in request.form:
            print("❌ ERROR: 'vendor_name' is missing from form submission!")
            return "Error: 'vendor_name' field is missing in form submission.", 400

        vendor_name = request.form["vendor_name"]
        crop_name = request.form["crop_name"]
        quantity = int(request.form["quantity"])
        max_price = float(request.form["max_price"])

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO vendor_requirements (vendor_name, crop_name, quantity, max_price) VALUES (?, ?, ?, ?)",
                           (vendor_name, crop_name, quantity, max_price))
            conn.commit()

        return redirect(url_for("vendor"))

    return render_template("post_requirements.html")

@app.route("/view_requirements")
def view_requirements():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT vendor_name, crop_name, quantity, max_price FROM vendor_requirements")
        requirements = cursor.fetchall()

    return render_template("vendor.html", requirements=requirements)
import sqlite3

def create_tables():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Table for real-time market prices (Optional)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT,
            region TEXT,
            price REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table to store farmer crop data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farmer_crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            name TEXT,
            crop TEXT,
            location TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Run the function to create/update the database
create_tables()

@app.route('/market_prices')
def market_prices():
    return render_template('market_prices.html')

@app.route('/farmer_groups')
def farmer_groups():
    return render_template('farmer_groups.html')

if __name__ == "__main__":
    print(app.url_map)

    app.run(debug=True)
