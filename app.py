from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# ---------------------------------------
# Load and preprocess dataset
# ---------------------------------------
df = pd.read_csv("financial_inclusion.csv")

if "user_id" in df.columns:
    df.drop(columns=["user_id"], inplace=True)

le_income = LabelEncoder()
le_employment = LabelEncoder()

df["income_level"] = le_income.fit_transform(df["income_level"])
df["employment_type"] = le_employment.fit_transform(df["employment_type"])

TARGET_COL = "loan_eligibility_flag"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(classification_report(y_test, model.predict(X_test)))


# ---------------------------------------
# Helper: Generate personalized advice
# ---------------------------------------
def generate_advice(inputs):
    advice = []

    if inputs["credit_score_proxy"] < 40:
        advice.append("📉 Your credit score seems low. Repay small debts on time to build trust with lenders.")
    if inputs["financial_literacy_score"] < 5:
        advice.append("📘 Improve your financial literacy. Try free local programs or RBI financial education resources.")
    if inputs["income_level"].lower() == "low":
        advice.append("💼 Consider skill-based side jobs or community microbusiness opportunities to boost income.")
    if not inputs["internet_availability"]:
        advice.append("🌐 Getting internet access will help you explore digital financial services and government schemes.")
    if not inputs["mobile_access"]:
        advice.append("📱 Mobile access can help manage savings and payments easily. Consider registering for a SIM or feature phone.")
    if not advice:
        advice.append("👍 You're on the right path! Maintain savings and repayment habits to become eligible soon.")

    return advice


# ---------------------------------------
# Flask Routes
# ---------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        income = request.form["income"]
        employment = request.form["employment"]
        credit = int(request.form["credit"])
        history = int(request.form["history"])
        mobile = int(request.form["mobile"])
        internet = int(request.form["internet"])
        literacy = int(request.form["literacy"])

        income_encoded = (
            le_income.transform([income])[0]
            if income in le_income.classes_
            else -1
        )
        employment_encoded = (
            le_employment.transform([employment])[0]
            if employment in le_employment.classes_
            else -1
        )

        sample = pd.DataFrame([{
            "age": age,
            "income_level": income_encoded,
            "employment_type": employment_encoded,
            "credit_score_proxy": credit,
            "previous_loan_history": history,
            "mobile_access": mobile,
            "internet_availability": internet,
            "financial_literacy_score": literacy
        }])

        pred = model.predict(sample)[0]

        if pred == 1:
            result = "✅ Eligible for Microloan"
            advice = []
        else:
            result = "❌ Not Eligible for Microloan"
            user_inputs = {
                "income_level": income,
                "credit_score_proxy": credit,
                "internet_availability": internet,
                "mobile_access": mobile,
                "financial_literacy_score": literacy,
            }
            advice = generate_advice(user_inputs)

    except Exception as e:
        result = f"Error: {str(e)}"
        advice = []

    return render_template("eligibility.html", prediction=result, advice=advice)


@app.route("/tips")
def tips():
    return render_template("tips.html")


@app.route("/eligibility")
def eligibility_page():
    return render_template("eligibility.html")


if __name__ == "__main__":
    app.run(debug=True)
