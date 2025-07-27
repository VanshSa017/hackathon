
from flask import Flask, request, render_template, make_response
import numpy as np
import pandas as pd
import joblib
import os
from io import BytesIO
from xhtml2pdf import pisa
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# Load model and transformers
model = joblib.load("best_tuned_fusion_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

NUMERIC_COLS = ['Age', 'Work Experience', 'Family  Size']
ORDINAL_COLS = {'Family Expenses': {'Low': 0, 'Average': 1, 'High': 2}}
NOMINAL_COLS = {
    'Sex': ['Female', 'Male'],
    'Bachelor': ['No', 'Yes'],
    'Graduated': ['No', 'Yes'],
    'Career': ['Private', 'Public'],
    'Variable': ['No', 'Yes']
}
MODEL_FEATURES = model.feature_names_in_

#  Segment Insight
def generate_insight(segment):
    return {
        'A': "Offer exclusive premium packages and loyalty benefits to Segment A customers.",
        'B': "Increase engagement through personalized recommendations and value-based offers.",
        'C': "Introduce educational content and budget-friendly plans to grow interest.",
        'D': "Focus on awareness campaigns and discount-driven acquisition strategies."
    }.get(segment, "Consider customer-specific strategies for better retention.")

#  User-Based Insights
def generate_insights(df, pred_label):
    insights = []
    age = df['Age'].values[0]
    family_exp = df['Family Expenses'].values[0]

    if age < 25:
        insights.append("📊 Younger age group — potential for digital services and lifestyle-focused products.")
    elif age > 50:
        insights.append("👨‍💼 Older segment — may value retirement, healthcare, and wealth management plans.")
    else:
        insights.append("🏡 Middle-aged group — focus on family, housing, and education-related offerings.")

    if family_exp == 0:
        insights.append("💰 Low expenses — cost-effective promotions or EMI-based schemes can work well.")
    elif family_exp == 1:
        insights.append("📈 Average expenses — customers may respond to balanced reward systems and bundled offers.")
    else:
        insights.append("🛍️ High spenders — premium loyalty programs and lifestyle partnerships are ideal.")

    return insights

#  LIME Explanation
def explain_prediction(df_input, model, scaler, feature_names, class_names):
    scaled_df = df_input.copy()
    scaled_df[NUMERIC_COLS] = scaler.transform(scaled_df[NUMERIC_COLS])
    dummy_data = np.tile(scaled_df.values[0], (10, 1))
    explainer = LimeTabularExplainer(
        training_data=dummy_data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=scaled_df.values[0],
        predict_fn=model.predict_proba
    )
    return explanation.as_html()

#  Home Page
@app.route('/')
def index():
    return render_template('form.html')

#  Prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Age': float(request.form['age']),
        'Work Experience': float(request.form['work_experience']),
        'Family  Size': float(request.form['family_size']),
        'Family Expenses': request.form['family_expenses'],
        'Sex': request.form['sex'],
        'Bachelor': request.form['bachelor'],
        'Graduated': request.form['graduated'],
        'Career': request.form['career'],
        'Variable': request.form['variable']
    }

    df = pd.DataFrame([input_data])

    for col, mapping in ORDINAL_COLS.items():
        df[col] = df[col].map(mapping)

    for col, categories in NOMINAL_COLS.items():
        for cat in categories[1:]:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df.drop(columns=col, inplace=True)

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[MODEL_FEATURES]
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    pred_encoded = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    lime_html = explain_prediction(df, model, scaler, MODEL_FEATURES, label_encoder.classes_)
    insights = [generate_insight(pred_label)] + generate_insights(df, pred_label)

    #  Static image paths from static folder
    feature_plot = "static/output_feature_dis.png"
    segment_pie_plot = "static/segment_pie_87bb0912ed3a4573a6cfc92f5c7e868c.png"
    corr_heatmap = "static/heatmap_corr_8ae6e63d8d5d4120932a107abc2a9789.png"
    expense_plot = "static/expense_bar_a758848083f140638ab114cbeafb4c43.png"

    return render_template("result.html",
                           prediction=pred_label,
                           explanation=lime_html,
                           insight=insights,
                           feature_plot=feature_plot,
                           segment_pie_plot=segment_pie_plot,
                           corr_heatmap=corr_heatmap,
                           expense_plot=expense_plot)

#  PDF Report Download
@app.route('/download_report')
def download_report():
    rendered = render_template('report_template.html',
                               prediction="Your Prediction",
                               explanation="<b>LIME HTML</b>")
    pdf = BytesIO()
    pisa.CreatePDF(rendered, dest=pdf)
    response = make_response(pdf.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    response.mimetype = 'application/pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)
