import os
import time
import re
import logging
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import Counter

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("gpa_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = RotatingFileHandler(
            str(LOG_FILE), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = setup_logger()

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    logger.info("Model loaded: best_model.pkl")
    return model

model = load_model()

st.set_page_config(page_title="🎓 Student GPA Predictor", layout="wide")
st.title("🎓 Student GPA Predictor")

view = st.sidebar.selectbox("Selecciona la vista", ["Estudiante", "Coordinador"])
st.header(f"Vista: {view}")

def get_usage_stats(log_path=str(LOG_FILE)):
    total, latencies, gpas = 0, [], []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Prediction made:" in line:
                    total += 1
                    # Extraer tiempo
                    m = re.search(r"in ([0-9.]+) seconds", line)
                    if m:
                        latencies.append(float(m.group(1)))
                    # Extraer GPA
                    g = re.search(r"Prediction made: ([0-9.]+)", line)
                    if g:
                        gpas.append(float(g.group(1)))
    except FileNotFoundError:
        return {
            "total": 0,
            "avg_lat": 0.0,
            "dist": {}
        }

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    # Distribución A–F
    dist = Counter()
    for gpa in gpas:
        if gpa >= 3.5:
            dist["A"] += 1
        elif gpa >= 3.0:
            dist["B"] += 1
        elif gpa >= 2.5:
            dist["C"] += 1
        elif gpa >= 2.0:
            dist["D"] += 1
        else:
            dist["F"] += 1

    return {
        "total": total,
        "avg_lat": avg_lat,
        "dist": dist
    }

with st.sidebar:
    # ----- Estadísticas -----
    stats = get_usage_stats()
    st.divider()
    st.subheader("📊 Estadísticas de uso")
    st.write(f"🧮 Total de predicciones: **{stats['total']}**")
    st.write(f"⏱️ Tiempo promedio: **{stats['avg_lat']:.3f} s**")

    if stats["dist"]:
        st.markdown("### 🏆 Distribución de calificaciones")
        total_preds = stats["total"] or 1
        for grade in ["A", "B", "C", "D", "F"]:
            count = stats["dist"].get(grade, 0)
            pct = (count / total_preds) * 100
            st.write(f"- {grade}: {count} ({pct:.1f}%)")

col1, col2 = st.columns([1,1])

with col1:
    Age = st.number_input("🎂 Edad", min_value=15, max_value=18, value=18)
    StudyTimeWeekly = st.number_input("📚 Horas Estudio/Semana", min_value=0, max_value=20, value=5)
    Absences = st.number_input("🚪 Ausencias", min_value=0, max_value=30, value=2)
    ParentalSupport = st.selectbox(
        "👨‍👩‍👦 Apoyo Parental",
        options=[0,1,2,3,4],
        format_func=lambda x: {0:"Ninguno",1:"Bajo",2:"Moderado",3:"Alto",4:"Muy alto"}[x]
    )
    ParentalEducation = st.selectbox(
        "🎓 Educación de los padres",
        options=["HighSchool","Bachelor","Master","PhD"]
    )

with col2:
    Tutoring = 1 if st.checkbox("👩‍🏫 Tutoría", value=False) else 0
    Extracurricular = 1 if st.checkbox("🎭 Actividades Extracurriculares", value=False) else 0
    Sports = 1 if st.checkbox("⚽ Deportes", value=False) else 0
    Music = 1 if st.checkbox("🎵 Música", value=False) else 0
    Volunteering = 1 if st.checkbox("🤝 Voluntariado", value=False) else 0

if st.button("📌 Calcular GPA"):
    # Variables base
    input_dict = {
        'Age':[Age],
        'StudyTimeWeekly':[StudyTimeWeekly],
        'Absences':[Absences],
        'ParentalSupport':[ParentalSupport],
        'Tutoring':[Tutoring],
        'Extracurricular':[Extracurricular],
        'Sports':[Sports],
        'Music':[Music],
        'Volunteering':[Volunteering],
        'ParentalEducation':[ParentalEducation]
    }

    df_input = pd.DataFrame(input_dict)

    # Aplicar get_dummies como en el entrenamiento
    df_input = pd.get_dummies(df_input, columns=['ParentalEducation'], drop_first=True)

    # Alinear columnas con el modelo entrenado
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[model_columns]
    if StudyTimeWeekly == 0:
        logger.warning("Estudiante con 0 horas de estudio/semana (dato atípico).")
    if Absences > 20:
        logger.warning(f"Estudiante con ausencias altas: {Absences}")

    # Predicción
    start = time.time()
    try:
        pred_gpa = round(float(model.predict(df_input)[0]), 2)
        latency = time.time() - start

        if pred_gpa < 2.0:
            logger.warning("Predicción de GPA muy baja (%.2f)", pred_gpa)

        logger.info(
            "Prediction made: %.2f GPA in %.3f seconds | features=%s",
            pred_gpa,
            latency,
            df_input.to_dict("records")[0],
        )
    except Exception:
        logger.exception("Falló la predicción")
        st.error("Ocurrió un error haciendo la predicción.")
        st.stop()
    
    st.metric("🎯 GPA Predicho", f"{pred_gpa:.2f}")

    # Clasificación por letra
    if pred_gpa >= 3.5: grade = "A"; color="green"
    elif pred_gpa >= 3.0: grade="B"; color="blue"
    elif pred_gpa >= 2.5: grade="C"; color="orange"
    elif pred_gpa >= 2.0: grade="D"; color="red"
    else: grade="F"; color="purple"

    st.markdown(
        f"""
        <div style="text-align:center; font-size:36px; font-weight:bold;">
            <span style="color:{'green' if grade=='A' else '#ccc'};">A</span>&nbsp;&nbsp;
            <span style="color:{'blue' if grade=='B' else '#ccc'};">B</span>&nbsp;&nbsp;
            <span style="color:{'orange' if grade=='C' else '#ccc'};">C</span>&nbsp;&nbsp;
            <span style="color:{'red' if grade=='D' else '#ccc'};">D</span>&nbsp;&nbsp;
            <span style="color:{'purple' if grade=='F' else '#ccc'};">F</span>
        </div>
        <div style="text-align:center; font-size:64px; font-weight:bold; color:{color}; margin-top:10px;">
            {grade}
        </div>
        """, unsafe_allow_html=True
    )

    # Recomendaciones según vista
    if view=="Estudiante":
        st.subheader("💡 Consejos motivacionales y recomendaciones")
        if pred_gpa < 3.0:
            st.success("¡Puedes mejorar! Aquí algunas acciones para aumentar tu GPA:")
            st.write("- Incrementa gradualmente tus horas de estudio semanales.")
            st.write("- Participa en tutorías para reforzar tus conocimientos.")
            st.write("- Reduce ausencias y mantén constancia en clases.")
            st.write("- Únete a actividades extracurriculares que te motiven.")
        else:
            st.success("¡Excelente desempeño! Mantén estos hábitos:")
            st.write("- Continúa con tu dedicación al estudio.")
            st.write("- Participa en actividades que disfrutes y te inspiren.")
            st.write("- Comparte tus estrategias exitosas con compañeros.")

    if view=="Coordinador":
        st.subheader("📌 Análisis para Coordinadores")
        if pred_gpa < 3.0:
            st.warning("Estudiante identificado con riesgo académico.")
            st.write("- Considerar seguimiento personalizado y tutorías.")
            st.write("- Ofrecer recursos motivacionales y programas de apoyo.")
        else:
            st.info("Estudiante con desempeño adecuado. Mantener seguimiento motivacional.")
