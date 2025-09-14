import streamlit as st
import joblib
import pandas as pd
import logging
import time
from io import StringIO

st.set_page_config(page_title="Student GPA Predictor", layout="wide")

# Buffer de logs en memoria
log_buffer = StringIO()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(log_buffer)]
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        st.error("No se pudo cargar el modelo. Verifica el archivo best_model.pkl")
        return None

model = load_model()

st.title("Student GPA Predictor")
logger.info("Aplicación GPA Predictor iniciada")

view = st.sidebar.selectbox("Selecciona la vista", ["Estudiante", "Coordinador", "Logs"])
st.header(f"Vista: {view}")

if view in ["Estudiante", "Coordinador"]:
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
        Tutoring = 1 if st.checkbox("👩‍🏫 Tutoría") else 0
        Extracurricular = 1 if st.checkbox("🎭 Actividades Extracurriculares") else 0
        Sports = 1 if st.checkbox("⚽ Deportes") else 0
        Music = 1 if st.checkbox("🎵 Música") else 0
        Volunteering = 1 if st.checkbox("🤝 Voluntariado") else 0

    if st.button("Calcular GPA") and model is not None:
        start_time = time.time()
        try:
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
            df_input = pd.get_dummies(df_input, columns=['ParentalEducation'], drop_first=True)
            model_columns = model.feature_names_in_
            for col in model_columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[model_columns]

            pred_gpa = round(model.predict(df_input)[0], 2)
            latency = time.time() - start_time
            logger.info(f"Predicción realizada | GPA={pred_gpa} | Vista={view} | Latencia={latency:.3f}s")

            st.metric("GPA Predicho", f"{pred_gpa:.2f}")

        except Exception as e:
            logger.error(f"Error en la predicción: {e}")
            st.error("Ocurrió un error al realizar la predicción.")

if view == "Logs":
    st.subheader("Logs en tiempo real")
    st.text(log_buffer.getvalue())
