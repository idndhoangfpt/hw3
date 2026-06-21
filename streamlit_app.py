from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN & DANH SÁCH MÔ HÌNH (GIỮ NGUYÊN TỪ FILE GỐC CỦA BẠN)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "artifacts" / "models"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"

MODEL_ORDER = [
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "VotingClassifier",
]

MODEL_LABELS = {
    "DecisionTreeClassifier": "Decision Tree",
    "KNeighborsClassifier": "K-NN",
    "GaussianNB": "Naive Bayes",
    "RandomForestClassifier": "Random Forest",
    "AdaBoostClassifier": "AdaBoost",
    "GradientBoostingClassifier": "Gradient Boosting",
    "XGBClassifier": "XGBoost",
    "VotingClassifier": "Ensemble (Soft Voting)",
}

FEATURE_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

EXAMPLE_PATIENTS = {
    "Example 1 (No Heart Disease)": {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 1,
        "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 3, "actual_target": 0,
    },
    "Example 2 (Heart Disease)": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6, "actual_target": 1,
    },
}

# ==============================================================================
# LOGIC XỬ LÝ DỮ LIỆU & LOAD ARTIFACTS
# ==============================================================================
def scale_continuous(value: float, minimum: float, maximum: float) -> float:
    clipped = min(max(value, minimum), maximum)
    return (clipped - minimum) / (maximum - minimum)

def encode_patient_input(patient_input: dict[str, float | int]) -> pd.DataFrame:
    encoded = {
        "age": scale_continuous(float(patient_input["age"]), 29, 77),
        "trestbps": scale_continuous(float(patient_input["trestbps"]), 94, 200),
        "chol": scale_continuous(float(patient_input["chol"]), 126, 564),
        "thalach": scale_continuous(float(patient_input["thalach"]), 71, 202),
        "oldpeak": scale_continuous(float(patient_input["oldpeak"]), 0.0, 6.2),
        "sex": float(patient_input["sex"]),
        "cp": (int(patient_input["cp"]) - 1) / 3,
        "fbs": float(patient_input["fbs"]),
        "restecg": int(patient_input["restecg"]) / 2,
        "exang": float(patient_input["exang"]),
        "slope": (int(patient_input["slope"]) - 1) / 2,
        "ca": int(patient_input["ca"]) / 3,
        "thal": {3: 0.0, 6: 0.5, 7: 1.0}[int(patient_input["thal"])],
    }
    return pd.DataFrame([encoded], columns=FEATURE_COLUMNS)

@st.cache_resource
def load_models(_version: tuple[float, ...]) -> dict[str, object]:
    return {
        model_name: joblib.load(MODELS_DIR / f"{model_name}.joblib")
        for model_name in MODEL_ORDER
    }

@st.cache_data
def load_reports(_version: tuple[float, ...]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        report_path = REPORTS_DIR / f"{model_name}_metrics.json"
        if report_path.exists():
            reports[model_name] = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            reports[model_name] = {}
    return reports

def get_artifact_version() -> tuple[float, ...]:
    stamps: list[float] = []
    for model_name in MODEL_ORDER:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        report_path = REPORTS_DIR / f"{model_name}_metrics.json"
        stamps.append(model_path.stat().st_mtime if model_path.exists() else 0.0)
        stamps.append(report_path.stat().st_mtime if report_path.exists() else 0.0)
    return tuple(stamps)

def predict_all_models(
    patient_df: pd.DataFrame,
    patient_input: dict[str, float | int],
    actual_target: int | None,
) -> pd.DataFrame:
    artifact_version = get_artifact_version()
    models = load_models(artifact_version)
    reports = load_reports(artifact_version)
    rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDER:
        model = models[model_name]
        probability = float(model.predict_proba(patient_df)[0][1])
        predicted_target = int(probability >= 0.5)
        confidence = probability if predicted_target == 1 else 1 - probability

        predicted_label = "Heart Disease" if predicted_target == 1 else "No Heart Disease"
        correctness = None if actual_target is None else predicted_target == actual_target

        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "prediction": predicted_label,
                "prediction_target": predicted_target,
                "confidence": confidence,
                "heart_disease_probability": probability,
                "test_accuracy": reports[model_name].get("test_metrics", {}).get("accuracy"),
                "test_f1": reports[model_name].get("test_metrics", {}).get("f1"),
                "is_correct": correctness,
                "correctness_label": "Correct" if correctness is True else "Wrong",
            }
        )
    return pd.DataFrame(rows)

def build_prediction_signature(
    artifact_version: tuple[float, ...],
    patient_input: dict[str, float | int],
    actual_target: int | None,
) -> tuple:
    ordered_values = tuple(patient_input[column] for column in FEATURE_COLUMNS)
    return artifact_version + ordered_values + (actual_target,)

# ==============================================================================
# GIAO DIỆN & ĐỒ THỊ
# ==============================================================================
def render_chart(results_df: pd.DataFrame) -> None:
    chart_df = results_df.copy()
    chart_df["confidence_percent"] = (chart_df["confidence"] * 100).round(1)
    chart_df["bar_text"] = chart_df["confidence_percent"].astype(str) + "%"
    chart_df["inside_text"] = chart_df["prediction"].map(
        {
            "No Heart Disease": "✅ No Heart Disease",
            "Heart Disease": "🫀 Heart Disease",
        }
    )

    bar_colors = []
    for row in chart_df.itertuples():
        if row.confidence < 0.70:
            bar_colors.append("#c92f4a")  # Màu đỏ cảnh báo cho cột dưới 70%
        else:
            # Nếu từ 70% trở lên, giữ nguyên màu xanh nếu đoán Đúng, màu đỏ nếu đoán Sai
            bar_colors.append("#2e8540" if row.correctness_label == "Correct" else "#c92f4a")

    fig = px.bar(
        chart_df,
        x="model_label",
        y="confidence",
        text="bar_text",
        hover_data={
            "model": True,
            "model_label": False,
            "prediction": True,
            "correctness_label": True,
            "confidence_percent": True,
            "heart_disease_probability": ":.3f",
            "test_accuracy": ":.3f",
            "test_f1": ":.3f",
            "confidence": False,
            "inside_text": False,
        },
    )

    fig.update_traces(
        marker_color=bar_colors,
        textposition="outside",
        marker_line_color="#111111",
        marker_line_width=1.5,
        cliponaxis=False,
    )
    for row in chart_df.itertuples(index=False):
        fig.add_annotation(
            x=row.model_label,
            y=max(row.confidence * 0.52, 0.12),
            text=row.inside_text,
            showarrow=False,
            textangle=90,
            font=dict(size=11, color="white"),
        )
    fig.update_layout(
        title="Model Predictions Confidence Overview",
        xaxis_title="Model",
        yaxis_title="Prediction Confidence",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        font=dict(size=14, color="#ffffff"),
        margin=dict(l=10, r=10, t=52, b=10),
    )
    fig.update_xaxes(tickangle=-28, gridcolor="#374151")
    fig.update_yaxes(gridcolor="#374151")
    st.plotly_chart(fig, use_container_width=True)

def apply_styles() -> None:
    st.markdown(
        """
        <style>
        /* Ép Streamlit bung lụa 100% bề ngang màn hình máy tính */
        .main .block-container {
            max-width: 100% !important;
            padding-top: 1.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 2rem !important;
        }
        div[class*="stMainBlockContainer"] {
            max-width: 100% !important;
            width: 100% !important;
            padding: 1.5rem 2rem !important;
        }
        .hero-title {
            color: #ff4b4b;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
        }
        .panel-header {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 1rem;
            border-bottom: 2px solid #ff4b4b;
            padding-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# HÀM MAIN ĐIỀU HƯỚNG CHÍNH
# ==============================================================================
def main() -> None:
    apply_styles()
    
    st.markdown('<div class="hero-title">Heart Disease Prediction — Web App Demo</div>', unsafe_allow_html=True)

    example_names = list(EXAMPLE_PATIENTS.keys())
    if "example_patient" not in st.session_state:
        st.session_state["example_patient"] = example_names[0]
        
    st.selectbox("📋 Select Example Patient", example_names, key="example_patient")
    
    selected_example = st.session_state["example_patient"]
    defaults = EXAMPLE_PATIENTS[selected_example]
    artifact_version = get_artifact_version()

    left_col, right_col = st.columns([1.1, 1.0], gap="large")

    with left_col:
        st.markdown('<div class="panel-header">✍️ Enter Patient Features</div>', unsafe_allow_html=True)

        # --- HÀNG PHÂN KHU 1 ---
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("age (years)", min_value=1, max_value=120, value=int(defaults["age"]))
        sex = c2.selectbox(
            "sex (0=female, 1=male)", [0, 1], index=int(defaults["sex"]),
            format_func=lambda x: "1 (male)" if x==1 else "0 (female)"
        )
        cp = c3.selectbox(
            "cp (chest pain type)", [1, 2, 3, 4], index=int(defaults["cp"]) - 1,
            help="Loại đau ngực lâm sàng:\n1: Điển hình (Typical Angina)\n2: Không điển hình (Atypical Angina)\n3: Không do tim (Non-anginal Pain)\n4: Không triệu chứng (Asymptomatic)"
        )
        trestbps = c4.number_input(
            "trestbps (resting BP)", min_value=60, max_value=250, value=int(defaults["trestbps"]),
            help="Huyết áp tâm thu đo lúc nghỉ ngơi (mmHg). Chỉ số lý tưởng ở người bình thường là quanh mức 120."
        )

        # --- HÀNG PHÂN KHU 2 ---
        c1, c2, c3, c4 = st.columns(4)
        chol = c1.number_input(
            "chol (serum cholesterol)", min_value=100, max_value=700, value=int(defaults["chol"]),
            help="Chỉ số mỡ máu lượng serum cholesterol (mg/dl). Vượt ngưỡng 200 mg/dl biểu thị nguy cơ cao."
        )
        fbs = c2.selectbox(
            "fbs (>120 mg/dl? 1/0)", [0, 1], index=int(defaults["fbs"]),
            format_func=lambda x: "1 (True)" if x==1 else "0 (False)",
            help="Đường huyết đói (Fasting Blood Sugar). Nếu lớn hơn 120 mg/dl (Giá trị 1) cảnh báo nguy cơ tiểu đường."
        )
        restecg = c3.selectbox(
            "restecg (0..2)", [0, 1, 2], index=int(defaults["restecg"]),
            help="Kết quả điện tâm đồ khi nghỉ:\n0: Bình thường\n1: Sóng ST-T bất thường\n2: Phì đại thất trái (Left ventricular hypertrophy)"
        )
        thalach = c4.number_input(
            "thalach (max heart rate)", min_value=50, max_value=250, value=int(defaults["thalach"]),
            help="Nhịp tim cao nhất ghi nhận được trong bài kiểm tra thể lực gắng sức tối đa."
        )

        # --- HÀNG PHÂN KHU 3 ---
        c1, c2, c3, c4 = st.columns(4)
        exang = c1.selectbox(
            "exang (exercise angina)", [0, 1], index=int(defaults["exang"]),
            format_func=lambda x: "1 (Yes)" if x==1 else "0 (No)",
            help="Xảy ra cơn đau thắt ngực khi đang vận động/tập thể dục hay không."
        )
        oldpeak = c2.number_input(
            "oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=float(defaults["oldpeak"]), step=0.1,
            help="Độ võng/hạ xuống của đoạn ST trên điện tâm đồ khi vận động so với trạng thái nghỉ ngơi."
        )
        slope = c3.selectbox(
            "slope (1..3)", [1, 2, 3], index=int(defaults["slope"]) - 1,
            help="Hướng biến đổi/độ dốc của đỉnh đoạn ST khi tập gắng sức:\n1: Dốc lên (Upsloping)\n2: Đi ngang (Flat)\n3: Dốc xuống (Downsloping)"
        )
        ca = c4.selectbox(
            "ca (major vessels)", [0, 1, 2, 3], index=int(defaults["ca"]),
            help="Số lượng mạch máu chính (0-3 mạch vành) quan sát thấy qua kỹ thuật nhuộm màu nội soi huỳnh quang."
        )

        # --- HÀNG PHÂN KHU 4 ---
        thal = st.selectbox(
            "thal (3=normal, 6=fixed, 7=reversible)", [3, 6, 7], index=[3, 6, 7].index(int(defaults["thal"])),
            format_func=lambda x: f"{x} (Normal)" if x==3 else (f"{x} (Fixed Defect)" if x==6 else f"{x} (Reversible Defect)"),
            help="Kết quả chụp xạ hình cơ tim Thalassemia:\n3: Bình thường\n6: Tổn thương cố định (vùng cơ tim chết)\n7: Tổn thương có thể phục hồi (vùng thiếu máu cục bộ)"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

    # BACKEND EXECUTION (KẾT NỐI VÀ GỌI CHẠY MODEL THẬT)
    actual_target = int(EXAMPLE_PATIENTS[st.session_state["example_patient"]]["actual_target"])
    patient_input = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    current_signature = build_prediction_signature(artifact_version, patient_input, actual_target)
    saved_signature = st.session_state.get("prediction_signature")

    if predict_clicked or saved_signature != current_signature or "prediction_results" not in st.session_state:
        patient_df = encode_patient_input(patient_input)
        # Gọi hàm chính thức để nạp dữ liệu vào mô hình
        st.session_state["prediction_results"] = predict_all_models(patient_df, patient_input, actual_target)
        st.session_state["actual_target"] = actual_target
        st.session_state["prediction_signature"] = current_signature

    with right_col:
        results_df = st.session_state.get("prediction_results")
        st.markdown('<div class="panel-header">📊 Model Predictions Overview</div>', unsafe_allow_html=True)
        
        # Biểu đồ Plotly động nhận dữ liệu thật từ mô hình
        render_chart(results_df)
        
        tinh_trang_thuc_te = "❤️ Heart Disease (Có bệnh)" if actual_target == 1 else "✅ Healthy "
        st.write("")
        st.metric(label="Actual Patient Condition:", value=tinh_trang_thuc_te)


        # ----------------------------------------------------------------------
        # DỮ LIỆU ĐỘNG: ĐỌC ACCURACY TỪ FILE METRICS THẬT CỦA MODEL
        # ----------------------------------------------------------------------
        st.markdown("<br><hr style='border-color: #374151;'>", unsafe_allow_html=True)
        
        # Gọi hàm load_reports để lấy thông tin (sử dụng artifact_version đã khai báo ở trên)
        all_reports = load_reports(artifact_version)

        # --- HÀNG 1: TRAINING ACCURACY (VALIDATION SET) ---
        st.markdown("<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 0.8rem; font-weight: 600;'>Model training accuracy (validation set):</p>", unsafe_allow_html=True)
        train_cols = st.columns(8)
        
        for i, model_name in enumerate(MODEL_ORDER):
            # Lấy train accuracy từ file json, nếu không có thì để mặc định "N/A"
            # Lưu ý: Bạn cần kiểm tra xem trong file json của bạn key lưu train accuracy tên là gì (VD: "train_accuracy" hoặc giống cấu trúc test bên dưới)
            model_metrics = all_reports.get(model_name, {})
            train_acc_raw = model_metrics.get("train_metrics", {}).get("accuracy") or model_metrics.get("validation_metrics", {}).get("accuracy")
            
            if train_acc_raw is not None:
                train_acc_str = f"{float(train_acc_raw) * 100:.1f}%" if float(train_acc_raw) <= 1.0 else f"{float(train_acc_raw):.1f}%"
            else:
                train_acc_str = "N/A"

            with train_cols[i]:
                st.markdown(
                    f"""
                    <div style='text-align: left; line-height: 1.2; margin-bottom: 1rem;'>
                        <div style='font-size: 1.4rem; font-weight: 700; color: #38bdf8;'>{train_acc_str}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        # Khoảng cách nhỏ giữa 2 tầng chỉ số
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

        # --- HÀNG 2: TEST-SET ACCURACY ---
        st.markdown("<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 0.8rem; font-weight: 600;'>Model test-set accuracy (trained on scaled Data_raw):</p>", unsafe_allow_html=True)
        test_cols = st.columns(8)
        
        for i, model_name in enumerate(MODEL_ORDER):
            # Lấy đúng cấu trúc giống hàm predict_all_models: reports[model_name].get("test_metrics", {}).get("accuracy")
            model_metrics = all_reports.get(model_name, {})
            test_acc_raw = model_metrics.get("test_metrics", {}).get("accuracy")
            
            if test_acc_raw is not None:
                # Tự động nhân 100 nếu giá trị dạng thập phân (0.839 -> 83.9%)
                test_acc_str = f"{float(test_acc_raw) * 100:.1f}%" if float(test_acc_raw) <= 1.0 else f"{float(test_acc_raw):.1f}%"
            else:
                test_acc_str = "N/A"
                
            model_label = MODEL_LABELS.get(model_name, model_name)

            with test_cols[i]:
                st.markdown(
                    f"""
                    <div style='text-align: left; line-height: 1.2;'>
                        <div style='font-size: 1.4rem; font-weight: 700; color: #ffffff;'>{test_acc_str}</div>
                        <div style='font-size: 0.8rem; color: #9ca3af; min-height: 2.2rem; margin-top: 0.3rem;'>{model_label}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()