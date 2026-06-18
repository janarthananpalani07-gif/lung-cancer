import streamlit as st
from pipeline import run_pipeline
import numpy as np
import cv2
import pandas as pd

# --------------------------------------------------
# 🧠 PRETRAINED DL TUMOR SEGMENTATION MODEL
# --------------------------------------------------
st.set_page_config(layout="wide")

st.title("🫁 Lung Cancer Detection & Patient Risk Analysis System")

st.markdown("Upload MULTIPLE CT slices of one patient")


def get_patient_risk_label(patient_cancer_score, slices_with_nodule):
    if slices_with_nodule == 0:
        return "LOW RISK PATIENT"
    if patient_cancer_score >= 0.60:
        return "HIGH RISK PATIENT"
    if patient_cancer_score >= 0.30:
        return "MEDIUM RISK PATIENT"
    return "LOW RISK PATIENT"


def get_clinical_message(patient_risk, slices_with_nodule, cancer_positive_slices):
    if slices_with_nodule == 0:
        return "No clear lung nodules were detected across the analysed slices. Recommend routine clinical review."
    if patient_risk == "HIGH RISK PATIENT":
        return "Patient shows high likelihood of malignant lung nodules. Recommend radiologist review."
    if patient_risk == "MEDIUM RISK PATIENT":
        return "Patient shows mixed suspicious findings across detected nodules. Recommend radiologist review and follow-up."
    if cancer_positive_slices > 0:
        return "Only a small portion of detected nodules appear suspicious. Recommend clinical correlation."
    return "Detected nodules appear more likely non-cancerous in this scan set. Recommend radiologist review."

# --------------------------------------------------
# 📂 MULTI IMAGE UPLOAD (PATIENT LEVEL)
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Patient CT Slices",
    type=["jpg","png"],
    accept_multiple_files=True
)

# --------------------------------------------------
# 🚀 RUN ANALYSIS
# --------------------------------------------------
if uploaded_files:

    patient_results = []
    cancer_probs = []
    total_nodules = 0

    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):

        # save temp slice
        temp_path = f"slice_{i}.jpg"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        result = run_pipeline(temp_path)
        patient_results.append(result)

        total_nodules += result["nodules"]
        cancer_probs.append(result["confidence"])

        progress.progress((i+1)/len(uploaded_files))

    st.success("Patient scan analysis completed")

    total_slices = len(patient_results)
    slices_with_nodule = sum(1 for r in patient_results if r["tumor_detected"])
    cancer_positive_slices = sum(
        1 for r in patient_results if r["tumor_detected"] and r["prediction"] == "Cancer"
    )
    non_cancer_slices = max(0, slices_with_nodule - cancer_positive_slices)
    no_nodule_slices = max(0, total_slices - slices_with_nodule)
    patient_cancer_score = (
        cancer_positive_slices / slices_with_nodule if slices_with_nodule > 0 else 0.0
    )
    patient_risk = get_patient_risk_label(patient_cancer_score, slices_with_nodule)
    clinical_message = get_clinical_message(
        patient_risk,
        slices_with_nodule,
        cancer_positive_slices,
    )

    # --------------------------------------------------
    # 🧾 PATIENT SUMMARY
    # --------------------------------------------------
    st.header("🧾 Patient Final Report")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total CT Slices Analysed", total_slices)
    col2.metric("Slices with Nodules", slices_with_nodule)
    col3.metric("Cancer Positive Slices", cancer_positive_slices)
    col4.metric("Cancer Probability Score", f"{patient_cancer_score * 100:.1f}%")

    if slices_with_nodule > 0:
        st.markdown(
            f"**Cancer Probability Score:** `{cancer_positive_slices} / {slices_with_nodule} = {patient_cancer_score * 100:.1f}%`"
        )
    else:
        st.markdown("**Cancer Probability Score:** No nodule-detected slices available for scoring.")

    if patient_risk == "HIGH RISK PATIENT":
        st.error(f"Risk Stratification: {patient_risk}")
    elif patient_risk == "MEDIUM RISK PATIENT":
        st.warning(f"Risk Stratification: {patient_risk}")
    else:
        st.success(f"Risk Stratification: {patient_risk}")

    chart_data = pd.DataFrame(
        {
            "Category": [
                "Cancer slices",
                "Non-cancer slices",
                "No nodule slices",
            ],
            "Slices": [
                cancer_positive_slices,
                non_cancer_slices,
                no_nodule_slices,
            ],
        }
    ).set_index("Category")

    st.markdown("### 📈 Statistics Graph")
    st.bar_chart(chart_data)

    st.markdown("### 🏥 Final Clinical Message")
    st.info(f"AI Suggestion: {clinical_message}")

    st.markdown("---")

    # --------------------------------------------------
    # 🔬 SLICE BY SLICE VISUALIZATION
    # --------------------------------------------------
    st.header("🔬 Slice-by-Slice Pipeline Output")

    for idx, res in enumerate(patient_results):

        with st.expander(f"CT Slice {idx+1} Analysis"):

            st.subheader("📌 Pipeline Stages")

            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7 = st.columns(3)

            # ===============================
            # ROW 1 — ALWAYS SHOW
            # ===============================
            col1.image(res["original"], channels="BGR", caption="Original")
            col2.image(res["preprocessed"], channels="RGB", caption="Preprocessed")
            col3.image(
                res["lung_segmented"],
                channels="RGB",
                caption="Segmented Lung",
            )
            col4.image(
                res["box_tumor_detected"],
                channels="RGB",
                caption="Box Shape Tumor Detection",
            )

            # ===============================
            # CASE 1 — NO TUMOR DETECTED
            # ===============================
            if res["tumor_detected"]:
                col5.image(
                    res["tumor_segmented"],
                    channels="RGB",
                    caption="Tumor Boundary (Irregular Shape)",
                )
                col6.image(
                    res["tumor_only_segmented"],
                    channels="RGB",
                    caption="Segmented Tumor Only (Zoomed)",
                )

                if res["prediction"] == "Non-Cancer":
                    col7.info("Benign tumor - no cancer heatmap required")
                else:
                    col7.image(
                        res["overlay"],
                        channels="RGB",
                        caption="Heatmap on Tumor Location",
                    )
            elif res["prediction"] == "Cancer":
                st.warning("Cancer pattern detected, but YOLO did not localize a clear nodule in this slice")
            else:
                st.info("No tumor detected in this slice")

            if res["prediction"] == "Cancer" and "heatmap" in res and "overlay" in res:
                st.markdown("### 🔥 Cancer Visualization")
                colA, colB = st.columns(2)
                colA.image(res["heatmap"], channels="RGB", caption="AI Heatmap (Localized)")
                colB.image(
                    res["tumor_color_overlay"],
                    channels="RGB",
                    caption="Single-Color Irregular Tumor Overlay",
                )

        # ===============================
        # METRICS
        # ===============================
        st.markdown("### 📊 Slice Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nodules", res["nodules"])
        m2.metric("Prediction", res["prediction"])
        m3.metric("Confidence", f"{res['confidence']*100:.1f}%")

st.success("Analysis completed for entire patient ✔")
