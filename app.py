import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

st.subheader(':blue[⚙️ Prediction of Cavitation in Centrifugal Pump ⚙️]')
#Data Set Input to train and to test 

st.warning("ตัวอย่าง ค่าการสั่นสะเทือนตามแนวแกน x, y, z และค่าเป้าหมาย (Target) เพื่อใชัในการเรียนรู้และทดสอบของเครื่อง (machine learning) ")
vibra_input = pd.read_csv('Cavi.csv')
button1 = st.button("click เพื่อดูข้อมูล")
if button1:
    st.write(vibra_input.sample(3))

X = vibra_input[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)','Acceleration z (m/s^2)']]
y = vibra_input['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#To Scale Data Set
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

Cavi_rf = RandomForestClassifier(n_estimators=100, random_state=42)
Cavi_rf.fit(X_train_sc, y_train)
y_pred_rf = Cavi_rf.predict(X_test_sc)

Cavi_svm = SVC()
Cavi_svm.fit(X_train_sc, y_train)
y_pred_svm = Cavi_svm.predict(X_test_sc)

st.info(':blue[กรุณาป้อนค่าการสั่นสะเทือนตามแนวแกน x, y, z เพื่อทำนายการเกิด คาวิเตชันในปั๊มหอยโข่ง ชุดทดสอบ multi pump test ที่ วศ.เครื่องกล ม.บูรพา ชลบุรี 🏖 🧜‍♀️ 🏝 🇹🇭]')
col1, col2, col3 = st.columns(3)
# Create New Data Input for Prediction
with col1:
    vx = st.number_input('📳 ค่าการสั่นสะเทือนตามแนวแกน X $(m/s^2)$', value=0.0)
with col2:
    vy = st.number_input('📳 ค่าการสั่นสะเทือนตามแนวแกน Y $(m/s^2)$', value=0.0)
with col3:
    vz = st.number_input('📳 ค่าการสั่นสะเทือนตามแนวแกน Z $(m/s^2)$', value=0.0)

# Create a DataFrame for the new data point
new_data_point = pd.DataFrame([{
    'Acceleration x (m/s^2)': vx,
    'Acceleration y (m/s^2)': vy,
    'Acceleration z (m/s^2)': vz
}])

# To Scale New Data Input
new_data_point_sc = scaler.transform(new_data_point)

# Define icons based on prediction results
status_icons = {
    "NoCavi": "✅",
    "50Cavi": "⚠️",
    "Cavi": "🚨"
}
col4, col5 = st.columns(2)
with col4:
    # Make a prediction using the Random Forest model
    predictionRF = Cavi_rf.predict(new_data_point_sc)

    current_iconRF = status_icons.get(predictionRF[0], "⚙️")

    st.subheader(f"{current_iconRF}:green[ ปั๊มทำงานที่สะภาวะ:] {predictionRF[0]}")

    status_accuracyRF = {
        "NoCavi": "0.79",
        "50Cavi": "0.79",
        "Cavi": "0.76"
        }
    current_accuracyRF = status_accuracyRF.get(predictionRF[0],'0.00')
    #scoreRF = accuracy_score(y_test, y_pred_rf)

    st.success(f"ด้วยวิธี Random Forest ให้ความแม่นยำเท่ากับ: {current_accuracyRF}")
with col5:
# Make a prediction using the SVM model

    predictionSVM = Cavi_svm.predict(new_data_point_sc)
    current_iconSVM = status_icons.get(predictionSVM[0], "⚙️")
    st.subheader(f"{current_iconSVM} :green[ปั๊มทำงานที่สะภาวะ:] {predictionSVM[0]}")


    status_accuracySVM = {
        "NoCavi": "0.92",
        "50Cavi": "0.44",
        "Cavi": "0.86"
        }
    current_accuracySVM = status_accuracySVM.get(predictionSVM[0],'0.00')
    #scoreSVM = accuracy_score(y_test, y_pred_svm)
    st.success(f"ด้วยวิธี Support Vector Machine ให้ความแม่นยำเท่ากับ:{current_accuracySVM}")
