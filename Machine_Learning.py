import streamlit as st
import pandas as pd

st.set_page_config(page_title="Intelligent-System-project", layout="wide")

st.sidebar.header("Navigation")
st.sidebar.page_link("Machine_Learning.py", icon="🤖", disabled=True)
st.sidebar.page_link("pages/Neural_Network.py", icon="🧠")
st.sidebar.page_link("pages/Demo_Machine_Learning.py", icon="📊")
st.sidebar.page_link("pages/Demo_Neural_Network.py", icon="📈")

st.markdown('<h1 style="font-size: 40px;">🤖 Machine Learning Deployment</h1>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

with st.expander("📌 **Machine Learning คืออะไร!**"):
    st.info("""
    **Machine Learning คือ** กระบวนการที่ทำให้คอมพิวเตอร์สามารถเรียนรู้และพัฒนาการทำงานให้ดีขึ้นเอง จากข้อมูลและสภาพแวดล้อมที่ได้รับ  
    - ไม่ต้องมีมนุษย์คอยกำกับหรือเขียนโปรแกรมใหม่เมื่อมีข้อมูลรูปแบบใหม่ ๆ  
    - คอมพิวเตอร์สามารถ **ตีความและตอบสนอง** ต่อข้อมูลได้เอง  
    - **ช่วยธุรกิจและอุตสาหกรรม** ในการวิเคราะห์ข้อมูล ลดต้นทุน และเพิ่มประสิทธิภาพในการแข่งขัน  
    """)

with st.expander("📌 **ประโยชน์ของ Machine Learning!**"):
    st.info("""
    **Machine Learning** สามารถนำมาใช้ทำประโยชน์ได้มากมาย ขึ้นอยู่กับจินตนาการของผู้พัฒนา  
    - **Google Maps**: ช่วยค้นหาเส้นทางที่ประหยัดเวลามากที่สุด  
    - **Google Translate**: นำ Automation มาทำงานร่วมกับ Machine Learning เพื่อช่วยแปลภาษาได้แม่นยำขึ้น  
    - **Speech-to-Text** (เช่น LINE Chat): ช่วยแปลงเสียงพูดเป็นข้อความ เพื่อลดเวลาการพิมพ์  
    """)

st.markdown('''
    <p style="font-size: 20px;">
            From Chat Gpt Create the Dataset
        <a href="https://drive.google.com/file/d/1c1AkKC3XJyQbBHK1xOpBkQByOFnRYvPq/view" 
           target="_blank" style="font-size: 25px; color: blue;">
           health-nutrition-survey.csv
        </a>.
        <br>
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📚 เนื้อหาเกี่ยวกับ</h1>', unsafe_allow_html=True)
st.markdown('''
    <p style="font-size: 20px;">
        ข้อมูลนี้เกี่ยวกับ สุขภาพและโภชนาการของเด็กอายุ 8-12 ปี ประกอบด้วยข้อมูล อายุ, เพศ, น้ำหนัก, ส่วนสูง, BMI, ปริมาณอาหารที่บริโภค, ระดับกิจกรรมทางกาย, และสภาวะสุขภาพ
            
        ข้อสังเกตที่น่าสนใจบางประการจากข้อมูล:
            - มีค่าข้อมูลที่หายไปในบางคอลัมน์
            - ช่วงอายุของเด็กอยู่ระหว่าง 8-12 ปี
            - ค่าดัชนีมวลกาย (BMI) มีความสัมพันธ์กับภาวะสุขภาพ
            - การบริโภคอาหาร (Nutritional Intake) มีแนวโน้มเชื่อมโยงกับภาวะสุขภาพ
            - ระดับกิจกรรมทางกาย (Physical Activity) อาจมีอิทธิพลต่อภาวะสุขภาพ
            - สัดส่วนของกลุ่มสุขภาพ (Health Condition) อาจไม่สมดุลกัน
            - ความสัมพันธ์ระหว่างเพศกับภาวะสุขภาพ
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📊 Features หลักๆ ที่มีอยู่ใน Dataset นี้</h1>', unsafe_allow_html=True)
with st.expander("📌 **Click Here to Learn More!**"):
    st.markdown("""
    - **Age (อายุ)**: อาจมีผลต่อ BMI และโภชนาการของเด็ก;**ตัวเลข (Integer)**
    - **Gender (เพศ)**: อาจมีแนวโน้มที่เด็กชายและเด็กหญิงจะมีค่าทางโภชนาการแตกต่างกัน;**หมวดหมู่ (M/F)**
    - **Height (ส่วนสูง)**: ใช้ร่วมกับน้ำหนักเพื่อคำนวณ BMI;**ตัวเลขทศนิยม (Float)**
    - **Weight (น้ำหนัก)**: ใช้คำนวณ BMI ซึ่งเป็นตัวแปรสำคัญในการพยากรณ์ภาวะสุขภาพ;**ตัวเลขทศนิยม (Float)**
    - **BMI (ดัชนีมวลกาย)**: Feature สำคัญที่สุด ใช้บ่งบอกว่าน้ำหนักของเด็กอยู่ในเกณฑ์ปกติหรือไม่;**ตัวเลขทศนิยม (Float)**
    - **Nutritional Intake (ปริมาณอาหารที่บริโภค)**: มีผลต่อภาวะโภชนาการของเด็ก;**ตัวเลขทศนิยม (Float)**
    - **Physical Activity (กิจกรรมทางกาย)**: อาจมีผลต่อ BMI และสุขภาพของเด็ก;**หมวดหมู่ (High, Medium, Low)**
    """, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">Show DataFrame As Dataset</h1>', unsafe_allow_html=True)
df = pd.read_csv("Dataset/health_nutrition_survey.csv")  
st.dataframe(df)  

st.write("<br><br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🛠️ การเตรียมข้อมูล | โมเดล | อัลกอริทึมที่ใช้พัฒนา
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('## health_nutrition_survey.csv')

code = '''
    # โหลดข้อมูล
    from google.colab import files
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">อัพโหลดไฟล์เข้ามาในงาน</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # ตรวจสอบข้อมูล
    print(df.info())
    print(df.head())
'''
st.code(code, language="python")

st.markdown('<h5 style="font-size: 20px;">ดูข้อมูล 5 แถวแรกและดูข้อมูล</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # พล็อต Boxplot เพื่อตรวจสอบ Outliers
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
    plt.title('Boxplot of Numeric Features')
    plt.show()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ใช้เพื่อดูมีข้อมูลที่เกินออกมานอกขอบเขตหรือไม่หรือ Outlier</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # พล็อต Histogram ของตัวแปรตัวเลข
    df.select_dtypes(include=['float64', 'int64']).hist(figsize=(12, 8), bins=20)
    plt.suptitle('Histogram of Numeric Features')
    plt.show()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ใช้สำหรับดูข้อมูลในรายละเอียดของตัวเลขใน Feature โดย plot Histogram</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # พล็อต Heatmap ของความสัมพันธ์ระหว่างตัวแปรตัวเลข
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">หาความสัมพันะ์ระหว่างตัวแปรโดยใช้ Heatmap plot corelation</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # Countplot สำหรับตัวแปร categorical
    categorical_cols = ['Gender', 'Physical Activity', 'Health Condition']
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'Countplot of {col}')
        plt.show()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">Countplot สำหรับตัวแปร categorical เพื่อดูข้อมูลแต่ละ Column</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # จัดการค่าขาดหาย
    for col in ['Weight', 'BMI', 'Nutritional Intake']:
        df.loc[:, col] = df[col].fillna(df[col].median())
    df.loc[:, 'Physical Activity'] = df['Physical Activity'].fillna(df['Physical Activity'].mode()[0]).infer_objects(copy=False)

    # Label Encoding
    label_encoders = {}
    categorical_cols = ['Gender', 'Physical Activity', 'Health Condition']
    for col in categorical_cols:
        le = LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col])
        label_encoders[col] = le
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">จัดการค่าที่หายไปและแปลงข้อมูลประเภท Categorical ให้เป็นตัวเลข</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # แยก Features และ Target
    X = df.drop(columns=['Health Condition'])
    y = df['Health Condition'].astype(int)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">แยก Features และ Target เพื่อเตรียมนำไปแบ่งข้อมูล</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # แก้ปัญหา Class Imbalance ด้วย SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">แก้ปัญหา Class Imbalance ด้วย SMOTE</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ส่วนสำหรับฝึกสอน (Training set) และ ส่วนสำหรับทดสอบ (Testing set) เพื่อใช้ในการสร้างและประเมินโมเดล Machine Learning</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # สร้างและ Train โมเดล Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้างโมเดล Random Forest และฝึกสอนโมเดลด้วยข้อมูล Training set เพื่อให้โมเดลสามารถเรียนรู้รูปแบบและความสัมพันธ์ในข้อมูล และนำไปใช้ในการทำนายผลในภายหลัง</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # ทำนายผลและประเมินโมเดล
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:\n', report)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ใช้โมเดลที่ฝึกสอนแล้วเพื่อทำนายผลบนข้อมูล Testing set และประเมินประสิทธิภาพของโมเดลโดยใช้ metrics ต่างๆ เช่น Accuracy และ Classification report ซึ่งช่วยให้เราทราบว่าโมเดลทำงานได้ดีแค่ไหน</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        📚 อัลกอริทึมสำหรับการพัฒนา
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <div style="
        background-color: #1E1E1E; 
        padding: 25px; 
        border-radius: 12px;
        box-shadow: 3px 3px 12px rgba(255,255,255,0.2);
        margin: 20px 0px;
    ">
    <div 
        <h3 style="color: #FF5733;">1.  Random Forest</h3>
        <p style="color: #F8F8FF;">เป็นโมเดลที่ใช้ หลาย Decision Tree มารวมกันเพื่อพยากรณ์ผลลัพธ์แต่ละต้นไม้จะใช้ ส่วนหนึ่งของข้อมูลและ Features ในการฝึก
            ใช้วิธี Voting เพื่อตัดสินว่ากลุ่มใดเป็นคำตอบที่ดีที่สุด</p>
    </div>
    <div 
        <h4 style="color: #FFA07A;"> 🔹 กระบวนการทำงานของ Random Forest:</h4>
        <ul style="color: #F8F8FF;">
            <li><b>สร้างชุดข้อมูลสุ่ม (Bootstrap Sampling):</b> – เลือกข้อมูลมาแบบสุ่มหลายชุด</li>
            <li><b>สร้าง Decision Trees:</b> – แต่ละต้นไม้ใช้เกณฑ์ Gini Impurity หรือ Entropy</li>
            <li><b>รวมผลลัพธ์ของทุกต้นไม้:</b> – ใช้วิธี Voting (สำหรับ Classification)</li>
            <li><b>วัดผลโมเดล:</b> – ตรวจสอบ Accuracy, Precision, Recall</li>
        </ul>
        <p>📌 ข้อดี: ความแม่นยำสูง, ทนต่อข้อมูลที่ซับซ้อน
            📌 ข้อเสีย: ใช้ทรัพยากรมากกว่าปกติ</p>
    </div>
    <div 
        <h3 style="color: #FF5733;">2. Logistic Regression</h3>
        <p style="color: #F8F8FF;">เป็น โมเดลทางสถิติ ที่ใช้พยากรณ์ Health Condition ว่าเด็กจะอยู่ในกลุ่ม Underweight, Healthy หรือ Overweight/Obese ใช้ฟังก์ชัน Sigmoid แปลงค่าผลลัพธ์เป็น ค่าความน่าจะเป็น ระหว่าง 0-1 ใช้ Threshold (0.5) เพื่อจัดกลุ่ม</p>
    </div>
    <div
        <h4 style="color: #FFA07A;"> 🔹 กระบวนการทำงานของ Logistic Regression:</h4>
        <ul style="color: #F8F8FF;">
            <li><b>เตรียมข้อมูล:</b> – เลือก Features สำคัญ เช่น Age, Gender, BMI, Nutritional Intake, Physical Activity"</li>
            <li><b>สร้างสมการพยากรณ์:</b> – ใช้ฟังก์ชัน Logistic เพื่อคำนวณโอกาสของแต่ละกลุ่ม</li>
            <li><b>พยากรณ์ผลลัพธ์:</b> – ถ้า P(Y) > 0.5 → Overweight/Obese, ถ้า P(Y) < 0.5 → Healthy หรือ Underweight</li>
            <li><b>วัดผลโมเดล:</b> – ใช้ค่าความแม่นยำ (Accuracy), Precision และ Recall</li>
        </ul>
        <p>📌 ข้อดี: เข้าใจง่าย, ใช้ทรัพยากรต่ำ
            📌 ข้อเสีย: อาจไม่แม่นยำหากข้อมูลมีความซับซ้อน</p>
    </div>
        <div 
            <h3 style="color: #FF5733;">🔹 เปรียบเทียบระหว่าง Random Forest และ Logistic Regression</h3>
            <table style="color: #F8F8FF; width: 100%; border-collapse: collapse; border: 1px solid white; text-align: center;">
                <tr style="background-color: #333;">
                    <th style="padding: 12px; border: 1px solid white;">อัลกอริทึม</th>
                    <th style="padding: 12px; border: 1px solid white;">ความแม่นยำ</th>
                    <th style="padding: 12px; border: 1px solid white;">ความซับซ้อน</th>
                    <th style="padding: 12px; border: 1px solid white;">ความสามารถอธิบายผลลัพธ์</th>
                    <th style="padding: 12px; border: 1px solid white;">การจัดการข้อมูลที่ซับซ้อน</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid white;">Logistic Regression</td>
                    <td style="padding: 10px; border: 1px solid white;">ปานกลาง</td>
                    <td style="padding: 10px; border: 1px solid white;">ต่ำ</td>
                    <td style="padding: 10px; border: 1px solid white;">สูง</td>
                    <td style="padding: 10px; border: 1px solid white;">จำกัด</td>
                </tr>
                <tr style="background-color: #222;">
                    <td style="padding: 10px; border: 1px solid white;">Random Forest</td>
                    <td style="padding: 10px; border: 1px solid white;">สูง</td>
                    <td style="padding: 10px; border: 1px solid white;">สูง</td>
                    <td style="padding: 10px; border: 1px solid white;">ปานกลาง</td>
                    <td style="padding: 10px; border: 1px solid white;">ดี</td>
                </tr>
            </table>
        </div>
    </div>
""", unsafe_allow_html=True)

st.write("<br><br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🚀 Machine Learning (health_nutrition_survey.csv)
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

with st.expander("📌 **ขั้นตอนการพัฒนาโมเดล!**"):
    st.markdown("""
        <div style="
            background-color: #1E1E1E; 
            padding: 25px; 
            border-radius: 12px;
            box-shadow: 3px 3px 12px rgba(255,255,255,0.2);
            margin: 20px 0px;
        ">
            <ul style="color: #F8F8FF; font-size: 18px; line-height: 1.6;">
                <li>1️⃣ เตรียมข้อมูล (Data Preprocessing)</li>
                <li>2️⃣ พัฒนาโมเดล Logistic Regression</li>
                <li>3️⃣ พัฒนาโมเดล Random Forest</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('## โค้ดตัวอย่าง Random Forest และ Logistic Regression')
code = '''
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def preprocess_data(df):
    # Drop irrelevant columns if exists
    if 'Student ID' in df.columns:
        df.drop(columns=['Student ID'], inplace=True)
    
    # Fill missing values for numerical columns with mean
    num_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    return df

def train_model(df):
    X = df.drop(columns=['Health Condition'])
    y = df['Health Condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, cm, X.columns

st.title("Health & Nutrition Random Forest")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Raw Data")
    st.write(df.head())
    
    st.write("### Exploratory Data Analysis")
    st.write("#### Missing Values")
    st.write(df.isnull().sum())
    
    st.write("#### Data Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Preprocessing
    df = preprocess_data(df)
    
    # Train model
    model, accuracy, report, cm, feature_names = train_model(df)
    
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("#### Classification Report")
    st.write(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    st.write("#### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(df['Health Condition']), yticklabels=np.unique(df['Health Condition']))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
    
    # Feature Importance
    st.write("#### Feature Importance")
    feature_importance = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, ax=ax)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest")
    st.pyplot(fig)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">โค้ดนี้สร้างแอป Streamlit ที่ช่วยให้ผู้ใช้ อัปโหลดข้อมูลสุขภาพ, วิเคราะห์ข้อมูล, ฝึกโมเดล Random Forest และแสดงผลลัพธ์การพยากรณ์ภาวะสุขภาพของเด็ก </h5>', unsafe_allow_html=True)

code = '''
    # Step 1: Load Data
st.title("Predict BMI With Column Weight and Height")
st.title("Logistic Regression Model")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.write("### BMI Classification Table")
st.write("""
| BMI Range        | Classification       |
|-----------------|---------------------|
| BMI < 18.5      | Underweight         |
| 18.5 ≤ BMI < 24.9 | Normal weight      |
| BMI ≥ 25        | Overweight/Obese    |
""")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Select Features
    feature_cols = st.multiselect("Select Feature Columns", df.columns, default=["Weight", "Height"])

    if "Weight" in feature_cols and "Height" in feature_cols:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

        # Classify BMI
        df["BMI_Class"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 24.9, np.inf],
            labels=["Underweight", "Normal weight", "Overweight/Obese"]
        )

        # Drop missing values before training
        df_cleaned = df.dropna(subset=["Weight", "Height", "BMI_Class"])

        X = df_cleaned[feature_cols].copy()
        y = df_cleaned["BMI_Class"]

        # Step 2: Preprocessing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fill missing values in X_train and X_test
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Step 3: Train Model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Step 4: Evaluate Model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("#### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Step 5: Hyperparameter Tuning
        st.write("### Hyperparameter Tuning")
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        st.write(f"Best Parameters: {grid_search.best_params_}")

        # Step 6: Deployment (Simple Prediction)
        st.write("### Make a Prediction")
        weight = st.number_input("Enter Weight (kg)", value=55.0)
        height = st.number_input("Enter Height (cm)", value=170.0)

        if st.button("Predict"):
            user_bmi = weight / ((height / 100) ** 2)
            scaled_input = scaler.transform([[weight, height]])
            prediction = model.predict(scaled_input)

            st.write(f"Predicted BMI: {user_bmi:.2f}")
            st.write(f"Predicted Class: {prediction[0]}")
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">โค้ดนี้สร้างแอป Streamlit ที่ช่วยให้ผู้ใช้ อัปโหลดข้อมูลสุขภาพ, วิเคราะห์ข้อมูล, ฝึกโมเดล Logistic Regression และดูผลการพยากรณ์ภาวะสุขภาพของเด็ก</h5>', unsafe_allow_html=True)
