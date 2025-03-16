import streamlit as st
import pandas as pd

st.set_page_config(page_title="Intelligent-System-project", layout="wide")

st.markdown('<h1 style="font-size: 40px;">🧠 Neural Network Deployment</h1>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

with st.expander("📌 **Neural Network คืออะไร!**"):
    st.info("""
    นิวรัลเน็ตเวิร์ค (Neural Network) เป็นเทคโนโลยีที่ก้าวหน้ามากในวงการปัญญาประดิษฐ์ (AI) ที่ทำให้คอมพิวเตอร์สามารถประมวลผลข้อมูลในลักษณะที่คล้ายกับการทำงานของสมองมนุษย์ โดยการใช้โครงสร้างที่ประกอบด้วยโหนดหรือนิวรอนที่เชื่อมต่อกันในเลเยอร์ต่าง ๆ กระบวนการนี้เรียกว่าดีปเลิร์นนิ่ง (Deep Learning) ซึ่งเป็นสาขาหนึ่งของแมชชีนเลิร์นนิ่ง (Machine Learning) ที่มีความซับซ้อนและมีศักยภาพสูงมาก  
    """)

with st.expander("📌 **ประโยชน์ของ Neural Network!**"):
    st.info("""
    นิวรัลเน็ตเวิร์คมีหลายกรณีการใช้งานในอุตสาหกรรมต่างๆ มากมาย เช่น

    - การวินิจฉัยทางการแพทย์โดยการจำแนกภาพทางการแพทย์
    - การตลาดแบบกำหนดเป้าหมายโดยการกรองเครือข่ายสังคมและการวิเคราะห์ข้อมูลพฤติกรรม
    - การคาดการณ์ทางการเงินโดยการประมวลผลข้อมูลเก่าของเครื่องมือทางการเงิน
    - การพยากรณ์โหลดไฟฟ้าและความต้องการพลังงาน
    - กระบวนการและการควบคุมคุณภาพ
    - การระบุสารประกอบทางเคมี 
    """)

st.write("<br>", unsafe_allow_html=True)

st.markdown('''
    <p style="font-size: 20px;">
        From Chat Gpt Create The Dataset
        <a href="https://drive.google.com/file/d/11wC2ZALEDL3RoaUpw6Wr43rU6s8oLvFX/view" target="_blank" style="font-size: 25px; color: blue;">student-academic-performance.csv</a>.
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📚 เนื้อหาเกี่ยวกับ</h1><br>', unsafe_allow_html=True)
st.markdown('''
    <p style="font-size: 20px;">
        ชุดข้อมูลนี้เป็นข้อมูลเกี่ยวกับนักศึกษาและปัจจัยที่ส่งผลต่อผลการเรียน (Final Grade) โดยมีตัวแปรต่างๆ เช่น อายุ เพศ จำนวนชั่วโมงเรียน อัตราการเข้าเรียน กิจกรรมนอกหลักสูตร และระดับการศึกษาของผู้ปกครอง ซึ่งสามารถนำไปวิเคราะห์เพื่อดูแนวโน้มและปัจจัยที่มีผลต่อเกรดสุดท้ายของนักศึกษาได้
        ข้อสังเกตที่น่าสนใจบางประการจากข้อมูล:
            - จำนวนนักศึกษาที่มีข้อมูลขาดหาย (Missing Data)
            - ความสัมพันธ์ระหว่าง "Study Hours" และ "Final Grade"
            - อัตราการเข้าเรียน (Attendance Rate) มีผลต่อเกรด
            - ผลของกิจกรรม Extracurricular Activities ต่อผลการเรียน
            - ระดับการศึกษาของผู้ปกครองมีผลต่อเกรดของนักศึกษา
            - พบค่าผิดปกติ (Outliers) ใน Study Hours
    </p>
''', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">📊 Features หลักๆ ที่มีอยู่ใน Dataset นี้</h1>', unsafe_allow_html=True)
with st.expander("📌 **Click Here to Learn More!**"):
    st.markdown("""
    - **Student ID**:  รหัสนักศึกษา
    - **Age**: อายุของนักศึกษา
    - **Gender**: เพศ
    - **Study Hours per Week**: จำนวนชั่วโมงที่ใช้เรียนต่อสัปดาห์
    - **Attendance Rate**: อัตราการเข้าเรียน (%)
    - **Extracurricular Activities**: กิจกรรมเสริมหลักสูตรที่นักศึกษาทำ เช่น กีฬา ดนตรี ศิลปะ หรือไม่มี
    - **Parental Education Level**: ระดับการศึกษาของผู้ปกครอง (มัธยม, ปริญญาตรี, ปริญญาโท ฯลฯ)
    - **Final Grade**: เกรดสุดท้ายของนักศึกษา
    """, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">Show DataFrame As Dataset</h1>', unsafe_allow_html=True)
df = pd.read_csv("Dataset/student_academic_performance.csv")  
st.dataframe(df)  

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <h1 style="font-size: 40px; color: #FF5733;">
        🛠️ การเตรียมข้อมูล | โมเดล | อัลกอริทึมที่ใช้พัฒนา
    </h1>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('## student_academic_performance.csv')

code = '''
    # อัปโหลดไฟล์ CSV
    print("Upload your CSV file")
    uploaded = files.upload()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">อัพโหลดไฟล์ CSV</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    # โหลดข้อมูล
    for filename in uploaded.keys():
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    print("\n🔍 Data Preview")
    display(df.head())
    print("\n📊 Summary Statistics")
    display(df.describe())
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">เมื่ออัพโหลดไฟล์เข้ามาเสร็จแล้วก็ทำการเรียกดูข้อมูล 5 แถวแรกและเรียกดูรายละเอียดของข้อมูล</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
        # เลือก Target Column
    target_column = widgets.Dropdown(
        options=df.columns,
        description='🎯 Target:',
        style={'description_width': 'initial'}
    )
    display(target_column)
'''
st.code(code, language="python")

st.write("<br>", unsafe_allow_html=True)

code = '''
        # เลือก Features
    feature_columns = widgets.SelectMultiple(
        options=[col for col in df.columns],
        description='📌 Features:',
        style={'description_width': 'initial'}
    )
    display(feature_columns)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">เลือก Features ของข้อมูลและ target column เพื่อเอาไปสร้าง Model และเอาไป predict ภายหน้า</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    def train_and_evaluate(target_column, feature_columns):
    if target_column and feature_columns:
        X = df[list(feature_columns)]
        y = df[target_column]
        
        # 📌 แบ่งข้อมูล Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 📌 Standardize ข้อมูล
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 📌 สร้าง Neural Network Model (MLPRegressor)
        model = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", solver="adam", max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        
        # 📌 Evaluate Model
        y_pred = model.predict(X_test)
        
        print("\n📈 Model Performance")
        print(f"🔹 R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"🔹 Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"🔹 Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
        
        # 📌 Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.show()
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">สร้างฟังก์ชัน train_and_evalute เพื่อ train model และ เริ่ม predict model</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

code = '''
    train_button = widgets.Button(description="✨ Train & Evaluate Model")
    train_button.on_click(lambda x: train_and_evaluate(target_column.value, feature_columns.value))
    display(train_button)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">ประกาศตัวแปร train_button เพื่อเรียกใช้ widget ปุ่มเมื่อกดจะเป็นการ train model</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)
code = '''
        # พยากรณ์ข้อมูลใหม่
    def predict_new_data():
        new_input = {}
        for feature in feature_columns.value:
            new_input[feature] = float(input(f"📂 Enter {feature}: "))
            
        input_df = pd.DataFrame([new_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        print(f"🔮 Predicted {target_column.value}: {prediction:.4f}")

    predict_button = widgets.Button(description="🔮 Predict New Data")
    predict_button.on_click(lambda x: predict_new_data())
    display(predict_button)
'''
st.code(code, language="python")
st.markdown('<h5 style="font-size: 20px;">การพยาการณ์กรข้อมูลใหม่และสร้างฟังก์ชันเพื่อเริ่มการทำนายข้อมูลใหม่</h5>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown("""
    <div style="
        background-color: #1E1E1E; 
        padding: 25px; 
        border-radius: 12px;
        box-shadow: 3px 3px 12px rgba(255,255,255,0.2);
        margin: 20px 0px;
    ">
        <div>
            <h3 style="color: #FF5733;">1. พื้นฐานของ Regression Model</h3>
            <p style="color: #F8F8FF;">
                Regression Model เป็นเทคนิคที่ใช้คาดการณ์ค่าผลลัพธ์จากตัวแปรอินพุต ซึ่งสามารถแบ่งเป็นประเภทหลัก ๆ ได้แก่
            </p>
            <ul style="color: #F8F8FF;">
                <li><b>Linear Regression</b>: โมเดลที่ใช้สมการเชิงเส้น <br> y = Wx + b</li>
                <li><b>Polynomial Regression</b>: ขยายจาก Linear Regression โดยเพิ่มกำลังของตัวแปร x², x³, ...</li>
                <li><b>Multiple Regression</b>: มีมากกว่าหนึ่งตัวแปรอิสระ <br> y = w₁x₁ + w₂x₂ + ⋯ + b</li>
                <li><b>Neural Network Regression</b>: ใช้โครงข่ายประสาทเทียมที่มีหลายชั้นเพื่อทำการเรียนรู้ฟังก์ชันที่ซับซ้อน</li>
            </ul>
        </div>
    </div>
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
        <div>
            <h3 style="color: #FF5733;">2. Neural Network Regression Model</h3>
            <p style="color: #F8F8FF;">
                Neural Network Regression Model ใช้ Multilayer Perceptron (MLP) ในการทำ Regression โดยโครงสร้างหลักมีดังนี้
            </p>
            <ul style="color: #F8F8FF;">
                <li><b>Input Layer</b>: รับค่า features หรือคุณลักษณะของข้อมูล</li>
                <li><b>Hidden Layers</b>: ทำการแปลงค่าผ่าน weights (W) และ bias (b) ตาม Activation Function</li>
                <li><b>Output Layer</b>: ให้ค่าผลลัพธ์ต่อเนื่องออกมา</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #FFA07A;">🔹 ฟังก์ชันที่ใช้ใน Neural Network Regression</h4>
            <ul style="color: #F8F8FF;">
                <li><b>Activation Function:</b>
                    <ul>
                        <li>ReLU (Rectified Linear Unit): เหมาะสำหรับ hidden layer</li>
                        <li>Linear Activation (Identity Function): ใช้ที่ output layer</li>
                    </ul>
                </li>
                <li><b>Loss Function:</b>
                    <ul>
                        <li>Mean Squared Error (MSE): ใช้วัดค่าความผิดพลาด</li>
                        <li>Mean Absolute Error (MAE): วัดค่าความผิดพลาดโดยใช้ค่าสัมบูรณ์</li>
                    </ul>
                </li>
            </ul>
        </div>
        <div>
            <h4 style="color: #FFA07A;">🔹 ตัวอย่างสมการของ Neural Network Regression</h4>
            <p style="color: #F8F8FF; text-align: center;">
                y = W₃ ⋅ ReLU(W₂ ⋅ ReLU(W₁ X + b₁) + b₂) + b₃
            </p>
            <p style="color: #F8F8FF;">
                โดยที่:
                <ul>
                    <li><b>W</b> และ <b>b</b> คือ weights และ bias ของแต่ละชั้น</li>
                    <li><b>ReLU</b> คือ Activation Function ของ Hidden Layers</li>
                    <li>Output Layer ใช้ <b>Linear Activation</b></li>
                </ul>
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 40px;">แหล่งที่มา | อ้างอิง</h1>', unsafe_allow_html=True)
st.markdown('''
    <p style="font-size: 20px;">
            Neural Network คืออะไร novelbiz.co.th
        <a href="https://www.novelbiz.co.th/neural-network/" 
           target="_blank" style="font-size: 25px; color: blue;">
           Click here
        </a>. <br>
            aws.amazon.com
        <a href="https://aws.amazon.com/th/what-is/neural-network/" 
           target="_blank" style="font-size: 25px; color: blue;">
           Click here
        </a>. <br>
            Regression คืออะไร มาเรียนรู้แบบง่าย ๆ สไตล์เด็กวิทย์คอมกัน (ตอนที่ 1)
        <a href="https://medium.com/@2pm.tayyoshi/regression-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-%E0%B8%A1%E0%B8%B2%E0%B9%80%E0%B8%A3%E0%B8%B5%E0%B8%A2%E0%B8%99%E0%B8%A3%E0%B8%B9%E0%B9%89%E0%B9%81%E0%B8%9A%E0%B8%9A%E0%B8%87%E0%B9%88%E0%B8%B2%E0%B8%A2-%E0%B9%86-%E0%B8%AA%E0%B9%84%E0%B8%95%E0%B8%A5%E0%B9%8C%E0%B9%80%E0%B8%94%E0%B9%87%E0%B8%81%E0%B8%A7%E0%B8%B4%E0%B8%97%E0%B8%A2%E0%B9%8C%E0%B8%84%E0%B8%AD%E0%B8%A1%E0%B8%81%E0%B8%B1%E0%B8%99-%E0%B8%95%E0%B8%AD%E0%B8%99%E0%B8%97%E0%B8%B5%E0%B9%88-1-4fc7bdd57c13" 
           target="_blank" style="font-size: 25px; color: blue;">
           Click here
        </a>. 
        <br>
    </p>
''', unsafe_allow_html=True)





