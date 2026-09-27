# คู่มือสถาปัตยกรรมระบบ: DIY Water Contamination FDD System using PCA

เอกสารฉบับนี้จัดทำขึ้นเพื่อให้สมาชิกในทีมวิศวกรรมเข้าใจทิศทาง ภาพรวมการทำงาน สถาปัตยกรรมฮาร์ดแวร์ และหลักการทางคณิตศาสตร์ของโปรเจกต์ทั้งหมดอย่างเป็นรูปธรรม

---

## 1. ภาพรวมโปรเจกต์ (Project Overview & Objectives)

โปรเจกต์นี้คือการพัฒนาระบบ **Fault Detection and Diagnosis (FDD)** สำหรับตรวจจับสารปนเปื้อนในน้ำและตรวจจับความผิดปกติของเซนเซอร์แบบ Real-time โดยประยุกต์ใช้ **Multivariate Statistical Process Control (MSPC)** ด้วยอัลกอริทึม **Principal Component Analysis (PCA)** ร่วมกับระบบ **IoT Telemetry** บนบอร์ดไมโครคอนโทรลเลอร์ ESP32

![Lab Hardware Setup Bench](C:/Users/mekan/.gemini/antigravity/brain/531a46e0-7bc4-42f1-8b0a-55be48e0103a/diy_acrylic_fdd_box_1790486596787.jpg)

### วัตถุประสงค์หลักของระบบ:
 - Early Anomaly Detection: ตรวจจับสิ่งปนเปื้อนในน้ำได้ตั้งแต่ความเข้มข้นต่ำ ก่อนที่สารจะกระจายตัวออกสู่อุปกรณ์ปลายน้ำ
 - Root Cause Isolation: ชี้ชัดว่าเซนเซอร์ตัวใดเป็นสาเหตุหลักของ Anomaly ผ่าน Contribution Plot
 - Alarm Rationalization: ลด False Alarm ที่เกิดจากสัญญาณรบกวน (Noise) หรือการแกว่งตัวตามธรรมชาติของกระบวนการ
 - Software-in-the-Loop Simulation: รองรับทั้งการรับข้อมูลสตรีมมิ่งจริงจาก ESP32 และการรัน Sandbox จำลองทางเคมี

---

## 2. เปรียบเทียบ: ทำไมระบบนี้ถึงดีกว่าแบบดั้งเดิม? (Why PCA over Univariate Threshold?)

ในระบบ SCADA หรือตู้ควบคุมทั่วไป มักใช้ **Univariate Thresholding** (เช่น ตั้ง Hi-Alarm ถ้า TDS > 300 ppm หรือ pH < 6.0) ซึ่งมีข้อจำกัดร้ายแรงในงานอุตสาหกรรม

| มิติการเปรียบเทียบ | ระบบดั้งเดิม (Univariate Threshold) | ระบบใหม่ของเรา (Multivariate PCA FDD) |
| :--- | :--- | :--- |
| **การตรวจจับความสัมพันธ์** | พิจารณาตัวแปรแยกเดี่ยว ไม่เห็นความสัมพันธ์ข้ามตัวแปร | ดักจับ Cross-Correlation ระหว่าง 4 เซนเซอร์พร้อมกัน |
| **สารปนเปื้อนความเข้มข้นต่ำ** | หลุดรอดได้ง่าย หากค่าเดี่ยวยังไม่แตะขีดจำกัด Alarm | ตรวจพบได้ทันที เนื่องจากความสัมพันธ์ของ Baseline เสียรูปทรง |
| **อัตราการเกิด False Alarm** | สูงมาก หาก Sensor ตัวใดตัวหนึ่งมี Noise ชั่วขณะ | ต่ำมาก ประเมินจากสถิติภาพรวม $T^2$ และ $Q$ |
| **การระบุ Root Cause** | ผู้ควบคุมต้องเดาและไล่เปิดเช็คทีละจุด | คำนวณเปอร์เซ็นต์ชี้นิ้วบอก Sensor ต้นเหตุอัตโนมัติ |
| **การเตรียมข้อมูล (Training)** | ต้องมีประวัติค่าผิดปกติและเงื่อนไข Alarm ล่วงหน้า | ใช้ Unsupervised Learning ฝึกเฉพาะตอนน้ำสะอาด (NOC) |

### สรุปจุดเด่นทางวิศวกรรม:
 - Multivariate Correlation: ดักจับค่าผิดปกติซ่อนเร้น กรณีที่ค่า Variable เดี่ยวยังไม่เกิน Threshold
 - Fault Isolation & Diagnosis: ระบุ Root Cause ด้วย Contribution Plot ได้แม่นยำ
 - Alarm Rationalization: ประเมินจากสถิติ Hotelling's $T^2$ และ $Q$ เพื่อลด False Alarm
 - Unsupervised Learning: พัฒนาโมเดลจาก Normal Operating Condition จึงไม่ต้องเสี่ยงฉีดสารพิษเพื่อเก็บข้อมูลจริง

---

## 3. ขอบเขตและข้อจำกัดของโปรเจกต์ (Scope & Constraints)

 - Course Context: วิชา IoT สำหรับวิศวกรรมเคมี (ผู้สอน: ดร. รัชชนนท์ เปี่ยมใจสว่าง)
 - Budget Limit: งบประมาณรวมทั้งโปรเจกต์ไม่เกิน 2,500 บาท
 - Project Timeline: ระยะเวลาพัฒนาและทดสอบทั้งหมด 45 วัน
 - Sensing Dimension: เซนเซอร์วัดคุณสมบัติทางเคมีและกายภาพ 4 ชนิด
 - Compute Architecture: ESP32 ทำหน้าที่ Edge Telemetry ส่วน PCA Analytics รันบน Local Host / Dashboard

---

## 4. โครงสร้างฮาร์ดแวร์และการติดตั้ง (Hardware & Setup Architecture)

ระบบประกอบด้วยถังปฏิกรณ์แบบกวนต่อเนื่อง (Continuous Stirred-Tank Reactor: CSTR) ขนาดกะทัดรัดสำหรับใช้ทดสอบในแล็บ

### รายการอุปกรณ์หลัก:
 - DIY Acrylic Tank: กล่องอะคริลิกใสทรงสี่เหลี่ยมเชื่อมขอบด้วยกาวอะคริลิก พร้อมเจาะรูท่อน้ำเข้า (Inlet) และท่อน้ำล้น (Overflow Outlet)
 - Overhead Motorized Stirrer: มอเตอร์กระแสตรงพร้อมแกนสแตนเลสและใบกวน เพื่อให้สารปนเปื้อนผสมเป็นเนื้อเดียวกันอย่างรวดเร็ว
 - Temperature Sensor: โพรบกันน้ำ DS18B20 สำหรับชดเชยค่าอุณหภูมิของปฏิกิริยาและของเหลว
 - TDS Sensor: โพรบวัด Total Dissolved Solids สำหรับประเมินความเข้มข้นของไอออนในสารละลาย
 - Turbidity Sensor: โพรบวัดความขุ่นแบบแสง (Optical Turbidity) สำหรับตรวจจับอนุภาคแขวนลอย
 - pH Sensor: โพรบแก้วพร้อมโมดูลแปลงสัญญาณ BNC สำหรับวัดความเป็นกรด-ด่าง
 - Microcontroller Unit: บอร์ด ESP32 เชื่อมต่อ Wi-Fi ทำการแปลงสัญญาณ ADC และ Publish ข้อมูลผ่าน MQTT Protocol
 - Power Supply Unit: สวิตชิ่งเพาเวอร์ซัพพลายจ่ายไฟ 12V DC ให้มอเตอร์กวน และ 5V DC ให้บอร์ดไมโครคอนโทรลเลอร์

### ผังการไหลของข้อมูล (Data Pipeline):
 - Sensing: โพรบทั้ง 4 ตัวจุ่มอยู่ในถัง CSTR อ่านค่าและแปลงเป็นสัญญาณไฟฟ้า
 - Edge Ingestion: บอร์ด ESP32 อ่านค่า Sampling ทุก 1 วินาที และคำนวณ Moving Average เพื่อตัด Noise
 - Telemetry: ESP32 แพ็กเกจข้อมูลเป็น JSON Payload แล้ว Publish ขึ้น MQTT Broker ผ่านเครือข่าย Wi-Fi
 - Inference Engine: ระบบคอมพิวเตอร์รับ Telemetry เข้าสู่ Data Pipeline ทำ Scaling และประมวลผลโมเดล PCA
 - Action Trigger: หากสถิติเกินขีดจำกัด UCL ระบบจะเปลี่ยนสถานะเป็น Alarm และไฮไลต์บาร์เซนเซอร์ต้นเหตุ

---

## 5. หลักการของอัลกอริทึม PCA (Algorithm Mathematics)

การทำงานของระบบ FDD แบ่งออกเป็น 2 ขั้นตอนหลัก:

### Phase 1: Model Training (NOC Baseline Formulation)
**ขั้นตอนการทำงานของระบบแบบ Step-by-Step:**
 - Step 1 Data Acquisition: เก็บตัวอย่างข้อมูลเซนเซอร์น้ำสะอาดในถัง CSTR จำนวน 500 Samples เพื่อสร้าง Baseline ตัวแทนสภาวะปกติ
 - Step 2 Feature Standardization: ปรับสเกลข้อมูลให้เป็น Z-Score ด้วย StandardScaler (μ, σ) ป้องกันตัวแปรหน่วยใหญ่กลบหน่วยเล็ก
 - Step 3 Covariance Decomposition: คำนวณ Covariance Matrix และทำ Eigenvalue Decomposition สกัด 2 แกนหลัก (PC1, PC2)
 - Step 4 Scree Plot Analysis: ประเมิน Cumulative Variance เพื่อยืนยันว่า 2 PCs เก็บ Total Variance เกิน 80% ตามเกณฑ์วิศวกรรม
 - Step 5 Threshold Formulation: คำนวณขีดจำกัดบน 99% UCL สำหรับ Hotelling's T² และ SPE / Q-Statistic เพื่อใช้ตรวจจับ Anomaly

### Phase 2: Online Fault Detection & Diagnosis (Real-Time Inference)
**ขั้นตอนการทำงานและการทดสอบแบบ Step-by-Step:**
 - Step 1 Start Telemetry: รับ Telemetry จาก ESP32 เข้าสู่ Data Pipeline และคำนวณ Z-Score แบบ Real-time ทุก 1 วินาที
 - Step 2 Baseline Confirmation: ตรวจสอบว่าในสภาวะน้ำสะอาด ค่า T² และ Q ต้องอยู่ต่ำกว่า UCL และพิกัด Biplot เกาะกลุ่มในวงรี
 - Step 3 Fault Injection: จำลองการปนเปื้อนของสารเคมี (เช่น NaCl, Acetic Acid, Milk Protein หรือ Sensor Drift)
 - Step 4 Anomaly Detection: หากพิกัดหลุดวงรีหรือดัชนี T² หรือ Q ทะลุเส้น UCL ระบบจะส่งสัญญาณ ALARM ทันที
 - Step 5 Root Cause Isolation: อ่านค่าจาก Contribution Plot เพื่อชี้เป้าเซนเซอร์ที่มีสัดส่วนความผิดปกติสูงสุดให้ Operator แก้ไข

---

## 6. พฤติกรรมของสารปนเปื้อนในการทดสอบ (Fault Library Signature)

 - NaCl (สารละลายเกลือ): ค่า TDS พุ่งสูงขึ้นอย่างรุนแรง ตัวแปรอื่นคงที่ $\rightarrow$ ดักจับได้ด้วย $T^2$ และชี้เป้า TDS
 - Acetic Acid (น้ำส้มสายชู): ค่า pH ลดลงฮวบฮาบ และ TDS เพิ่มขึ้นเล็กน้อย $\rightarrow$ ทำลาย Correlation ปกติ ดีดค่า $Q$ ทันที
 - Milk Protein (โปรตีนจากนม): ค่า Turbidity สูงลิ่ว และ pH ดรอปเล็กน้อย $\rightarrow$ ดีดทั้ง $T^2$ และ $Q$ พร้อมระบุ Turbidity
 - Sensor Drift: ค่าอุณหภูมิไต่ระดับผิดธรรมชาติเนื่องจาก Hardware Failure $\rightarrow$ ทริกเกอร์ Alarm เตือนอุปกรณ์ชำรุด
