from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from sklearn.externals import joblib
import pandas as pd
import numpy as np
schoolname_array = ["กมลาไสย", "กันทรลักษ์วิทยา", "กัลยาณวัตร", "กาญจนาภิเษกวิทยาลัย",
                    "กาฬสินธุ์พิทยาสรรพ์", "กุมภวาปี", "ขอนแก่นวิทยายน", "ขอนแก่นวิทยายน", "ขอนแก่นวิทยายน ", "ขามแก่นนคร",
                    "จันทรุเบกษาอนุสรณ์", "จุฬาภรณราชวิทยาลัย เลย", "ชนบทศึกษา", "ชัยภูมิภักดีชุมพล", "ชุมพลโพนพิสัย", "ชุมแพศึกษา",
                    "ดอนบอสโก", "ท่าบ่อ", "ธาตุนารายณ์วิทยา", "นครขอนแก่น", "นางรอง", "สาธิตมหาวิทยาลัยขอนแก่น (ศึกษาศาสตร์)",
                    "น้ำพองศึกษา", "บัวขาว", "บุญวัฒนา", "บุรีรัมย์พิทยาคม", "บ้านดุงวิทยา", "บ้านผือพิทยาสรรค์", "บ้านไผ่", "ปทุมรัตต์พิทยาคม",
                    "ปทุมเทพวิทยาคาร", "ประจักษ์ศิลปาคาร", "ประโคนชัยพิทยาคม", "ปากช่อง", "ปิยะมหาราชาลัย", "ผดุงนารี", "พิมายวิทยา",
                    "ภูเขียว", "ภูเวียงวิทยาคม", "มหาวิทยาลัยขอนแก่น", "มหาไถ่ศึกษาภาคตะวันออกเฉียงเหนือ", "มัญจาศึกษา", "มารีย์วิทยา", "มุกดาหาร",
                    "ร.ร.สุรนารีวิทยา", "ราชสีมาวิทยาลัย", "ร่องคำ", "ร้อยเอ็ดวิทยาลัย", "ลำปลายมาศ", "วาปีปทุม", "วิทยาลัยอาชีวศึกษาขอนแก่น",
                    "ศรีกระนวนวิทยาคม", "ศรีสงครามวิทยา", "ศรีสะเกษวิทยาลัย", "สกลราชวิทยานุกูล", "สตรีชัยภูมิ", "สตรีราชินูทิศ", "สตรีศึกษา", "สตรีสิริเกศ",
                    "สมเด็จพิทยาคม", "สาธิตมหาวิทยาลัยขอนแก่น (มอดินแดง)", "นารีนุกูล", "สาธิตมหาวิทยาลัยมหาสารคาม", "สารคามพิทยาคม", "สิรินธร", "ชัยภูมภักดีชุมพล",
                    "เสลภูมิพิทยาคม", "สุรธรรมพิทักษ์", "สุรนารีวิทยา", "สุรวิทยาคาร", "สุวรรณภูมิพิทยไพศาล", "หนองคายวิทยาคาร", "หนองเรือวิทยา", "อนุกูลนารี",
                    "อุดรพัฒนาการ", "อุดรพิชัยรักษ์พิทยา", "อุดรพิทยานุกูล", "อุบลรัตน์พิทยาคม", "เซนต์เมรี่", "เดชอุดม", "เทศบาลวัดกลาง", "เบ็ญจะมะมหาราช", "เมืองพลพิทยาคม",
                    "เลยพิทยาคม", "สีชมพูศึกษา", "แก่นนครวิทยาลัย", "แก้งคร้อวิทยา", "โกสุมวิทยาสรรค์", "โพนทองพัฒนาวิทยา", "โรงเรียนปทุมเทพวิทยาคาร"]
id_school_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
school_province_array = ['กรุงเทพมหานคร', 'กาฬสินธุ์', 'ขอนแก่น', 'ชัยภูมิ', 'นครพนม', 'นครราชสีมา', 'บุรีรัมย์', 'มหาสารคาม',
                         'มุกดาหาร', 'ร้อยเอ็ด', 'ศรีสะเกษ', 'สกลนคร', 'สุรินทร์', 'หนองคาย', 'อุดรธานี', 'อุบลราชธานี', 'เลย']
id_school_province = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

app = Flask(__name__)
api = Api(app)
# Model
model = joblib.load('DT_model_student.pkl')


class Iris(Resource):

    def get(self):
        return {"greeting": "Hello From student prediction"}

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ey')
        parser.add_argument('fa')
        parser.add_argument('sn')
        parser.add_argument('sp')
        parser.add_argument('eg')
        parser.add_argument('sex')
        args = parser.parse_args()

        def check_entry_type(str_entry_type):
            if str_entry_type == "สอบคัดเลือกประเภทโควตาภาคตะวันออกเฉียงเหนือ":
                Entry_type = 2
            elif str_entry_type == 'สอบคัดเลือกจากระบบกลาง(Admissions)':
                Entry_type = 1
            elif str_entry_type == 'การคัดเลือกโดยวิธีพิเศษ':
                Entry_type = 0
            elif str_entry_type == 'โครงการรับนักเรียนที่เป็นผู้มีคุณธรรม จริยธรรม และบริการสังคม':
                Entry_type = 3
            return Entry_type

        def check_sex(str_sex):
            if str_sex == 'f':
                sex = 0
            if str_sex == 'm':
                sex = 1
            return sex

        def check_entrygpa(GPA):
            GPA = float(GPA)
            # print(type(GPA))
            if GPA >= 3:
                str_GPA = 1
            elif GPA < 3:
                str_GPA = 0
            return str_GPA

        def check_facultyname(str_faculty):
            if str_faculty == 'คณะทันตแพทยศาสตร์':
                faculty = 0
            elif str_faculty == 'คณะบริหารธุรกิจและการบัญชี':
                faculty = 1
            elif str_faculty == 'คณะพยาบาลศาสตร์':
                faculty = 2
            elif str_faculty == 'คณะมนุษยศาสตร์และสังคมศาสตร์':
                faculty = 3
            elif str_faculty == 'คณะวิทยาการจัดการ':
                faculty = 4
            elif str_faculty == 'คณะวิทยาศาสตร์':
                faculty = 5
            elif str_faculty == 'คณะวิศวกรรมศาสตร์':
                faculty = 6
            elif str_faculty == 'คณะศิลปกรรมศาสตร์':
                faculty = 7
            elif str_faculty == 'คณะศึกษาศาสตร์':
                faculty = 8
            elif str_faculty == 'คณะสถาปัตยกรรมศาสตร์':
                faculty = 9
            elif str_faculty == 'คณะสัตวแพทยศาสตร์':
                faculty = 10
            elif str_faculty == 'คณะสาธารณสุขศาสตร์':
                faculty = 11
            elif str_faculty == 'คณะเกษตรศาสตร์':
                faculty = 12
            elif str_faculty == 'คณะเทคนิคการแพทย์':
                faculty = 13
            elif str_faculty == 'คณะเทคโนโลยี':
                faculty = 14
            elif str_faculty == 'คณะเภสัชศาสตร์':
                faculty = 15
            elif str_faculty == 'คณะเศรษฐศาสตร์':
                faculty = 16
            elif str_faculty == 'คณะแพทยศาสตร์':
                faculty = 17
            elif str_faculty == 'วิทยาลัยการปกครองท้องถิ่น':
                faculty = 18
            elif str_faculty == 'วิทยาเขตหนองคาย':
                faculty = 19
            return faculty

        def check_schoolname(str_school_name):
            check_id = schoolname_array.index(str_school_name)
            arr_id_school_nanme = np.array(id_school_name)

            # print('id school name: ',arr_id_school_nanme[check_id])
            return arr_id_school_nanme[check_id]

        def check_schoolprovince(str_school_province):
            check_id = school_province_array.index(str_school_province)
            arr_id_school_province = np.array(id_school_province)
            # print('id school province: ',arr_id_school_province[check_id])
            return arr_id_school_province[check_id]
        # print(type(args['eg']))
       # d = {'ENTRY_TYPE': args['ey'] , 'FACULTYNAME': args['fa'] , 'SCHOOL_NAME' : args['sn'] , 'SCHOOL_PROVINCE' : args['sp'] , 'ENTRYGPA_3_Less_Up' : args['eg'] , 'STUDENTSEX' : args['sex']}
        x = pd.DataFrame([[check_entry_type(args['ey']), check_facultyname(args['fa']), check_schoolname(args['sn']), check_schoolprovince(args['sp']), check_entrygpa(args['eg']), check_sex(args['sex'])]],
                         columns=['ENTRY_TYPE', 'FACULTYNAME', 'SCHOOL_NAME', 'SCHOOL_PROVINCE', 'ENTRYGPA_3_Less_Up', 'STUDENTSEX'])
        #x = pd.DataFrame(data=d)
        result_model = model.predict(x)
        # print(x)
        # print("predict :", result_model[0])
        result = int(result_model[0])
        if (result == 0):
            str_result = 'สำเร็จการศึกษา'
        elif (result == 1):
            str_result = 'ไม่สำเร็จการศึกษา'
        return {"result": str_result}, 201


api.add_resource(Iris, "/iris")
app.run(debug=True)
