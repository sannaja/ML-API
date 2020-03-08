from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from sklearn.externals import joblib
import pandas as pd
import numpy as np
school_name_array= ["กมลาไสย","กันทรลักษ์วิทยา","กัลยาณวัตร","กาญจนาภิเษกวิทยาลัย","กาฬสินธุ์พิทยาสรรพ์","กุมภวาปี","ขอนแก่นวิทยายน","ขามแก่นนคร",\
"คำเขื่อนแก้วชนูปถัมภ์","อุบลรัตน์พิทยาคม","จันทรุเบกษาอนุสรณ์","จุฬาภรณราชวิทยาลัย  มุกดาหาร","ชนบทศึกษา","ชัยภูมิภักดีชุมพล","ชุมพลโพนพิสัย","ชุมแพศึกษา",\
"ดอนบอสโก","ดอนบอสโกวิทยา","ท่าบ่อ","ธาตุนารายณ์วิทยา","ธาตุพนม","นครขอนแก่น","นางรอง","นารีนุกูล","น้ำพองศึกษา","บัวขาว","บัวใหญ่","บึงกาฬ",\
"บุญวัฒนา","บุรีรัมย์พิทยาคม","ลำปลายมาศ","บ้านผือพิทยาสรรค์","บ้านไผ่","ปทุมรัตต์พิทยาคม","ปทุมเทพวิทยาคาร","ประจักษ์ศิลปาคาร","ประโคนชัยพิทยาคม",\
"ปากช่อง","ปิยะมหาราชาลัย","ผดุงนารี","พิบูลมังสาหาร","พิบูลวิทยาลัย","พิมายวิทยา","พุทไธสง","ภูเขียว","ภูเวียงวิทยาคม","มหาวิทยาลัยขอนแก่น","มหาไถ่ศึกษาภาคตะวันออกเฉียงเหนือ",\
"มัญจาศึกษา","มัธยมวานรนิวาส","มารีย์วิทยา","มุกดาหาร","ยโสธรพิทยาคม","ร.ร.สุรนารีวิทยา","รัตนบุรี","ราชสีมาวิทยาลัย","ร่องคำ","ร้อยเอ็ดวิทยาลัย","บ้านดุงวิทยา",\
"วาปีปทุม","วิทยาลัยการสาธารณสุขสิรินธร จังหวัดขอนแก่น","วิทยาลัยอาชีวศึกษาขอนแก่น","โพนทองพัฒนาวิทยา","วิทยาลัยเทคนิคอุดรธานี","ศรีกระนวนวิทยาคม","ศรีบุญเรืองวิทยาคาร",\
"ศรีสงครามวิทยา","ศรีสะเกษวิทยาลัย","สกลราชวิทยานุกูล","สตรีชัยภูมิ","สตรีราชินูทิศ","สตรีศึกษา","สตรีสิริเกศ","เดชอุดม","สถาบันเทคโนโลยีราชมงคล วิทยาเขตขอนแก่น",\
"สระบุรีวิทยาคม","สมเด็จพิทยาคม","หนองบัวพิทยาคาร","สาธิตมหาวิทยาลัยขอนแก่น (มอดินแดง)","สาธิตมหาวิทยาลัยขอนแก่น (ศึกษาศาสตร์)","สาธิตมหาวิทยาลัยมหาสารคาม","สารคามพิทยาคม",\
"สิรินธร","สีชมพูศึกษา","สุรธรรมพิทักษ์","สุรนารีวิทยา","สุรวิทยาคาร","สุวรรณภูมิพิทยไพศาล","หนองคายวิทยาคาร","สาธิตมหาวิทยาลัยขอนแก่น",\
"หนองเรือวิทยา","อนุกูลนารี","อมตวิทยา","อัสสัมชัญ นครราชสีมา","อำนาจเจริญ","อุดรพัฒนาการ","อุดรพิชัยรักษ์พิทยา","อุดรพิทยานุกูล","คำแสนวิทยาสรรค์","เฉลิมพระเกียรติสมเด็จพระศรีนครินทร์",\
"เซนต์เมรี่","สถาบันเทคโนโลยีราชมงคล","เตรียมอุดมศึกษา","เทคโนโลยีภาคตะวันออกเฉียงเหนือ","เทศบาลวัดกลาง","เบ็ญจะมะมหาราช","เมืองพลพิทยาคม","เรณูนครวิทยานุกูล",\
"เลยพิทยาคม","เสลภูมิพิทยาคม","แก่นนครวิทยาลัย","แก้งคร้อวิทยา","โกสุมวิทยาสรรค์","วิทยาลัยเทคนิคขอนแก่น","โรงเรียนปทุมเทพวิทยาคาร","โรงเรียนสุรวิทยาคาร","โรงเรียนอื่นๆ","โรงเรียนเทศบาลวัดกลาง"]

school_name_array_use_gpa= ["ชุมพลโพนพิสัย","ชุมแพศึกษา","ดอนบอสโก","ดอนบอสโกวิทยา","ท่าบ่อ","ธาตุนารายณ์วิทยา","เรณูนครวิทยานุกูล","นครขอนแก่น","ประจักษ์ศิลปาคาร",\
"นารีนุกูล","น้ำพองศึกษา","บัวขาว","ธาตุพนม","เซนต์เมรี่","บุญวัฒนา","บุรีรัมย์พิทยาคม","บ้านดุงวิทยา","บ้านผือพิทยาสรรค์","บ้านไผ่","ปทุมรัตต์พิทยาคม","ปทุมเทพวิทยาคาร",\
"นางรอง","ประโคนชัยพิทยาคม","ปากช่อง","ปิยะมหาราชาลัย","ผดุงนารี","รัตนบุรี","พิบูลวิทยาลัย","พิมายวิทยา","พุทไธสง","ภูเขียว","ภูเวียงวิทยาคม","มหาวิทยาลัยขอนแก่น",\
"มหาไถ่ศึกษาภาคตะวันออกเฉียงเหนือ","มัญจาศึกษา","มัธยมวานรนิวาส","มารีย์วิทยา","มุกดาหาร","ยโสธรพิทยาคม","ร.ร.สุรนารีวิทยา","พิบูลมังสาหาร","ราชสีมาวิทยาลัย","คำแสนวิทยาสรรค์",\
"ร้อยเอ็ดวิทยาลัย","ลำปลายมาศ","วาปีปทุม","วิทยาลัยการสาธารณสุขสิรินธร จังหวัดขอนแก่น","วิทยาลัยอาชีวศึกษาขอนแก่น","วิทยาลัยเทคนิคขอนแก่น","วิทยาลัยเทคนิคอุดรธานี","ศรีกระนวนวิทยาคม",\
"ศรีบุญเรืองวิทยาคาร","ศรีสงครามวิทยา","ศรีสะเกษวิทยาลัย","สกลราชวิทยานุกูล","สตรีชัยภูมิ","สตรีราชินูทิศ","สตรีศึกษา","สตรีสิริเกศ","สถาบันเทคโนโลยีราชมงคล","สถาบันเทคโนโลยีราชมงคล วิทยาเขตขอนแก่น",\
"สมเด็จพิทยาคม","สระบุรีวิทยาคม","สาธิตมหาวิทยาลัยขอนแก่น","สาธิตมหาวิทยาลัยขอนแก่น (มอดินแดง)","สาธิตมหาวิทยาลัยขอนแก่น (ศึกษาศาสตร์)","สาธิตมหาวิทยาลัยมหาสารคาม","สารคามพิทยาคม",\
"สิรินธร","สีชมพูศึกษา","สุรธรรมพิทักษ์","สุรนารีวิทยา","สุรวิทยาคาร","เทศบาลวัดกลาง","หนองคายวิทยาคาร","หนองบัวพิทยาคาร","หนองเรือวิทยา","อนุกูลนารี",\
"อมตวิทยา","อัสสัมชัญ นครราชสีมา","อำนาจเจริญ","อุดรพัฒนาการ","อุดรพิชัยรักษ์พิทยา","อุดรพิทยานุกูล","อุบลรัตน์พิทยาคม","เฉลิมพระเกียรติสมเด็จพระศรีนครินทร์","บึงกาฬ","เดชอุดม",\
"เตรียมอุดมศึกษา","เทคโนโลยีภาคตะวันออกเฉียงเหนือ","สุวรรณภูมิพิทยไพศาล","เบ็ญจะมะมหาราช","เมืองพลพิทยาคม","บัวใหญ่","เลยพิทยาคม","เสลภูมิพิทยาคม","แก่นนครวิทยาลัย","แก้งคร้อวิทยา",\
"โกสุมวิทยาสรรค์","โพนทองพัฒนาวิทยา","โรงเรียนปทุมเทพวิทยาคาร","โรงเรียนสุรวิทยาคาร","โรงเรียนอื่นๆ","โรงเรียนเทศบาลวัดกลาง"]

id_school_name_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,\
41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,84,\
85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118]

faculty_name_array=["คณะทันตแพทยศาสตร์","คณะนิติศาสตร์","คณะบริหารธุรกิจและการบัญชี","คณะพยาบาลศาสตร์","คณะมนุษยศาสตร์และสังคมศาสตร์","คณะวิทยาการจัดการ",\
"คณะวิทยาศาสตร์","คณะวิศวกรรมศาสตร์","คณะศิลปกรรมศาสตร์","คณะศึกษาศาสตร์","คณะสถาปัตยกรรมศาสตร์","คณะสัตวแพทยศาสตร์","คณะสาธารณสุขศาสตร์","คณะเกษตรศาสตร์",\
"คณะเทคนิคการแพทย์","คณะเทคโนโลยี","คณะเภสัชศาสตร์","คณะเศรษฐศาสตร์","คณะแพทยศาสตร์","วิทยาลัยการปกครองท้องถิ่น","วิทยาลัยนานาชาติ","วิทยาเขตหนองคาย","คณะวิทยาศาสตร์ประยุกต์และวิศวกรรมศาสตร์ วิทยาเขตหนองคาย",\
"คณะสังคมศาสตร์บูรณาการ  วิทยาเขตหนองคาย","คณะบริหารธุรกิจ  วิทยาเขตหนองคาย","คณะศิลปศาสตร์ วิทยาเขตหนองคาย"]

id_faculty_name_array=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,21,21,21,21]

education_array = ["ประถมศึกษา","ปริญญาตรี","ปริญญาโท","ปริญญาเอก","มัธยมศึกษาตอนต้น","มัธยมศึกษาตอนปลาย","อนุปริญญาหรือเทียบเท่า","อาชีวศึกษา","อื่นๆ ระบุ"]
id_father_education_array=[0,1,3,2,4,5,6,7,8]
id_mother_education_array=[0,1,3,2,4,5,6,7,8]
id_mother_education_array_use_gpa=[0,1,3,2,4,5,6,7,8]

father_ocupation_array=["เกษตรกร/ประมง","เจ้าของหรือผู้ประกอบธุรกิจส่วนตัว,ค้าขาย","พนักงานรัฐวิสาหกิจ","พนักงานหรือลูกจ้างเอกชน","ไม่ประกอบอาชีพ","รับจ้าง",\
"รับราชการ ( ข้าราชการ, ลูกจ้างประจำ)","รับราชการ ( ลูกจ้างชั่วคราว )","อื่นๆ ระบุ"]
id_father_ocupation_array=[6,7,0,1,8,2,3,4,5]
mother_ocupation_array=["เกษตรกร/ประมง","เจ้าของหรือผู้ประกอบธุรกิจส่วนตัว,ค้าขาย","พนักงานรัฐวิสาหกิจ","พนักงานหรือลูกจ้างเอกชน","แม่บ้าน","ไม่ประกอบอาชีพ","รับจ้าง","รับราชการ ( ลูกจ้างชั่วคราว )","อื่นๆ ระบุ","รับราชการ ( ข้าราชการ, ลูกจ้างประจำ )"]
id_mother_ocupation_array=[6,7,0,1,8,9,2,3,4,5]
id_mother_ocupation_array_use_gpa=[6,7,0,1,8,9,2,3,4,5]
app = Flask(__name__)
api = Api(app)


# Model
model = joblib.load('SVM_model_student_V3.pkl')
model_gpa =joblib.load('SVM_model_student_V3_use_gpa.pkl')
class student_predict(Resource):
    def get(self):        
        return {"student_predication":"Welcome to API for student prediction perfomance no use GPA university year 1 semesters 2",
                "How": "Use Post and add 10 parameter as follows",
                "Key_1":"ADMISSIONS_TYPE",
                "Key_2":"FACULTY",
                "Key_3":"SCHOOL_NAME",
                "Key_4":"ENTRY_GPA",
                "Key_5":"YEAR_COME",
                "Key_6":"STUDENT_GENDER",
                "Key_7":"FATHER_EDUCATION",
                "Key_8":"FATHER_OCUPATION",
                "Key_9":"MOTHER_EDUCATION",
                "Key_10":"MOTHER_OCUPATION"}

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ADMISSIONS_TYPE')
        parser.add_argument('FACULTY')
        parser.add_argument('SCHOOL_NAME')
        parser.add_argument('ENTRY_GPA')
        parser.add_argument('YEAR_COME')
        parser.add_argument('STUDENT_GENDER')
        parser.add_argument('FATHER_EDUCATION')
        parser.add_argument('FATHER_OCUPATION')
        parser.add_argument('MOTHER_EDUCATION')
        parser.add_argument('MOTHER_OCUPATION')
        args = parser.parse_args()
        
        def check_admissions_type(str_entry_type):
            if str_entry_type == "สอบคัดเลือกประเภทโควตาภาคตะวันออกเฉียงเหนือ":
                Entry_type = 3
            elif str_entry_type == 'สอบคัดเลือกจากระบบกลาง(Admissions)':
                Entry_type = 2
            elif str_entry_type == 'มหาวิทยาลัยจัดสอบ (โครงการพิเศษ)':
                Entry_type = 1
            elif str_entry_type == 'การคัดเลือกโดยวิธีพิเศษ':
                Entry_type = 0
            else:
                Entry_type = 4
            return Entry_type

        def check_sex(str_sex):
            if str_sex == 'f' or str_sex == 'F' or str_sex == "หญิง" or str_sex == "เพศหญิง":
                sex = 0
            elif str_sex == 'm'or str_sex == 'M' or str_sex == "ชาย" or str_sex == "เพศชาย":
                sex = 1
            else:
                sex = 'Sex ไม่ถูกต้อง'
            return sex

        def check_entry_gpa(GPA):
            try:
                GPA = float(GPA)
                print(type(GPA))
                if GPA >= 3.5 and GPA <= 4:
                    str_GPA = 3
                elif GPA >= 3 and GPA < 3.5:
                    str_GPA = 2
                elif GPA >= 2.5 and GPA < 3:
                    str_GPA = 0
                elif GPA >= 2 and GPA < 2.5:
                    str_GPA = 4
                elif GPA >= 0 and GPA < 2:
                    str_GPA = 1
                else :
                    str_GPA = 'GPA ไม่ถูกต้อง'
                return str_GPA
            except:
                str_GPA = 'GPA ไม่รองรับข้อความ'
                return str_GPA
            
        def check_faculty_name(str_faculty):
            try:
                check_id = faculty_name_array.index(str_faculty)
                arr_id_faculty_name = np.array(id_faculty_name_array)
                return arr_id_faculty_name[check_id]
            except:
                re_faculty_name = 'faculty name ไม่ถูกต้อง'
                return re_faculty_name
            
        def check_school_name(str_school_name):
            try:
                check_id = school_name_array.index(str_school_name)
                arr_id_school_nanme = np.array(id_school_name_array)
                print('id school name: ',arr_id_school_nanme[check_id])
                return arr_id_school_nanme[check_id]
            except:
                re_school_name = 117
                return re_school_name

        def check_year_come(str_year_come):
            try:
                year_come = int(str_year_come)
                return year_come
            except:
                re_year_come = 'อายุไม่ถูกต้อง'
                return re_year_come

        def check_father_education(str_father_education):
            try:
                check_id = education_array.index(str_father_education)
                arr_id_father_education = np.array(id_father_education_array)
                return arr_id_father_education[check_id]
            except:
                re_father_education = 'ระดับการศึกษาของบิดาไม่ถูกต้อง'
                return re_father_education

        def check_mother_education(str_mother_education):
            try:
                check_id = education_array.index(str_mother_education)
                arr_id_mother_education = np.array(id_mother_education_array)
                return arr_id_mother_education[check_id]
            except:
                re_mother_education = 'ระดับการศึกษาของมารดาไม่ถูกต้อง'
                return re_mother_education

        def check_father_ocupation(str_father_ocupation):
            try:
                check_id = father_ocupation_array.index(str_father_ocupation)
                arr_id_father_ocupation_array = np.array(id_father_ocupation_array)
                return arr_id_father_ocupation_array[check_id]
            except:
                re_father_ocupation = 'อาชีพบิดาไม่ถูกต้อง'
                return re_father_ocupation

        def check_mother_ocupation(str_mother_eocupation):
            try:
                check_id = mother_ocupation_array.index(str_mother_eocupation)
                arr_id_mother_ocupation_array = np.array(id_mother_ocupation_array)
                return arr_id_mother_ocupation_array[check_id]
            except:
                re_mother_education = 'อาชีพมารดาไม่ถูกต้อง'
                return re_mother_education 
         
        id_admissions_type = check_admissions_type(args['ADMISSIONS_TYPE'])
        id_faculty_name = check_faculty_name(args['FACULTY'])
        id_school_name = check_school_name(args['SCHOOL_NAME'])
        id_entry_gpa = check_entry_gpa(args['ENTRY_GPA'])
        id_sex = check_sex(args['STUDENT_GENDER'])
        id_year_come = check_year_come(args['YEAR_COME'])
        id_father_education = check_father_education(args['FATHER_EDUCATION'])
        id_father_ocupation = check_father_ocupation(args['FATHER_OCUPATION'])
        id_mother_education = check_mother_education(args['MOTHER_EDUCATION'])
        id_mother_ocupation = check_mother_ocupation(args['MOTHER_OCUPATION'])
        print('admissions_type_str : ',id_admissions_type)
        print('facultyname_str :',id_faculty_name)
        print('schoolname_str :',id_school_name)
        print('entry_gpa_str : ',id_entry_gpa)
        print('sex_str :',id_sex)
        print('year_come_str : ',id_year_come)
        print('father_education_str : ',id_father_education)
        print('father_ocupation_str : ',id_father_ocupation)
        print('mother_education_str : ',id_mother_education)
        print('mother_ocupation_str : ',id_mother_ocupation)
        check_str_admissions_type = isinstance(id_admissions_type, str)
        check_str_faculty_name = isinstance(id_faculty_name, str) 
        check_str_school_name = isinstance(id_school_name, str) 
        check_str_entry_gpa = isinstance(id_entry_gpa, str) 
        check_str_sex = isinstance(id_sex, str) 
        check_str_year_come = isinstance(id_year_come, str)
        check_str_id_father_education = isinstance(id_father_education, str)
        check_str_id_father_ocupation = isinstance(id_father_ocupation, str)
        check_str_id_mother_education = isinstance(id_mother_education, str)
        check_str_id_mother_ocupation = isinstance(id_mother_ocupation, str)
        
        if(check_str_admissions_type == False and check_str_faculty_name == False and check_str_school_name == False and check_str_entry_gpa == False \
        and check_str_sex == False and check_str_year_come == False and check_str_id_father_education == False and check_str_id_father_ocupation == False\
        and check_str_id_mother_education == False and check_str_id_mother_ocupation == False):

             # d = {'ENTRY_TYPE': args['ey'] , 'FACULTYNAME': args['fa'] , 'SCHOOL_NAME' : args['sn'] , 'SCHOOL_PROVINCE' : args['sp'] , 'ENTRYGPA_3_Less_Up' : args['eg'] , 'STUDENTSEX' : args['sex']}
            x = pd.DataFrame([[id_admissions_type ,id_faculty_name,id_school_name,id_entry_gpa,id_year_come,id_sex,id_father_education,id_father_ocupation,id_mother_education,id_mother_ocupation]] ,\
            columns=['ADMISSIONS_TYPE', 'FACULTY', 'SCHOOL_NAME', 'ENTRY_GPA','YEAR_COME','STUDENT_GENDER','FATHER_EDUCATION','FATHER_OCUPATION','MOTHER_EDUCATION','MOTHER_OCUPATION'])
            #x = pd.DataFrame(data=d)
            result_model = model.predict(x)
            print(x)
            print("predict :", result_model[0])
            result= int(result_model[0])
            if (result == 0):
                str_result = 'สำเร็จการศึกษา'
            elif (result == 1):
                str_result = 'ไม่สำเร็จการศึกษา'
            return {"result": str_result}, 201
        else:
            if check_str_faculty_name == True:
                str_result = 'Faculty name ไม่ถูกต้อง'
                return {"result": str_result}, 201
            elif check_str_entry_gpa == True:
                str_result = 'Entry GPA ไม่ถูกต้อง หรือ Entry GPA ไม่รองรับข้อความ'
                return {"result": str_result}, 201
            elif check_str_sex == True:
                str_result = 'Gender ไม่ถูกต้อง'
                return {"result": str_result}, 201
            elif check_str_year_come == True:
                str_result = 'Year come ไม่รองรับข้อความ กรุณากรองเป็นตัวเลข'
                return {"result": str_result}, 201
            elif check_str_id_father_education == True:
                str_result = 'ระดับการศึกษาของบิดาไม่ถูกต้องหรือไม่รองรับระดับการศึกษาในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_father_ocupation == True:
                str_result = 'อาชีพของบิดาไม่ถูกต้องหรือไม่รองรับอาชีพในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_mother_education == True:
                str_result = 'ระดับการศึกษาของมารดาไม่ถูกต้องหรือไม่รองรับระดับการศึกษาในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_mother_ocupation == True:
                str_result = 'อาชีพของมารดาไม่ถูกต้องหรือไม่รองรับอาชีพในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_school_name == True:
                str_result = 'ชื่อโรงเรียนไม่ถูกต้อง'
                return {"result": str_result}, 201
            elif check_str_admissions_type == True:
                str_result = 'Admissions type ไม่ถูกต้อง'
                return {"result": str_result}, 201
api.add_resource(student_predict, "/student_predict")

class student_predict_gpa(Resource):
    def get(self): 
        return {"student_predication":"Welcome to API for student prediction perfomance use GPA university year 1 semesters 2",
                "How": "Use Post and add 11 parameter as follows",
                "Key_1":"ADMISSIONS_TYPE",
                "Key_2":"FACULTY",
                "Key_3":"SCHOOL_NAME",
                "Key_4":"ENTRY_GPA",
                "Key_5":"YEAR_COME",
                "Key_6":"STUDENT_GENDER",
                "Key_7":"FATHER_EDUCATION",
                "Key_8":"FATHER_OCUPATION",
                "Key_9":"MOTHER_EDUCATION",
                "Key_10":"MOTHER_OCUPATION",
                "Key_11":"GPA_YEAR_1_SEMESTER_2"}
                
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ADMISSIONS_TYPE')
        parser.add_argument('FACULTY')
        parser.add_argument('SCHOOL_NAME')
        parser.add_argument('ENTRY_GPA')
        parser.add_argument('YEAR_COME')
        parser.add_argument('STUDENT_GENDER')
        parser.add_argument('FATHER_EDUCATION')
        parser.add_argument('FATHER_OCUPATION')
        parser.add_argument('MOTHER_EDUCATION')
        parser.add_argument('MOTHER_OCUPATION')
        parser.add_argument('GPA_YEAR_1_SEMESTER_2')
        args = parser.parse_args()
        
        def check_admissions_type(str_entry_type):
            if str_entry_type == "สอบคัดเลือกประเภทโควตาภาคตะวันออกเฉียงเหนือ":
                Entry_type = 3
            elif str_entry_type == 'สอบคัดเลือกจากระบบกลาง(Admissions)':
                Entry_type = 2
            elif str_entry_type == 'มหาวิทยาลัยจัดสอบ (โครงการพิเศษ)':
                Entry_type = 1
            elif str_entry_type == 'การคัดเลือกโดยวิธีพิเศษ':
                Entry_type = 0
            else:
                Entry_type = 4
            return Entry_type

        def check_sex(str_sex):
            if str_sex == 'f' or str_sex == 'F' or str_sex == "หญิง" or str_sex == "เพศหญิง":
                sex = 0
            elif str_sex == 'm'or str_sex == 'M' or str_sex == "ชาย" or str_sex == "เพศชาย":
                sex = 1
            else:
                sex = 'Sex ไม่ถูกต้อง'
            return sex

        def check_entry_gpa(GPA):
            try:
                GPA = float(GPA)
                print(type(GPA))
                if GPA >= 3.5 and GPA <= 4:
                    str_GPA = 3
                elif GPA >= 3 and GPA < 3.5:
                    str_GPA = 2
                elif GPA >= 2.5 and GPA < 3:
                    str_GPA = 0
                elif GPA >= 2 and GPA < 2.5:
                    str_GPA = 4
                elif GPA >= 0 and GPA < 2:
                    str_GPA = 1
                else :
                    str_GPA = 'GPA ไม่ถูกต้อง'
                return str_GPA
            except:
                str_GPA = 'GPA ไม่รองรับข้อความ'
                return str_GPA
            
        def check_faculty_name(str_faculty):
            try:
                check_id = faculty_name_array.index(str_faculty)
                arr_id_faculty_name = np.array(id_faculty_name_array)
                return arr_id_faculty_name[check_id]
            except:
                re_faculty_name = 'faculty name ไม่ถูกต้อง'
                return re_faculty_name
            
        def check_school_name(str_school_name):
            try:
                check_id = school_name_array_use_gpa.index(str_school_name)
                arr_id_school_nanme = np.array(id_school_name_array)
                print('id school name: ',arr_id_school_nanme[check_id])
                return arr_id_school_nanme[check_id]
            except:
                re_school_name = 117
                return re_school_name
        def check_year_come(str_year_come):
            try:
                year_come = int(str_year_come)
                return year_come
            except:
                re_year_come = 'อายุไม่ถูกต้อง'
                return re_year_come

        def check_father_education(str_father_education):
            try:
                check_id = education_array.index(str_father_education)
                arr_id_father_education = np.array(id_father_education_array)
                return arr_id_father_education[check_id]
            except:
                re_father_education = 'ระดับการศึกษาของบิดาไม่ถูกต้อง'
                return re_father_education

        def check_mother_education(str_mother_education):
            try:
                check_id = education_array.index(str_mother_education)
                arr_id_mother_education = np.array(id_mother_education_array_use_gpa)
                return arr_id_mother_education[check_id]
            except:
                re_mother_education = 'ระดับการศึกษาของมารดาไม่ถูกต้อง'
                return re_mother_education

        def check_father_ocupation(str_father_ocupation):
            try:
                check_id = father_ocupation_array.index(str_father_ocupation)
                arr_id_father_ocupation_array = np.array(id_father_ocupation_array)
                return arr_id_father_ocupation_array[check_id]
            except:
                re_father_ocupation = 'อาชีพบิดาไม่ถูกต้อง'
                return re_father_ocupation

        def check_mother_ocupation(str_mother_ocupation):
            try:
                check_id = mother_ocupation_array.index(str_mother_ocupation)
                arr_id_mother_ocupation_array = np.array(id_mother_ocupation_array_use_gpa)
                return arr_id_mother_ocupation_array[check_id]
            except:
                re_mother_education = 'อาชีพมารดาไม่ถูกต้อง'
                return re_mother_education 

        def check_gpa_year_1_semester_2(str_gpa_year_1_semester_2):
            try:
                GPA_1_2 = float(str_gpa_year_1_semester_2)
                print(type(GPA_1_2))
                if GPA_1_2 >= 3.5 and GPA_1_2 <= 4:
                    str_GPA_1_2 = 3
                elif GPA_1_2 >= 3 and GPA_1_2 < 3.5:
                    str_GPA_1_2 = 2
                elif GPA_1_2 >= 2.5 and GPA_1_2 < 3:
                    str_GPA_1_2 = 0
                elif GPA_1_2 >= 2 and GPA_1_2 < 2.5:
                    str_GPA_1_2 = 4
                elif GPA_1_2 >= 1.5 and GPA_1_2 < 2:
                    str_GPA_1_2 = 1
                elif GPA_1_2 >= 0 and GPA_1_2 < 1.5:
                    str_GPA_1_2 = 5   
                else :
                    str_GPA_1_2 = 'GPA_1_2 ไม่ถูกต้อง'
                return str_GPA_1_2
            except:
                str_GPA_1_2 = 'GPA_1_2 ไม่รองรับข้อความ'
                return str_GPA_1_2

        id_admissions_type = check_admissions_type(args['ADMISSIONS_TYPE'])
        id_faculty_name = check_faculty_name(args['FACULTY'])
        id_school_name = check_school_name(args['SCHOOL_NAME'])
        id_entry_gpa = check_entry_gpa(args['ENTRY_GPA'])
        id_sex = check_sex(args['STUDENT_GENDER'])
        id_year_come = check_year_come(args['YEAR_COME'])
        id_father_education = check_father_education(args['FATHER_EDUCATION'])
        id_father_ocupation = check_father_ocupation(args['FATHER_OCUPATION'])
        id_mother_education = check_mother_education(args['MOTHER_EDUCATION'])
        id_mother_ocupation = check_mother_ocupation(args['MOTHER_OCUPATION'])
        id_gpa_1_2 = check_gpa_year_1_semester_2(args['GPA_YEAR_1_SEMESTER_2'])
        print('admissions_type_str : ',id_admissions_type)
        print('facultyname_str :',id_faculty_name)
        print('schoolname_str :',id_school_name)
        print('entry_gpa_str : ',id_entry_gpa)
        print('sex_str :',id_sex)
        print('year_come_str : ',id_year_come)
        print('father_education_str : ',id_father_education)
        print('father_ocupation_str : ',id_father_ocupation)
        print('mother_education_str : ',id_mother_education)
        print('mother_ocupation_str : ',id_mother_ocupation)
        print('GPA_1_2_str : ',id_gpa_1_2)
        check_str_admissions_type = isinstance(id_admissions_type, str)
        check_str_faculty_name = isinstance(id_faculty_name, str) 
        check_str_school_name = isinstance(id_school_name, str) 
        check_str_entry_gpa = isinstance(id_entry_gpa, str) 
        check_str_sex = isinstance(id_sex, str) 
        check_str_year_come = isinstance(id_year_come, str)
        check_str_id_father_education = isinstance(id_father_education, str)
        check_str_id_father_ocupation = isinstance(id_father_ocupation, str)
        check_str_id_mother_education = isinstance(id_mother_education, str)
        check_str_id_mother_ocupation = isinstance(id_mother_ocupation, str)
        check_str_id_gpa_1_2 = isinstance(id_gpa_1_2, str)
        
        if(check_str_admissions_type == False and check_str_faculty_name == False and check_str_school_name == False and check_str_entry_gpa == False \
        and check_str_sex == False and check_str_year_come == False and check_str_id_father_education == False and check_str_id_father_ocupation == False\
        and check_str_id_mother_education == False and check_str_id_mother_ocupation == False and check_str_id_gpa_1_2 == False):

             # d = {'ENTRY_TYPE': args['ey'] , 'FACULTYNAME': args['fa'] , 'SCHOOL_NAME' : args['sn'] , 'SCHOOL_PROVINCE' : args['sp'] , 'ENTRYGPA_3_Less_Up' : args['eg'] , 'STUDENTSEX' : args['sex']}
            x = pd.DataFrame([[id_admissions_type ,id_faculty_name,id_school_name,id_entry_gpa,id_year_come,id_sex,id_father_education,id_father_ocupation,id_mother_education,id_mother_ocupation,id_gpa_1_2]] ,\
            columns=['ADMISSIONS_TYPE', 'FACULTY', 'SCHOOL_NAME', 'ENTRY_GPA','YEAR_COME','STUDENT_GENDER','FATHER_EDUCATION','FATHER_OCUPATION','MOTHER_EDUCATION','MOTHER_OCUPATION','GPA_YEAR_1_SEMESTER_2'])
            #x = pd.DataFrame(data=d)
            result_model = model_gpa.predict(x)
            print(x)
            print("predict :", result_model[0])
            result= int(result_model[0])
            if (result == 0):
                str_result = 'สำเร็จการศึกษา'
            elif (result == 1):
                str_result = 'ไม่สำเร็จการศึกษา'
            return {"result": str_result}, 201
        else:
            if check_str_faculty_name == True:
                str_result = 'Faculty name ไม่ถูกต้อง'
                return {"result": str_result}, 201
            elif check_str_entry_gpa == True:
                str_result = 'Entry GPA ไม่ถูกต้อง หรือ Entry GPA ไม่รองรับข้อความ'
                return {"result": str_result}, 201
            elif check_str_sex == True:
                str_result = 'Gender ไม่ถูกต้อง'
                return {"result": str_result}, 201
            elif check_str_year_come == True:
                str_result = 'Year come ไม่รองรับข้อความ กรุณากรองเป็นตัวเลข'
                return {"result": str_result}, 201
            elif check_str_id_father_education == True:
                str_result = 'ระดับการศึกษาของบิดาไม่ถูกต้องหรือไม่รองรับระดับการศึกษาในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_father_ocupation == True:
                str_result = 'อาชีพของบิดาไม่ถูกต้องหรือไม่รองรับอาชีพในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_mother_education == True:
                str_result = 'ระดับการศึกษาของมารดาไม่ถูกต้องหรือไม่รองรับระดับการศึกษาในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_mother_ocupation == True:
                str_result = 'อาชีพของมารดาไม่ถูกต้องหรือไม่รองรับอาชีพในการทำนาย'
                return {"result": str_result}, 201
            elif check_str_id_gpa_1_2 == True:
                str_result = 'GPA ปี1 เทอม 2 ไม่ถูกต้องหรือไม่รองรับข้อความ'
                return {"result": str_result}, 201
            elif check_str_school_name == True:
                str_result = 'ชื่อโรงเรียนไม่ถูกต้อง'
                return {"result": str_result}, 201
            elif check_str_admissions_type == True:
                str_result = 'Admissions type ไม่ถูกต้อง'
                return {"result": str_result}, 201
api.add_resource(student_predict_gpa, "/student_predict_gpa")
app.run(debug=True)