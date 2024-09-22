from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

# import os
#
# base_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(base_dir, 'datasets', 'symtoms_df.csv')
#
# df = pd.read_csv(file_path)
#


#load databases=================

sys_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

#load model=====================
svc = pickle.load(open('models/svc.pkl','rb'))


app = Flask(__name__)

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease']== dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    return desc,pre,med,die,wrkout

symptoms_dict = {
    "itching": 0,
    "skin rash": 1,
    "nodal skin eruptions": 2,
    "continuous sneezing": 3,
    "shivering": 4,
    "chills": 5,
    "joint pain": 6,
    "stomach pain": 7,
    "acidity": 8,
    "ulcers on tongue": 9,
    "muscle wasting": 10,
    "vomiting": 11,
    "burning micturition": 12,
    "spotting urination": 13,                 # 13
    "fatigue": 14,                            # 14
    "weight gain": 15,                        # 15
    "anxiety": 16,                            # 16
    "cold hands and feet": 17,                # 17
    "mood swings": 18,                        # 18
    "weight loss": 19,                        # 19
    "restlessness": 20,                       # 20
    "lethargy": 21,                           # 21
    "patches in throat": 22,                  # 22
    "irregular sugar level": 23,              # 23
    "cough": 24,                              # 24
    "high fever": 25,                         # 25
    "sunken eyes": 26,                        # 26
    "breathlessness": 27,                     # 27
    "sweating": 28,                           # 28
    "dehydration": 29,                        # 29
    "indigestion": 30,                        # 30
    "headache": 31,                           # 31
    "yellowish skin": 32,                     # 32
    "dark urine": 33,                         # 33
    "nausea": 34,                             # 34
    "loss of appetite": 35,                   # 35
    "pain behind the eyes": 36,               # 36
    "back pain": 37,                          # 37
    "constipation": 38,                       # 38
    "abdominal pain": 39,                     # 39
    "diarrhoea": 40,                          # 40
    "mild fever": 41,                         # 41
    "yellow urine": 42,                       # 42
    "yellowing of eyes": 43,                  # 43
    "acute liver failure": 44,                # 44
    "fluid overload": 45,                     # 45
    "swelling of stomach": 46,                # 46
    "swelled lymph nodes": 47,                # 47
    "malaise": 48,                            # 48
    "blurred and distorted vision": 49,       # 49
    "phlegm": 50,                             # 50
    "throat irritation": 51,                  # 51
    "redness of eyes": 52,                    # 52
    "sinus pressure": 53,                     # 53
    "runny nose": 54,                         # 54
    "congestion": 55,                         # 55
    "chest pain": 56,                         # 56
    "weakness in limbs": 57,                  # 57
    "fast heart rate": 58,                    # 58
    "pain during bowel movements": 59,        # 59
    "pain in anal region": 60,                # 60
    "bloody stool": 61,                       # 61
    "irritation in anus": 62,                 # 62
    "neck pain": 63,                          # 63
    "dizziness": 64,                          # 64
    "cramps": 65,                             # 65
    "bruising": 66,                           # 66
    "obesity": 67,                            # 67
    "swollen legs": 68,                       # 68
    "swollen blood vessels": 69,              # 69
    "puffy face and eyes": 70,                # 70
    "enlarged thyroid": 71,                   # 71
    "brittle nails": 72,                      # 72
    "swollen extremities": 73,                # 73
    "excessive hunger": 74,                   # 74
    "extra marital contacts": 75,             # 75
    "drying and tingling lips": 76,           # 76
    "slurred speech": 77,                     # 77
    "knee pain": 78,                          # 78
    "hip joint pain": 79,                     # 79
    "muscle weakness": 80,                    # 80
    "stiff neck": 81,                         # 81
    "swelling joints": 82,                    # 82
    "movement stiffness": 83,                 # 83
    "spinning movements": 84,                 # 84
    "loss of balance": 85,                    # 85
    "unsteadiness": 86,                       # 86
    "weakness of one body side": 87,          # 87
    "loss of smell": 88,                      # 88
    "bladder discomfort": 89,                 # 89
    "foul smell of urine": 90,                # 90
    "continuous feel of urine": 91,           # 91
    "passage of gases": 92,                   # 92
    "internal itching": 93,                   # 93
    "toxic look (typhos)": 94,                # 94
    "depression": 95,                         # 95
    "irritability": 96,                       # 96
    "muscle pain": 97,                        # 97
    "altered sensorium": 98,                  # 98
    "red spots over body": 99,                # 99
    "belly pain": 100,                        # 100
    "abnormal menstruation": 101,             # 101
    "dischromic patches": 102,                # 102
    "watering from eyes": 103,                # 103
    "increased appetite": 104,                # 104
    "polyuria": 105,                          # 105
    "family history": 106,                    # 106
    "mucoid sputum": 107,                     # 107
    "rusty sputum": 108,                      # 108
    "lack of concentration": 109,             # 109
    "visual disturbances": 110,               # 110
    "receiving blood transfusion": 111,       # 111
    "receiving unsterile injections": 112,    # 112
    "coma": 113,                              # 113
    "stomach bleeding": 114,                  # 114
    "distention of abdomen": 115,             # 115
    "history of alcohol consumption": 116,    # 116
    "fluid overload": 117,                    # 117
    "blood in sputum": 118,                   # 118
    "prominent veins on calf": 119,           # 119
    "palpitations": 120,                      # 120
    "painful walking": 121,                   # 121
    "pus filled pimples": 122,                # 122
    "blackheads": 123,                        # 123
    "scurring": 124,                          # 124
    "skin peeling": 125,                      # 125
    "silver like dusting": 126,               # 126
    "small dents in nails": 127,              # 127
    "inflammatory nails": 128,                # 128
    "blister": 129,                           # 129
    "red sore around nose": 130,              # 130
    "yellow crust ooze": 131,                 # 131
    "prognosis": 132                          # 132
}
diseases_list = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes ',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension ',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer disease',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'
}

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

#creating routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST' :
      symptoms = request.form.get('symptoms')
      user_symptoms = [s.strip() for s in symptoms.split(',')]
      user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
      predicted_disease = get_predicted_value(user_symptoms)

      desc, pre, med, die, wrkout = helper(predicted_disease)

      my_pre = []
      for i in pre[0]:
          my_pre.append(i)

      return render_template('index.html',predicted_disease=predicted_disease,dis_dec=desc,dis_pre=my_pre,dis_med=med,dis_wrkout=wrkout,dis_diet=die)



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

if __name__=="__main__":
    app.run(debug=True)

