import pandas as pd
from flask import Flask, request,  render_template,url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
	
	GRE_Score = request.form.get('GRE Score')
	TOEFL_Score = request.form.get('TOEFL Score')
	University_Rating = request.form.get('University Rating')
	SOP = request.form.get('SOP')
	LOR = request.form.get('LOR')
	CGPA = request.form.get('CGPA')
	Research = request.form.get('Research')
	
	final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])
	
	predict = model.predict(final_features)
	
	output = predict[0]/100
	
	return render_template('predict.html', prediction_text='Admission chances are {}'.format(output))
	
if __name__ == "__main__":
	app.run()