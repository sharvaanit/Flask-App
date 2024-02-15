from flask import Flask, request, jsonify, render_template
from transformers import BartForConditionalGeneration, BartTokenizer
import pickle
app = Flask(__name__)
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = pickle.load(open("bart.pkl","rb"))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from the form
        text = request.form.get('tweet')
        input_ids = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
        summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
        prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return render_template('index.html', prediction_text="Summarized text is {}".format(prediction))
    except Exception as e:
        return render_template('index.html', prediction_text="Error predicting sentiment: {}".format(str(e)))

if __name__ == '__main__':
    app.run(port=3000, debug=True)
