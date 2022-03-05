# Importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict" , methods=["POST"])
def predict():
    # Loading the model
    model = tf.keras.models.load_model('my_model_pooja.h5')

    # Taking the input text i.e. news using Post method
    if request.method == 'POST':
        text = request.form.get('text')

        # Predicting the news
        data = [text]
        # Tokenizing the data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)
        # word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(data)
        # Padding the Tokenize data
        padded_data = pad_sequences(sequences, maxlen=1000)
        # Predicting the news is fake or real
        predicted_val = model.predict(padded_data)

        if(predicted_val >= 0.5):
            prediction = "News is Fake don't Trust it !"
        else:
            prediction = "News is Real !"
        # rendering results page with prediction and given news by user
    return render_template('results.html', prediction = prediction,news = text)

    # if request to result page is not using post then back to home page
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)