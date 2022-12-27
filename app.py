from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open("model/model.pkl", "rb"))
loaded_vectorizer = pickle.load(open('model/vec.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods = ["GET", "POST"])
def classify():
    if request.method == "POST":

        prediction = loaded_model.predict(loaded_vectorizer.transform([x for x in request.form.values()]))[0]
        output = "POSITIVE" if prediction == 1 else "NEGATIVE"

        return render_template("index.html", prediction_text = f"The Entered Movie Review is {output}")

if __name__ == "__main__":
    app.debug = True
    app.run()