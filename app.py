from flask import Flask , render_template, request, jsonify
from chatbot import  _get_response,_predict_class
app=Flask(__name__)

@app.get("/")
def index_get ():
    return render_template("base.html")

@app.post("/predict")
def predict ():
    text=request.get_json().get("message")
    ints = _predict_class(text)

    response =_get_response(ints)
    message ={"answer": response}
    return jsonify(message)

if __name__=="__main__":
    app.run(debug=True)

