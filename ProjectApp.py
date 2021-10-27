# from flask import Flask,request,jsonify
from flask import Flask,request,jsonify
from ProjectClassifier import getPrediction 

app=Flask(__name__)
@app.route('/predictAlphabet',methods=['POST'])

def predictData():
    image=request.files.get("alphabet")
    prediction=getPrediction(image)
    return jsonify(
        {
            "prediction": prediction,
        }
    ),200

if __name__=='__main__':
    app.run(debug=True)