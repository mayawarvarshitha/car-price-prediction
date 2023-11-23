from flask import Flask, render_template,jsonify,request
import pandas as pd
import pickle
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method=="POST":
        make = request.form.get("make")
        model = request.form.get("model")
        year = request.form.get("year")
        fuel = request.form.get("fuel")
        hp = request.form.get("hp")
        cylinders = request.form.get("cylinders")
        transmission = request.form.get("transmission")
        wheels = request.form.get("wheels")
        doors = request.form.get("doors")
        size = request.form.get("size")
        style = request.form.get("style")
        highway = request.form.get("highway")
        city = request.form.get("city")
        popularity = request.form.get("popularity")
        data = pd.read_json("new.json")
        
        make_encode = data['Make_encode'][data['Make']==make].values[0]
        model_encode = data['Model_encode'][data['Model']==model].values[0]
        eft_encode = data['Engine Fuel Type_encode'][data['Engine Fuel Type']==fuel].values[0]
        tt_encode = data['Transmission Type_encode'][data['Transmission Type']==transmission].values[0]
        dw_encode = data['Driven_Wheels_encode'][data['Driven_Wheels']==wheels].values[0]
        vsz_encode = data['Vehicle Size_encode'][data['Vehicle Size']==size].values[0]
        vs_encode = data['Vehicle Style_encode'][data['Vehicle Style']==style].values[0]        
        print(make_encode,model_encode,eft_encode,tt_encode,dw_encode,vsz_encode,vs_encode)
        with open('model.pkl','rb') as model:
            mlmodel=pickle.load(model)
            predict = mlmodel.predict([[int(year),float(hp),float(cylinders),float(doors),float(highway),float(city),float(popularity),make_encode,model_encode,eft_encode,tt_encode,dw_encode,vsz_encode,vs_encode]])
        print(predict)
        print(make,model,year,fuel,hp,cylinders,transmission,wheels,doors,size,style,highway,city,popularity)
        return jsonify({"Predicted result":f"result :{predict}"})
    else:
        return render_template("predict.html")




if __name__=='__main__':
    app.run()