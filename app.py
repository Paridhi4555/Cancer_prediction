from flask import Flask,render_template,request,url_for
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open('model_cancer1.pkl','rb'))
app = Flask(__name__, template_folder= "Templates")

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict',methods=["POST"])
def predict():
    radius_mean=float(request.form.get( 'radius_mean' ))
    texture_mean=float(request.form.get( 'texture_mean' ))
    perimeter_mean=float(request.form.get( 'perimeter_mean' ))
    area_mean=float(request.form.get( 'area_mean' ))
    smoothness_mean=float(request.form.get( 'smoothness_mean' ))
    symmetry_mean=float(request.form.get( 'symmetry_mean' ))
    fractal_dimension_mean=float(request.form.get( 'fractal_dimension_mean' ))
    radius_se=float(request.form.get( 'radius_se' ))
    texture_se=float(request.form.get( 'texture_se' ))
    perimeter_se=float(request.form.get( 'perimeter_se' ))
    area_se=float(request.form.get( 'area_se' ))
    smoothness_se=float(request.form.get( 'smoothness_se' ))
    compactness_se=float(request.form.get( 'compactness_se' ))
    concavity_se=float(request.form.get( 'concavity_se' ))
    concave_points_se=float(request.form.get( 'concave points_se' ))
    symmetry_se=float(request.form.get( 'symmetry_se' ))
    fractal_dimension_se=float(request.form.get( 'fractal_dimension_se' ))
    radius_worst=float(request.form.get( 'radius_worst' ))
    texture_worst=float(request.form.get( 'texture_worst' ))
    perimeter_worst=float(request.form.get( 'perimeter_worst' ))
    area_worst=float(request.form.get( 'area_worst' ))
    smoothness_worst=float(request.form.get( 'smoothness_worst' ))
    compactness_worst=float(request.form.get( 'compactness_worst' ))
    concavity_worst=float(request.form.get( 'concavity_worst' ))
    concave_points_worst=float(request.form.get( 'concave points_worst' ))
    symmetry_worst=float(request.form.get( 'symmetry_worst' ))
    fractal_dimension_worst=float(request.form.get( 'fractal_dimension_worst' ))
    compactness_mean=float(request.form.get( 'compactness_mean' ))
    concavity_mean=float(request.form.get( 'concavity_mean' ))
    concave_points_mean=float(request.form.get( 'concave points_mean' ))

    
    feature_final=np.array([[radius_mean,texture_mean, perimeter_mean, area_mean, smoothness_mean, symmetry_mean, compactness_mean, concavity_mean, concave_points_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])
    result=model.predict(feature_final)
    
    if result==1:
        return "<h1 style='color:red'>Breast cancer report Malignant</h1>"
    else:
        return "<h1 style='color:green'>Breast cancer report benign</h1>"

# if(__name__=='__main__'):
#     app.run(debug=True)

    