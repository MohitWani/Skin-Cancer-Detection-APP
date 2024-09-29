from flask import Flask, render_template, request, url_for
from Diabetic_Retinopathy import detection, model

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message= 'No files selected')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message = 'No image selected')
        
        if file:

            models = model.build_model()
            models.load_weights('best_model.keras')

            img_array = detection.preprocess_image(file)
            prediction = detection.detection(models, img_array)
            labels = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'),
                     5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}
            return render_template("index.html", message=labels[prediction])
    return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)