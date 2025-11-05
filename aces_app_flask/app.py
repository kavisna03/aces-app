import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import shutil
import re
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pickle
from language_complexity.language_model import process_and_transcribe_video, build_model, predict_grade_level
from presentation_complexity.presentation_feature_engineering import get_video_features

# create a flask app
aces = Flask(__name__)


# calls the home() function when requests are sent to the home/main page url or endpoint
@aces.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', language_prediction_text="", presentation_prediction_text="")


# predict presentation complexity and/or language complexity using the checker form inputs
@aces.route('/checker_form', methods=['POST'])
def predict_complexity():
    language_prediction_text = presentation_prediction_text = ""
    video_folder = 'uploaded_videos'
    video_path = ""

    try:
        # get the video file and save it temporarily for processing
        if not os.path.exists(video_folder):
            os.mkdir(video_folder)

        video_file = request.files['video-file']

        video_name = os.path.splitext(video_file.filename)[0]
        # remove special characters from video name to avoid any issues with saving the file
        video_name = re.sub(r'[^A-Za-z0-9]', '', video_name)
        # make the video name shorter to avoid any issues with writing and reading the file
        video_name_len = len(video_name) // 4
        video_name = video_name[:video_name_len]
        video_ext = os.path.splitext(video_file.filename)[1]

        video_path = os.path.join(video_folder, secure_filename(video_name) + video_ext)
        video_file.save(video_path)

        language_criterion = 'language-complexity' in request.form
        presentation_criterion = 'presentation-complexity' in request.form

        # check if the language complexity checkbox is selected
        if language_criterion:
            print("Language complexity processing has started.")

            # convert video audio to text
            transcription = process_and_transcribe_video(video_path)

            # build model
            language_model = build_model(transcription)
            # load the weights of language model
            weights_path = 'language_complexity/language_model_weights.h5'

            if os.path.exists(weights_path):
                language_model.load_weights(weights_path, by_name=True)
                print(f"Loaded weights from: {weights_path}")
            else:
                print("Weights file not found.")

            language_complexity = predict_grade_level(transcription, language_model)

            language_prediction_text = f"Language Complexity: {language_complexity}"

            print("Language complexity processing has completed.")

        # check if the presentation complexity checkbox is selected
        if presentation_criterion:
            print("Presentation complexity processing has started.")

            # load model
            presentation_model = pickle.load(open('presentation_complexity/presentation_rf_model.pkl', 'rb'))
            # load min max scaler needed for scaling the features of presentation complexity model
            presentation_scaler = pickle.load(open('presentation_complexity/presentation_scaler.pkl', 'rb'))

            # feature engineering
            video_features = [np.array(get_video_features(video_path))]
            # feature normalization
            video_features = presentation_scaler.transform(video_features)

            presentation_complexity = presentation_model.predict(video_features)

            if presentation_complexity == 0:
                presentation_prediction_text = "Presentation Complexity: Simple"
            elif presentation_complexity == 1:
                presentation_prediction_text = "Presentation Complexity: Hard"

            print("Presentation complexity processing has completed.")

        return render_template('index.html', language_prediction_text=language_prediction_text, presentation_prediction_text=presentation_prediction_text)
    except Exception as e:
        print(f"Error predicting video complexity: {e}")
        return None
    finally:
        if video_path:
            os.remove(video_path)

        shutil.rmtree(video_folder)


if __name__ == '__main__':
    aces.run(debug=False)