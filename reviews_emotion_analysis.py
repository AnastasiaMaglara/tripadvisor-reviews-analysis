import pandas as pd
import json
import os
import emotion_predictor

def review_emotion_analysis(modelclassification, modelsetting):
    # Pandas presentation options
    pd.options.display.max_colwidth = 100   # show whole tweet's content
    pd.options.display.width = 200          # don't break columns
    # pd.options.display.max_columns = 7      # maximal number of columns


    # Predictor for Ekman's and Plutchik's emotions in multiclass setting.
    model = emotion_predictor.EmotionPredictor(classification=modelclassification, setting=modelsetting)

    reviews = []
    ratings = []
    authors = []
    hotels = []

    path_to_json = 'training dataset/'
    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
      with open(path_to_json + file_name) as json_file:
        data = []
        data = json.load(json_file)
        # read list inside dict
        _list = data['Reviews']
        # read listvalue and load dict
        for v in _list:
            if 'Title' in v:
                fullContent = v['Title']
            else:
                fullContent = ''

            fullContent = fullContent + ' ' + v['Content']
            reviews.append(fullContent)

            ratings.append(v['Ratings'])
            authors.append(v['Author'])
            hotels.append(data['HotelInfo']['HotelID'])

    #predictions = model.predict_classes(reviews, ratings, authors, hotels)
    probabilities = model.predict_probabilities(reviews, ratings, authors, hotels)
    #embeddings = model.embed(reviews, ratings, authors, hotels)

    return probabilities