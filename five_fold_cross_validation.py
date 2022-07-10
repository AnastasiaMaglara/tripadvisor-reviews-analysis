import json
import os
import shutil
from collect_data import create_empty_folder
import collaborative_filtering
from reviews_emotion_analysis import *

def get_dimensions(classification):
    # define dimensions used in the overall process
    if classification == 'ekman':
        return ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    elif classification == 'plutchik':
        return ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust", "Anticipation"]
    elif classification == 'poms':
        return ["Anger", "Depression", "Fatigue", "Vigour", "Tension", "Confusion"]

def split_data():
    #create or replace directory
    create_empty_folder('test dataset')
    create_empty_folder('training dataset')

    path_to_json = '5-core/'

    list = os.listdir(path_to_json)
    number_files = len(list)
    number_files = 5 * (number_files // 5)

    for i, file_name in enumerate([file for file in os.listdir(path_to_json) if file.endswith('.json')]):
        if i < number_files:
            if i < (number_files / 5):
                shutil.copy(path_to_json + file_name, 'test dataset')
            else:
                shutil.copy(path_to_json + file_name, 'training dataset')
        else:
            return

def create_mult_dict(classification, data):
    ratings = {}
    DIMENSIONS = get_dimensions(classification)
    dict = json.loads(data)
    for v in dict:
       author_name = v['Author']
       hotel_id = v['Hotel']
       overall = v['Rating']['Overall']

       if author_name in ratings:
           ratings[author_name][hotel_id] = {}
           ratings[author_name][hotel_id]['Overall'] = float(overall)
           for dimension in DIMENSIONS:
               ratings[author_name][hotel_id][dimension] = v[dimension]
       else:

           ratings[author_name] = {}
           ratings[author_name][hotel_id] = {}
           ratings[author_name][hotel_id]['Overall'] = float(overall)
           for dimension in DIMENSIONS:
               ratings[author_name][hotel_id][dimension] = v[dimension]

    return ratings

def create_dict(path_to_json):
    ratings = {}

    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
        with open(path_to_json + file_name) as json_file:
            data = json.load(json_file)
            _list = data['Reviews']
            for v in _list:
                author_name = v['Author']
                hotel_id = data['HotelInfo']['HotelID']
                rating = v['Ratings']['Overall']

                if author_name in ratings:
                    ratings[author_name][hotel_id] = float(rating)
                else:
                    ratings[author_name] = {}
                    ratings[author_name][hotel_id] = float(rating)
    return ratings

def predictive_algorithm(classification, metric, setting, use_emotion_analysis):

    split_data()
    model = collaborative_filtering.CollaborativeFiltering(classification=classification, use_emotion_analysis=use_emotion_analysis)
    training_path = 'training dataset/'
    test_path = 'test dataset/'
    create_empty_folder('results')

    if not use_emotion_analysis:
        training_ratings = create_dict(training_path)
    else:
        training_dataset = review_emotion_analysis(classification, setting)
        training_ratings = create_mult_dict(classification, training_dataset)

    sim = {}
    if metric == 'Pearson Correlation':
        sim = model.pearsonCorrelation(training_ratings)
    elif metric == 'Cosine Similarity':
        sim = model.cosineSimilarity(training_ratings)
    else:
        raise ValueError('Unknown similarity metric: {}'.format(
            metric))

    #with open('results/' + 'pearsonDicts.json', "w") as outfile:
    #    json.dump(sim, outfile, indent=4)

    test_ratings = create_dict(test_path)
    predictions = {}
    test_values = []

    if metric == 'Pearson Correlation':
        for userToPredict in test_ratings:
            if userToPredict in sim:
                predictions[userToPredict] = {}
                test_values.append(test_ratings[userToPredict])
                for itemToPredict in test_ratings[userToPredict]:
                    predictions[userToPredict][itemToPredict] = model.predictRatingPearson(test_ratings, sim, userToPredict, itemToPredict)
    else:
        for userToPredict in test_ratings:
            if userToPredict in sim:
                predictions[userToPredict] = {}
                test_values.append(test_ratings[userToPredict])
                for itemToPredict in test_ratings[userToPredict]:
                    predictions[userToPredict][itemToPredict] = model.predictRatingCosine(test_ratings, sim, userToPredict, itemToPredict)

    #with open('results/' + 'predictRating.json', "w") as outfile:
    #    json.dump(predictions, outfile, indent=4)

    print('Result of MAE: ', model.mae(test_values, list(predictions.values())))
    print('Result of RMSE: ', model.rmse(test_values, list(predictions.values())))
