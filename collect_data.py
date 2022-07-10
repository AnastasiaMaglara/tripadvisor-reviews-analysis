import json
import os
import shutil

def create_empty_folder(dir):
    # if exists the directory then delete it
    if os.path.exists(dir):
        shutil.rmtree(dir)
    # create directory
    os.mkdir(dir)

def collect_data(path_to_json):

    create_empty_folder('5-core')
    authors = {"Name": "Count"}
    hotels = {"File_name": "Data"}

    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
      with open(path_to_json + file_name) as json_file:
        data= []
        data=json.load(json_file)
        _list = data['Reviews']
        #check if hotel has more than 5 reviews
        if len(_list) >= 5:
            hotels[file_name] = data
        else:
            continue

        for v in _list:
            if v['Author'] in authors.keys():
                authors[v['Author']] = authors.get(v['Author']) + 1
            else:
                authors[v['Author']] = 1

    authors.pop('Name')
    hotels.pop('File_name')

    for hotel in hotels:
        json_data = hotels.get(hotel)
        reviews = []
        for rating in json_data['Reviews']:
            if authors.get(rating['Author']) >= 5:
                reviews.append(rating)
        json_data['Reviews'] = reviews
        with open('5-core/'+hotel, 'w') as outfile:
            json.dump(json_data, outfile)
