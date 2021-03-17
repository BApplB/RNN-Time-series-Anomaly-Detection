import requests
import os
import json
from pathlib import Path
import pickle
from shutil import unpack_archive

with open('dataset_definitions.json','r') as json_file:
    data_definitions = json.load(json_file)
urls = data_definitions['urls']


def label_anomaly_txt(filepath,anomaly_chunks,offset=0,labelvalue=1.0):
    with open(str(filepath)) as f:
        labeled_data = []
        for i, line in enumerate(f):
            token_added = False
            tokens = [float(token) for token in line.split()]
            if offset > 0:
                for i in range(offset):
                    tokens.pop(i)
            for chunk in anomaly_chunks:
                if chunk[0] < i < chunk[1]:
                    tokens.append(labelvalue) 
                    token_added=True
            if not token_added:
                tokens.append(0.0)
            labeled_data.append(tokens)
    return labeled_data


def pickle_whole_dataset(labeled_data,filepath,rawdir):
    labeled_whole_dir = raw_dir.parent.joinpath('labeled', 'whole')
    labeled_whole_dir.mkdir(parents=True, exist_ok=True)
    with open(str(labeled_whole_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
        pickle.dump(labeled_data, pkl)
        
    
def pickle_data_subset(labeled_data, bounds, filepath, datadir):
     with open(str(datadir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
        if bounds[1] is None and bounds[0] is None:
              pickle.dump(labeled_data, pkl)
        elif bounds[1] is None:
              pickle.dump(labeled_data[bounds[0]:], pkl)
        elif bounds[0] is None:
              pickle.dump(labeled_data[:bounds[1]], pkl)
        else:
              pickle.dump(labeled_data[bounds[0]:bounds[1]], pkl)       

                
def pickle_test_train_dataset(labeled_data, rawdir, training_set_bounds, test_set_bounds):
    labeled_train_dir = raw_dir.parent.joinpath('labeled','train')
    labeled_test_dir = raw_dir.parent.joinpath('labeled','test')
    pickle_data_subset(labeled_data, training_set_bounds, filepath, labeled_train_dir)
    pickle_data_subset(labeled_data, test_set_bounds, filepath, labeled_test_dir)


for dataname in urls:
    raw_dir = Path('dataset', dataname, 'raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in urls[dataname]:
        filename = raw_dir.joinpath(Path(url).name)
        print('Downloading', url)
        resp =requests.get(url)
        filename.write_bytes(resp.content)
        if filename.suffix=='':
            filename.rename(filename.with_suffix('.txt'))
        print('Saving to', filename.with_suffix('.txt'))
        if filename.suffix=='.zip':
            print('Extracting to', filename)
            unpack_archive(str(filename), extract_dir=str(raw_dir))

    for filepath in raw_dir.glob('*.txt'):
        try:
            filedata = data_definitions['files'][filepath.name]
        except KeyError:
            print("\nWARN: %s not in dataset_definitions.json, skipping!\n"%(filepath.name))
            continue
        if raw_dir.parent.name== 'ecg':
            # Remove time-step channel
            labeled_data = label_anomaly_txt(filepath,filedata['anomaly_chunks'],offset=1)
        else:
            labeled_data = label_anomaly_txt(filepath,filedata['anomaly_chunks'],offset=0)

        # Fill in the point where there is no signal value.
        if filepath.name == 'ann_gun_CentroidA.txt':
            for i, datapoint in enumerate(labeled_data):
                for j,channel in enumerate(datapoint[:-1]):
                    if channel == 0.0:
                        labeled_data[i][j] = 0.5 * labeled_data[i - 1][j] + 0.5 * labeled_data[i + 1][j]

        # Save the labeled dataset as .pkl extension
        pickle_whole_dataset(labeled_data, filepath, raw_dir)
        # Divide the labeled dataset into trainset and testset, then save them
        pickle_test_train_dataset(labeled_data, raw_dir, filedata['training_data_bounds'], filedata['test_data_bounds'])
            
            
nyc_taxi_raw_path = Path('dataset/nyc_taxi/raw/nyc_taxi.csv')
filedata = data_definitions['files']['nyc_taxi.csv']

labeled_data = label_anomaly_txt(filepath,filedata['anomaly_chunks'])
pickle_test_train_dataset(labeled_data, nyc_taxi_raw_path, filedata['training_data_bounds'], filedata['test_data_bounds'])