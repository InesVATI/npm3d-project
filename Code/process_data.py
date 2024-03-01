import numpy as np
import pandas as pd
from ply import write_ply, read_ply
import xml.etree.ElementTree as ET
import json
from pathlib import Path

root_folder = Path(__file__).parent.parent
data_folder = root_folder / '__data'
cloud_path = data_folder / 'Cassette_GT.ply'

label_names = {0: 'Unclassified',
               1: 'Ground',
               2: 'Building', # Facade
               3: 'Traffic Signs',
                4: 'Pedestrians',
                5: 'Cars',
                6: 'Vegetation',
                7: 'Motorcycles',}

def get_dataset(filename=cloud_path):
    cloud_ply = read_ply(filename)
    cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

    # label : to change ??
    label = cloud_ply['class']

    return cloud, label

if __name__ == '__main__':
    
    # arr = np.zeros(20)
    # arr_v = np.arange(20)
    # dict = {3: [1, 3, 5],
    #         1: [2, 4, 6],}
    # for key in dict.keys():
    #     arr[np.isin(arr_v, dict[key])] = key
    # print(arr)

    cloud_ply = read_ply(cloud_path) 
    cloud_class_id = cloud_ply['class']
    print('class', cloud_class_id.shape)

    class_id = np.unique(cloud_class_id)
    print('class values', class_id)
    new_labels = np.zeros(cloud_class_id.shape[0])

    tree_xml = ET.parse(data_folder / 'classes_tree.xml')
    root_xml = tree_xml.getroot()

    # Create a dictionary mapping class ids to class names
    map_path = data_folder/'semantic_map.json'
    if map_path.exists():
        with open(map_path, 'r') as f:
            map_semantic_id = json.load(f)
    else : 
        map_semantic_id = {1: [],
                        2: [],
                        3: [302020600, 302020500],
                        4: [],
                        5: [],
                        6: [],
                        7: []}
        # parse the xml file to get the class ids
        for class_elem in root_xml.iter('class'):
            if int(class_elem.attrib['id']) in class_id:
                # check class 1 
                if ("2020" in class_elem.attrib['id'][:4]) and ("20206" not in class_elem.attrib['id'][:5]):
                    map_semantic_id[1].append(int(class_elem.attrib['id']))
                # check class 2
                if ("2030" in class_elem.attrib['id'][:4]):
                    map_semantic_id[2].append(int(class_elem.attrib['id']))
                # # check class 3
                # if (class_elem.attrib['en'] in ["traffic sign", "traffic light"]):
                #     map_semantic_id[3].append(int(class_elem.attrib['id']))
                # check class 4
                if ("30302" in class_elem.attrib['id'][:5]):
                    map_semantic_id[4].append(int(class_elem.attrib['id']))
                # check class 5
                if ("3030402" in class_elem.attrib['id'][:7]):
                    map_semantic_id[5].append(int(class_elem.attrib['id']))
                # check class 6
                if ("20206" in class_elem.attrib['id'][:5]) or ("3040" in class_elem.attrib['id'][:4]):
                    map_semantic_id[6].append(int(class_elem.attrib['id']))
                # check class 7
                if (np.array(["3030303", "3030304", "3030305"]) == class_elem.attrib['id'][:7]).any():
                    map_semantic_id[7].append(int(class_elem.attrib['id']))

        print(map_semantic_id)
        with open(data_folder/'semantic_map.json', 'w') as f:
            json.dump(map_semantic_id, f)

    cloud_for_classifiction_path = data_folder / 'Cassette_Cloud_forClassification.ply'
    if cloud_for_classifiction_path.exists():
        print('Simplified class cloud file already exists.')
        pass
    else :
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        print('Cloud shape', cloud.shape)
        for key in map_semantic_id.keys():
            mask = np.isin(cloud_class_id, map_semantic_id[key])
            new_labels[mask] = key
            print(f'Label {key} has {np.sum(mask)} points')
        # save a new ply file with simplified label classes 
        write_ply(str(cloud_for_classifiction_path), (cloud, new_labels), ['x', 'y', 'z', 'class'])
    
    
    


