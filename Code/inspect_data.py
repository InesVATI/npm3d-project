import numpy as np
import pandas as pd
from ply import write_ply, read_ply
import xml.etree.ElementTree as ET
import json

def get_dataset(filename='Cassette_idclass/Cassette_GT.ply'):
    cloud_ply = read_ply(filename)
    cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

    # label : to change ??
    label = cloud_ply['class']

    return cloud, label

if __name__ == '__main__':
    cloud_path = 'Cassette_idclass/Cassette_GT.ply'
    cloud_ply = read_ply(cloud_path)
    # print('id', cloud_ply['id'].shape)
    # # v = np.unique(cloud_ply['id'])
    # # print('nb id values', v.shape)
    # # print('id values', v)
    # print('class', cloud_ply['class'].shape)
    class_id = np.unique(cloud_ply['class'])
    print(class_id)
    # print('class', class_id.shape)
    # print('class values', class_id)
    cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    # print('Cloud shape', cloud.shape)
    

    tree_xml = ET.parse('Cassette_idclass/classes_tree.xml')
    root_xml = tree_xml.getroot()

    # Create a dictionary mapping class ids to class names
    semantic_class = {}
    for class_elem in root_xml.iter('class'):
        if int(class_elem.attrib['id']) in class_id:
            semantic_class[int(class_elem.attrib['id'])] = class_elem.attrib['en']

    print(semantic_class)
    print(len(semantic_class))

    # new_semantic_class = {0: 'unclassified',
    #                       1: ['facade', 301000000],
    #                       2: ['ground', 202020000, 202030000, 202040000],
    #                       3: ['motorcycles', ],
    #                       4: ['pedestrian', 303020000, 303020200, 303020300, 303020600],
    #                       5: ['vegetation', 304020000, 304040000],
    #                       } # dict to simplify classification ? in accordance with the result showed in the article
    
    with open('Cassette_idclass/semantic_map.json', 'w') as f:
        json.dump(semantic_class, f)
    


