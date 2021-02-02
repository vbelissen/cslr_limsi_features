import numpy as np
import csv


vidName             = 'DictaSign_lsf_S2_T1_A11_front'
path2leftHandedList = '/people/belissen/Python/CSLR_LIMSI/cslr_limsi_features/left_handed/left_handed_dictasign.csv'

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None

video_list = []
left_handed_list = []
with open(path2leftHandedList, 'r', encoding='utf-8-sig') as open_left_handed:
    reader_left_handed = csv.reader(open_left_handed, delimiter=';')
    for i in reader_left_handed:
        video_list.append(i[0])
        left_handed_list.append(int(i[1]))


left_handed_found = find_element_in_list(vidName, video_list)
if left_handed_found is None:
    left_handed = False
else:
    left_handed = bool(left_handed_list[left_handed_found])


print(left_handed)
