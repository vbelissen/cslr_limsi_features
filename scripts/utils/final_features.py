import numpy as np
import csv


path2leftHandedList = '/people/belissen/Python/CSLR_LIMSI/cslr_limsi_features/left_handed/left_handed_dictasign.csv'

video_list = []
left_handed_list = []
with open(path2leftHandedList, 'r', encoding='utf-8') as open_left_handed:
    reader_left_handed = csv.reader(open_left_handed, delimiter=';')
    for i in reader_left_handed:
        print(i)
        video_list.append(i[0])
        left_handed_list.append(i[1])

#gaucher =

print(video_list)
print(left_handed_list)
