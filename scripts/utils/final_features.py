import numpy as np
import csv



path2leftHandedList = '/Users/belissen/Code/cslr_limsi_features/left_handed/left_handed_dictasign.csv'

video_list = []
left_handed_list = []
with open(path2leftHandedList, 'r') as open_left_handed:
    reader_left_handed = csv.reader(open_left_handed)#, delimiter='\t')
    for i in reader_left_handed:
        video_list.append(i[0])
        left_handed_list.append(i[1])

#gaucher =

print(video_list)
print(left_handed_list)
