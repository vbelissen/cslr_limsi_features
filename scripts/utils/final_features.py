import numpy as np
import csv

nimg                = int(sys.argv[1])
vidName             = sys.argv[2]
path2features       = sys.argv[3]
handOP              = bool(sys.argv[4])
faceOP              = bool(sys.argv[5])
body3D              = bool(sys.argv[6])
face3D              = bool(sys.argv[7])
hsKoller            = bool(sys.argv[8])

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

if left_handed:
    handOrder = ['L', 'R']
else:
    handOrder = ['R', 'L']




a=np.zeros(10)
np.save(path2features+'final/'+vidName+'_bodyFace_2D_raw', a)
np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw', a)
np.save(path2features+'final/'+vidName+'_bodyFace_2D_features', a)
np.save(path2features+'final/'+vidName+'_bodyFace_3D_features', a)


if hsKoller:
    final_handShapes = np.zeros((nimg, 122))
    final_handShapes[:, :61] = np.load(path2features+'final/'+vidName+'_HS_probs_'+handOrder[0]+'.npy')
    final_handShapes[:, 61:] = np.load(path2features+'final/'+vidName+'_HS_probs_'+handOrder[1]+'.npy')
    np.save(path2features+'final/'+vidName+'_handShapes', final_handShapes)

if handOP:
    np.save(path2features+'final/'+vidName+'_hand_2D_raw', a)
