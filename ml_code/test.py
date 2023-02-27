import os
import pandas as pd
import re
from torch.utils.data import DataLoader, random_split
from dataloader_MDPose import dataset_LSTM, CustomDataset, CustomDataset_class, CustomDataset_window, CustomDataset_window_changequat, dataset_LSTM_changequat
from datetime import datetime
import torch
from kinect_quat_functions_nocv import abs2relquat, rel2absquat

# from velocity_functions import loadDataXYZ
# from kinect_quat_functions import vel2canvas


# files = ["./state_dict_model_outputlog_velocitymodel_shifted.txt"]
# for file in files:
#     quats = []
#     quat = []
#     with open(file) as vel_file:
#         for line in vel_file:
#             line = line.strip()
#             if line[0]=='t':
#                 if (quat != []):
#                     quats.append(quat)
#                     # print(quats)
#                     # time.sleep(2)
#                 quat = []
#                 line = line[9:-1] # remove tensor([ and ending comma
#             elif line[-2]==']':
#                 line = line[:-3] # remove ])
#             elif line[0]=='d':
#                 continue
#             else:
#                 line = line[:-1]
#             line = line.split(",")
#             digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#             for l in line:
#                 try:
#                     quat.append(float(l.strip()))
#                 except:
#                     # print(l)
#                     if l.strip()[0]=='d':
#                         continue
#                     if len(l)==0:
#                         continue
#                     else:
#                         while l[0] not in digits:
#                             l=l[1:]
#                             if len(l)==0:
#                                 break
#                         if len(l)==0:
#                             break
#                         while l[-1] not in digits:
#                             l=l[:-1]
#                             if len(l)==0:
#                                 break
#                         # l = l.strip()
#                         # l = l[:-2]
#                         quat.append(float(l.strip()))

#     print("done extracting quats")
#     # print(quat)

#     vel_file.close()

    
#     print_counter = 0
#     # print(len(quats))
#     quat_array = []
#     for quat_100 in quats:
#         # print(str(print_counter))

#         # print_counter +=1
        
#         # transformed_quat = []
#         # canvas = np.zeros((1800, 1920, 3), np.uint8)
#         # if print_date==1:
#         #   font = cv2.FONT_HERSHEY_PLAIN
#         #   cv2.putText(canvas, row[0], (20, 40), font, 2, (255, 255, 255), 2)
    
#         counter = 0
#         body_quat = []
#         # print(quat_100)
#         for single in quat_100:
#             # print(single)
#             if counter%3==0:
#                 if counter>0:
#                     body_quat.append(single_quat)
#                 single_quat = []
#                 single_quat.append(single)

#             else:
#                 single_quat.append(single)
#             counter +=1
#             # print(counter)
#         body_quat.append(single_quat)
#         quat_array.append(body_quat)
#     counter =0
#     new_quat_array=[]
#     for single in quat_array[0]:
#         if counter%25==0:
#             if counter>0:
#                 new_quat_array.append(single_body)
#             single_body = []
#             single_body.append(single)
#         else:
#             single_body.append(single)
#         counter +=1
#     print(new_quat_array)

#     original_coor = 

                
start_quat = [ #IA DW1 511

(0.0, 0.0, 0.0, 0.0),
(-0.04707345366477966, 0.9928267598152161, 0.0916907787322998, -0.06059674173593521),
(-0.050773657858371735, 0.981590211391449, 0.1726224720478058, -0.06406333297491074),
(-0.05271880328655243, 0.9828792810440063, 0.15587393939495087, -0.0828995332121849),
(-0.055708423256874084, 0.9811021089553833, 0.15483064949512482, -0.10179711878299713),
(0.6097254753112793, 0.7680047750473022, 0.17346587777137756, -0.09117614477872849),
(0.7949207425117493, -0.5663865208625793, -0.2174895852804184, 0.0023715635761618614),
(0.6274937391281128, 0.7045024633407593, -0.12439727038145065, -0.30733251571655273),
(0.7008910775184631, -0.6823100447654724, -0.025989053770899773, -0.20622630417346954),
(0.8269293904304504, -0.024939406663179398, 0.5411424040794373, -0.1507667601108551),
(0.8078917264938354, 0.04719074070453644, -0.29903674125671387, -0.5056294202804565),
(-0.571729838848114, 0.08838000148534775, 0.4037308394908905, 0.708742082118988),
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(0.8546380400657654, 0.021953344345092773, -0.467999666929245, -0.22380392253398895),
(-0.3276265561580658, -0.5377433896064758, 0.776140034198761, 0.033159371465444565),
(0.3741157352924347, 0.7144322395324707, -0.588370680809021, 0.05868496000766754),
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(-0.48251697421073914, 0.4210188090801239, -0.5229143500328064, 0.5625665783882141),
(0.7528988122940063, 0.11099760979413986, 0.6421276330947876, 0.09216790646314621),
(0.0, 0.0, 0.0, 0.0),
(-0.4309995472431183, -0.43326467275619507, 0.5260026454925537, 0.5914748311042786),
(-0.6685120463371277, 0.024979878216981888, 0.7431969046592712, 0.01122257299721241),
(0.0, 0.0, 0.0, 0.0)
]

quat523=[
(0.0, 0.0, 0.0, 0.0),
(-0.043671127408742905, 0.9934070706367493, 0.09298054873943329, -0.05088963732123375),
(-0.05026879906654358, 0.988684892654419, 0.13140803575515747, -0.05203019455075264),
(-0.05385955050587654, 0.9811245799064636, 0.16993381083011627, -0.07494192570447922),
(-0.05745959281921387, 0.9793098568916321, 0.16875053942203522, -0.0957803726196289),
(0.6167751550674438, 0.7710679173469543, 0.14721272885799408, -0.05806050822138786),
(0.8042399883270264, -0.5659787058830261, -0.17996083199977875, 0.021912377327680588),
(0.6206248998641968, 0.7096885442733765, -0.11097836494445801, -0.3144051730632782),
(0.7095484137535095, -0.6687377691268921, -0.02832886390388012, -0.22029128670692444),
(0.8327422738075256, -0.026022857055068016, 0.534562349319458, -0.14179600775241852),
(0.860197901725769, -0.13729386031627655, -0.01597135327756405, -0.4908715486526489),
(0.6603731513023376, -0.31955257058143616, -0.15957535803318024, -0.6605521440505981),
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(0.8582645654678345, 0.01002351101487875, -0.46489736437797546, -0.21714498102664948),
(-0.23671838641166687, -0.5416824817657471, 0.8056447505950928, -0.038483552634716034),
(0.31956279277801514, 0.6997244954109192, -0.6184013485908508, 0.1607637107372284),
(0.0, 0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 0.0),
(-0.4764614999294281, 0.41930121183395386, -0.5304596424102783, 0.5619461536407471),
(0.5574114918708801, 0.11609425395727158, 0.8202590346336365, 0.05467696115374565),
(0.0, 0.0, 0.0, 0.0),
(-0.43570396304130554, -0.42929166555404663, 0.5331853032112122, 0.5844519734382629),
(-0.6717790961265564, 0.018833067268133163, 0.7402865886688232, 0.01827508769929409),
(0.0, 0.0, 0.0, 0.0)
]

train_dataloader = DataLoader(dataset_LSTM_changequat(), batch_size= 100, shuffle=False)
f = open('test_estimate.txt', 'w+')

total_quats = []
for i, data in enumerate(train_dataloader, 0):
    nil, train_labels = data
    # train_labels: 100samples*5bodyquats*100(25*4)
    for x in range(len(train_labels)):
        body_quat = train_labels[x][2] # take the middle quat
        # print(body_quat)
        # error

        # counter = 0
        body_quat = torch.Tensor([(1)*((b/10)-1) for b in body_quat])
        body_quat = torch.reshape(body_quat, (25,4))
        # print(body_quat)
        # error
        # 
        total_quats.append(body_quat)
        # for quat in b
# print(len(total_quats))
# print(len(total_quats[0]))
abs_quats = rel2absquat(quat523, total_quats)
# print(abs_quats[0])
f.write(str(abs_quats))

    # print(len(train_labels))
    # print(len(train_labels[0]))
    # print(len(train_labels[0][0]))
    # train_inputs = train_inputs.cuda()
    # train_inputs = train_inputs[None, :]
    # train_labels = train_labels.cuda()
    # train_labels = train_labels[None, :]



    # print(quat_array)

    # abs_quats = rel2absquat(start_quat, quat_array)
    # quats = []
    # quats.append(start_quat)
    # for a in abs_quats:
    #     quats.append(a)
    
    # for q in quats:
    #     vector_array = quat2vector(q, cameraspace = False)
    #     canvas = np.zeros((1800, 1920, 3), np.uint8)
    #     canvas = vector2screen (vector_array, 50, (800,100), canvas)
    #     cv2.imshow("skeleton_quaternions",canvas)
    #     cv2.waitKey(0)











# x, y = loadDataXYZ(tx = True, actions = ["DW"])
# f = open('truth.txt', 'a+')
# f.write(str(y))
# f.close()

# print(y)

# labels_dir = "../data/8Dec/labels/"
# path = "250/rx/IA_DW1.txt"


# f = open(labels_dir+path, "r")
# lines = f.readlines()

# doppler_file = pd.read_csv(lines[0][:-1])
# quat_file = pd.read_csv(lines[1][:-1])
# print(lines[1][:-1])
# for x in range(2,len(lines)):
#     line = lines[x].split(":")
#     doppler_time = line[0]
#     doppler = doppler_file.loc[:,doppler_time].values
#     doppler = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in doppler]
#     doppler = [[float(x[0]), float(x[1])] for x in doppler]
#     # print(doppler)
#     # print(len(doppler))
#     # print(len(doppler[0]))

#     indexes = line[1].strip("\n").strip('][ ').split(', ')
#     for index in indexes:
#         index = int(index)
#         quat = quat_file.iloc[index].values[1:]
#         quat = [(re.findall(r"[-+]?(?:\d*\.*\d+)",x)) for x in quat]
#         quat = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in quat]
#         print(quat)
#         # print(quat_file.iloc[index].values) 
#         break


#     # print(doppler_time)
#     # print(indexes)
#     break




#     # # print(df_matlab_file)
#     # # print(df_joint_file['datetime'])
#     # for i in range(len(df_joint_file['datetime'])):
#     #     df_joint_file['datetime'][i]=df_joint_file['datetime'][i][11:].replace(':','').replace('.','')
#     # joint_times = (df_joint_file['datetime'])
#     # matlab_times = df_matlab_file.columns
#     # # print(matlab_times)

#     # joint_times = [int(x[:-3]) for x in joint_times]
#     # matlab_times = [int(x) for x in matlab_times]
        

        
            




#     #     img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#     #     image = read_image(img_path)
#     #     label = self.img_labels.iloc[idx, 1]
#     #     # if self.transform: