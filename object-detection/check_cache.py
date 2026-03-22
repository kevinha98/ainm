import pickle
c3 = pickle.load(open('cache_v3_29.pkl','rb'))
c4 = pickle.load(open('cache_v4_29.pkl','rb'))
k = list(c3.keys())[0]
print('V3 keys:', list(c3[k].keys()))
print('V4 keys:', list(c4[k].keys()))
print('V3 full_1280 dets:', len(c3[k]['full_1280']))
print('V4 full_1280 dets:', len(c4[k]['full_1280']))
print('Image IDs:', sorted(c3.keys()))
