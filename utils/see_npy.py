import numpy as np

data = np.load('breast_datasplit.npy', allow_pickle=True).item()
# print(data)

#
print(data['BMC'][1].keys())
print(data['BMC'][1]['x_train'])
print()
print(data['BMC'][1]['y_train'])
# print(len(data['BMC'][1]['y_test']))
#
# #

