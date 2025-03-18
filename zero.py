import torchvision
import torch
import pickle
import h5py
import torchvision.models as models
import torch.nn as nn


# # 파일 로드
checkpoint = torch.load("./my_checkpoint_ssd.pth.tar", map_location="cpu")

# 모델 가중치 확인
state_dict = checkpoint['model'].state_dict()

print(checkpoint.keys())
print(checkpoint['epoch'])


# 가중치 목록 출력
# for name, param in state_dict.items():
#     print(name, param.shape)  # 각 가중치 텐서 이름과 크기 출력


# overlap = torch.arange(60).reshape(6, 10)
# print(overlap)

# overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0) # 10개

# # _, prior_for_each_object = overlap.max(dim=1) # 6개

# prior_for_each_object = torch.tensor([4, 4, 3, 2, 4, 3])
# print("---------------------------------------------------")
# print(object_for_each_prior)
# print("---------------------------------------------------")
# object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(6)) 

# print(object_for_each_prior)



# a = torch.LongTensor(range(21))
# print(a) 

# a = torch.rand(4, 5)
# print(a)
# b = a.max(dim=1)
# print("--------------------------------------------------------")
# print(b) # dim 0 들끼리의 비교.



# w = torch.empty(3, 5)
# print(w)
# nn.init.constant_(w, 0.3)
# print("-------------------------------")
# print(w)
# '''
# tensor([[7.0555e-38, 8.7721e-43, 0.0000e+00, 0.0000e+00, 0.0000e+00], 
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], 
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
# -------------------------------
# tensor([[0.3000, 0.3000, 0.3000, 0.3000, 0.3000], 
#         [0.3000, 0.3000, 0.3000, 0.3000, 0.3000], 
#         [0.3000, 0.3000, 0.3000, 0.3000, 0.3000]])
# '''


# vgg16 = models.vgg16(pretrained=True).state_dict()

# # for name, param in resnet.state_dict().items():
# #     print(name, param.shape)
# print(vgg16.keys())






# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# with h5py.File("Model_allsubjects1.h5", "r") as f:
#     def print_h5_structure(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
#         elif isinstance(obj, h5py.Group):
#             print(f"Group: {name}")

#     f["model_weights"].visititems(print_h5_structure)

# print("-----------------------------------------------------------c")
# with h5py.File("Model_allsubjects1.h5", "r") as f:
#     # model_weights 그룹 내 데이터셋 목록 확인
#     print(list(f["model_weights"].keys()))










# with h5py.File("Model_allsubjects1.h5", "r") as f:
#     print(list(f.keys()))  # 파일 내 데이터셋(그룹) 목록 확인 # ['model_weights', 'optimizer_weights']
    
#     # 특정 데이터셋 읽기
#     dataset = f["model_weights"]  # 전체 데이터 로드
#     print(dataset)

# A = torch.FloatTensor([1, 2, 3, 4]).unsqueeze(0)
# print("A: {}, shape: {}".format(A, A.shape))

# pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
# pretrained_param_names = list(pretrained_state_dict.keys())

# # 파일 로드
# checkpoint = torch.load("../checkpoint_ssd300.pth.tar", map_location="cpu")

# # 내부 데이터 확인
# print(checkpoint.keys())  # 저장된 키 목록 확인

# # 모델 가중치 확인
# state_dict = checkpoint['model'].state_dict()

# # 가중치 목록 출력
# for name, param in state_dict.items():
#     print(name, param.shape)  # 각 가중치 텐서 이름과 크기 출력

# # 특정 레이어 가중치 확인
# layer_name = "conv1.weight"  # 확인하고 싶은 레이어 이름
# if layer_name in state_dict:
#     weights = state_dict[layer_name].cpu().detach().numpy()
#     print(weights)  # numpy 배열로 출력
# else:
#     print(f"{layer_name} 레이어가 없습니다.")

# print(pretrained_state_dict['classifier.3.weight'].shape)
# print(pretrained_state_dict['classifier.0.bias'].shape) # torch.Size([4096])
# print(pretrained_state_dict['classifier.0.weight'].shape) # torch.Size([4096, 25088])

''' print(pretrained_param_names)
['features.0.weight', 'features.0.bias', 
'features.2.weight', 'features.2.bias', 
'features.5.weight', 'features.5.bias', 
'features.7.weight', 'features.7.bias', 
'features.10.weight', 'features.10.bias', 
'features.12.weight', 'features.12.bias', 
'features.14.weight', 'features.14.bias', 
'features.17.weight', 'features.17.bias', 
'features.19.weight', 'features.19.bias', 
'features.21.weight', 'features.21.bias', 
'features.24.weight', 'features.24.bias', 
'features.26.weight', 'features.26.bias', 
'features.28.weight', 'features.28.bias', 
'classifier.0.weight', 'classifier.0.bias', 
'classifier.3.weight', 'classifier.3.bias', 
'classifier.6.weight', 'classifier.6.bias']
'''

'''
rescale_factors torch.Size([1, 512, 1, 1])
base.conv1_1.weight torch.Size([64, 3, 3, 3])
base.conv1_1.bias torch.Size([64])
base.conv1_2.weight torch.Size([64, 64, 3, 3])
base.conv1_2.bias torch.Size([64])
base.conv2_1.weight torch.Size([128, 64, 3, 3])
base.conv2_1.bias torch.Size([128])
base.conv2_2.weight torch.Size([128, 128, 3, 3])
base.conv2_2.bias torch.Size([128])
base.conv3_1.weight torch.Size([256, 128, 3, 3])
base.conv3_1.bias torch.Size([256])
base.conv3_2.weight torch.Size([256, 256, 3, 3])
base.conv3_2.bias torch.Size([256])
base.conv3_3.weight torch.Size([256, 256, 3, 3])
base.conv3_3.bias torch.Size([256])
base.conv4_1.weight torch.Size([512, 256, 3, 3])
base.conv4_1.bias torch.Size([512])
base.conv4_2.weight torch.Size([512, 512, 3, 3])
base.conv4_2.bias torch.Size([512])
base.conv4_3.weight torch.Size([512, 512, 3, 3])
base.conv4_3.bias torch.Size([512])
base.conv5_1.weight torch.Size([512, 512, 3, 3])
base.conv5_1.bias torch.Size([512])
base.conv5_2.weight torch.Size([512, 512, 3, 3])
base.conv5_2.bias torch.Size([512])
base.conv5_3.weight torch.Size([512, 512, 3, 3])
base.conv5_3.bias torch.Size([512])
base.conv6.weight torch.Size([1024, 512, 3, 3])
base.conv6.bias torch.Size([1024])
base.conv7.weight torch.Size([1024, 1024, 1, 1])
base.conv7.bias torch.Size([1024])
aux_convs.conv8_1.weight torch.Size([256, 1024, 1, 1])
aux_convs.conv8_1.bias torch.Size([256])
aux_convs.conv8_2.weight torch.Size([512, 256, 3, 3])
aux_convs.conv8_2.bias torch.Size([512])
aux_convs.conv9_1.weight torch.Size([128, 512, 1, 1])
aux_convs.conv9_1.bias torch.Size([128])
aux_convs.conv9_2.weight torch.Size([256, 128, 3, 3])
aux_convs.conv9_2.bias torch.Size([256])
aux_convs.conv10_1.weight torch.Size([128, 256, 1, 1])
aux_convs.conv10_1.bias torch.Size([128])
aux_convs.conv10_2.weight torch.Size([256, 128, 3, 3])
aux_convs.conv10_2.bias torch.Size([256])
aux_convs.conv11_1.weight torch.Size([128, 256, 1, 1])
aux_convs.conv11_1.bias torch.Size([128])
aux_convs.conv11_2.weight torch.Size([256, 128, 3, 3])
aux_convs.conv11_2.bias torch.Size([256])
pred_convs.loc_conv4_3.weight torch.Size([16, 512, 3, 3])
pred_convs.loc_conv4_3.bias torch.Size([16])
pred_convs.loc_conv7.weight torch.Size([24, 1024, 3, 3])
pred_convs.loc_conv7.bias torch.Size([24])
pred_convs.loc_conv8_2.weight torch.Size([24, 512, 3, 3])
pred_convs.loc_conv8_2.bias torch.Size([24])
pred_convs.loc_conv9_2.weight torch.Size([24, 256, 3, 3])
pred_convs.loc_conv9_2.bias torch.Size([24])
pred_convs.loc_conv10_2.weight torch.Size([16, 256, 3, 3])
pred_convs.loc_conv10_2.bias torch.Size([16])
pred_convs.loc_conv11_2.weight torch.Size([16, 256, 3, 3])
pred_convs.loc_conv11_2.bias torch.Size([16])
pred_convs.cl_conv4_3.weight torch.Size([84, 512, 3, 3])
pred_convs.cl_conv4_3.bias torch.Size([84])
pred_convs.cl_conv7.weight torch.Size([126, 1024, 3, 3])
pred_convs.cl_conv7.bias torch.Size([126])
pred_convs.cl_conv8_2.weight torch.Size([126, 512, 3, 3])
pred_convs.cl_conv8_2.bias torch.Size([126])
pred_convs.cl_conv9_2.weight torch.Size([126, 256, 3, 3])
pred_convs.cl_conv9_2.bias torch.Size([126])
pred_convs.cl_conv10_2.weight torch.Size([84, 256, 3, 3])
pred_convs.cl_conv10_2.bias torch.Size([84])
pred_convs.cl_conv11_2.weight torch.Size([84, 256, 3, 3])
pred_convs.cl_conv11_2.bias torch.Size([84])
'''