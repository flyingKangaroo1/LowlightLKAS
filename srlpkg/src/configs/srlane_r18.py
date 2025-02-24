"""
SRLane model settings.(cfg files) --- 로봇 카메라 픽셀 확인 후 오리지날 이미지 크기 수정 필요
"""

ori_img_w = 640 # 로봇 카메라 픽셀 확인 (imgpub : 1920)
ori_img_h = 360  # 로봇 카메라 픽셀 확인 (imgpub : 1080)
img_w = 800
img_h = 320
cut_height = 300 # 자르는 높이 확인 (imgpub : 200)
max_lanes = 4
n_strips = 71
n_offsets = 72

gpus = 0

angle_map_size = (4, 10)
hidden_dim = 64
z_mean = [0.9269]
z_std = [0.2379]
n_fpn = 3
feat_ds_strides = [8, 16, 32]
num_points = 72

net = dict(type="TwoStageDetector")

backbone = dict(type="ResNetWrapper",
                resnet="resnet18",
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,)

neck = dict(type="ChannelMapper",
            in_channels=[128, 256, 512],
            out_channels=hidden_dim,
            num_outs=3,)

rpn_head = dict(type="LocalAngleHead",
                num_points=num_points,
                in_channel=hidden_dim,)

roi_head = dict(type="CascadeRefineHead",
                refine_layers=1,
                fc_hidden_dim=hidden_dim * 3,
                prior_feat_channels=hidden_dim,
                sample_points=36,  # 36
                num_groups=6,)
test_process = [
    dict(type="GenerateLaneLine",
         transforms=[
             dict(name="Resize",
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type="ToTensor", keys=["img"]),
]
test_parameters = dict(conf_threshold=0.02, nms_thres=50, nms_topk=max_lanes) # conf_threshold도 깎아야함.


