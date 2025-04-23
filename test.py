import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse
import matplotlib.pyplot as plt

from model import PyramidFlow
from util import MOAIDataloader
from util import fix_randseed

from pathfilemgr import MPathFileManager
from hyp_data import MHyp

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')

opt = parser.parse_args()

##########################################################################################
### make dir and hyp
##########################################################################################

mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
mpfm.load_test_hyp(mhyp)


class IPyramidFlow(nn.Module):
    def __init__(self, flow_model, feat_mean):
        super().__init__()
        self.flow = flow_model
        self.feat_mean = feat_mean

    def forward(self, image):
        # 1. 피라미드 추출
        pyramid2 = self.flow(image) # flow 모델 실행

        # 3. pyramid_diff 계산
        pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, self.feat_mean)]

        # 4. 피라미드 합성
        # compose_pyramid는 flow 모델의 pyramid 객체에 있으므로 접근
        output = self.flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)

        return output

def test(resnetX, num_layer, vn_dims, ksize, channel, num_stack, device, mpfm):
    message = {
        "status": "prepare",
        "message": "추론 환경을 준비중입니다.",
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

    loader_dict = fix_randseed(seed=0)

    model_data = np.load(f'{mpfm.weight_path}/best.npz', allow_pickle=True)

    print(f'{mpfm.weight_path}/best.npz')

    cls_name = model_data['cls_name'].item() if 'cls_name' in model_data else cls_name
    resnetX = model_data['resnetX'] if np.isscalar(model_data['resnetX']) else model_data['resnetX'].item()
    num_layer = model_data['num_layer'] if np.isscalar(model_data['num_layer']) else model_data['num_layer'].item()
    vn_dims = (0, 1)# model_data['vn_dims'] # 튜플/리스트이므로 item()을 호출하지 않음
    ksize = model_data['ksize'] if np.isscalar(model_data['ksize']) else model_data['ksize'].item()
    channel = model_data['channel'] if np.isscalar(model_data['channel']) else model_data['channel'].item()
    num_stack = model_data['num_stack'] if np.isscalar(model_data['num_stack']) else model_data['num_stack'].item()
    #x_size = 256 if resnetX==0 else 1024
    #resnetX = 0 if resnetX=='none' else int(resnetX[6:])

    x_size = 256 if resnetX==0 else 1024

    print(f"클래스: {cls_name}")
    print(f"ResNet: {resnetX}")
    print(f"층 수: {num_layer}")
    print(f"정규화 차원: {vn_dims}")
    print(f"커널 크기: {ksize}")
    print(f"채널 수: {channel}")
    print(f"스택 수: {num_stack}")

    print(resnetX, channel, num_layer, num_stack, ksize, vn_dims, False)

    # model
    flow = PyramidFlow(resnetX, channel, num_layer, num_stack, ksize, vn_dims, False).to(device)

    try:
        # 키가 있는지 확인
        if 'state_dict_pixel' in model_data:
            print(f"[I]Pixel AUROC 기준 최적 모델 사용 중")
            state_dict_key = 'state_dict_pixel'
        elif 'state_dict_pro' in model_data:
            print(f"[I]PRO Score 기준 최적 모델 사용 중")
            state_dict_key = 'state_dict_pro'
        else:
            raise ValueError("모델 상태를 찾을 수 없습니다")
        
        # state_dict 가져오기
        state_dict_data = model_data[state_dict_key].item()
        print(f"[I]상태 딕셔너리 키 개수: {len(state_dict_data)}")
            
        # 일부 키만 출력
        some_keys = list(state_dict_data.keys())[:5]
        print(f"[I]상태 딕셔너리 일부 키: {some_keys}")
            
        # 상태 딕셔너리 변환
        state_dict = {}
        for k, v in state_dict_data.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v
            else:
                state_dict[k] = torch.from_numpy(v)
            
        # 모델 가중치 로딩
        print("[I]모델 가중치 로딩 중...")
        missing_keys, unexpected_keys = flow.load_state_dict(state_dict, strict=False)
            
        if len(missing_keys) > 0:
            print(f"[I]누락된 키: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[I]예상치 못한 키: {unexpected_keys}")
                
    except Exception as e:
        print(f"[E]모델 로딩 오류: {str(e)}")
        print("[E]가중치 로딩을 건너뛰고 계속 진행합니다.")

    # dataset
    val_dataset = MOAIDataloader(mode='val', x_size=x_size, y_size=256, datapath=mpfm.train_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, drop_last=False, **loader_dict)
    test_dataset = MOAIDataloader(mode='test', x_size=x_size, y_size=256, datapath=mpfm.test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, **loader_dict)

    message = {
        "status": "start",
        "message": "모델 추론을 시작합니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

    feat_mean = loadfm(device)

    # test
    flow.eval()
    diff_list, labels_list = [], []
    for test_dict in test_loader:
        image, fname = test_dict['images'].to(device), test_dict['fname']
        with torch.no_grad():
            pyramid2 = flow(image) 
            pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, feat_mean)]
            diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)# b,1,h,w
            diff_list.append(diff.cpu())

            diff = np.squeeze(diff.cpu().numpy())
            amap_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            np.savez(f'{mpfm.test_result}/{fname[0]}.npz', amap_norm=amap_norm)
            # plt.imsave(f'{mpfm.test_result}/{fname[0]}.png', amap_norm, cmap='jet')
            
    message = {
        "status": "complete",
        "message": "모델 추론이 완료되었습니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def loadfm(device):
    data = np.load(f'{mpfm.weight_path}/feat_mean.npz', allow_pickle=True)
    feat_mean_arr = data['feat_mean']
    feat_mean_tensor = [torch.from_numpy(arr).to(device) for arr in feat_mean_arr]
    return feat_mean_tensor

if __name__ == '__main__':

    if mhyp.volumeNorm == 'CVN':
        vn_dims = (0, 1)
    elif mhyp.volumeNorm == 'SVN':
        vn_dims = (0, 2, 3)

    resnetX = 0 if mhyp.encoder=='none' else int(mhyp.encoder[6:])

    test(mhyp.encoder, mhyp.numLayer, vn_dims, \
          ksize=mhyp.kernelSize, channel=mhyp.numChannel, num_stack=mhyp.numStack, \
          device=f'cuda:{mhyp.gpu}', mpfm=mpfm)

