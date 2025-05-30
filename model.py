import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
from scipy import linalg as la

from util import kornia_filter2d
from autoFlow import SequentialNF, InvertibleModule, SequentialNet

'''
class SemiInvertible_1x1Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        assert out_channels >= in_channels
        super(SemiInvertible_1x1Conv, self).__init__(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.weight)  # Orthogonal initialization

    def forward(self, input):
        # Ensure input dimensions are as expected
        assert input.size(1) == self.in_channels
        return super(SemiInvertible_1x1Conv, self).forward(input)

    def inverse(self, output):
        # Ensure output dimensions are as expected
        assert output.size(1) == self.out_channels
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(2).unsqueeze(3)
        return F.conv2d(output, weight_inv)
'''

class SemiInvertible_1x1Conv(nn.Conv2d):
    """
        Semi-invertible 1x1Conv is used at the first stage of NF.
    """
    def __init__(self, in_channels, out_channels ) -> None:
        # assert out_channels >= in_channels
        super().__init__(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.weight.data) # orth initialization
    def inverse(self, output):
        b, c, h, w = output.shape
        A = self.weight[..., 0,0] # outch, inch
        B = output.permute([1,0,2,3]).reshape(c, -1) # outch, bhw
        X = torch.linalg.lstsq(A, B)  # AX=B
        return X.solution.reshape(-1, b, h, w).permute([1, 0, 2, 3])
    @property
    def logdet(self):
        w = self.weight.squeeze() # out,in
        return 0.5*torch.logdet(w.T@w)


# class LaplacianMaxPyramid(nn.Module):
#     def __init__(self, num_levels) -> None:
#         super().__init__()
#         # 원본 커널을 버퍼로 등록 (device, dtype 자동 관리)
#         kernel_tensor = torch.tensor(
#             [[ [[1., 4., 6., 4., 1.],
#                 [4., 16., 24., 16., 4.],
#                 [6., 24., 36., 24., 6.],
#                 [4., 16., 24., 16., 4.],
#                 [1., 4., 6., 4., 1.]] ]]
#         ) / 256.0
#         self.register_buffer('base_kernel', kernel_tensor) # (1, 1, 5, 5) 형태
#         self.num_levels = num_levels - 1

#     def _apply_blur(self, input_tensor):
#         """ Reshape 트릭을 사용하여 ONNX 호환성을 높인 블러링 """
#         B, C, H, W = input_tensor.shape
#         device = input_tensor.device
#         dtype = input_tensor.dtype

#         # 1. 입력 Reshape: (B, C, H, W) -> (B*C, 1, H, W)
#         input_reshaped = input_tensor.view(B * C, 1, H, W)

#         # 2. 단일 커널 준비: (1, 1, 5, 5) 형태, device/dtype 맞춤
#         kernel_single = self.base_kernel.to(device=device, dtype=dtype)

#         # 3. 표준 Conv2d 적용 (groups=1)
#         output_reshaped = F.conv2d(input_reshaped, kernel_single, groups=1, padding=0, stride=1)

#         # 4. 출력 Reshape: (B*C, 1, H_out, W_out) -> (B, C, H_out, W_out)
#         _BXC, _Ch_out, H_out, W_out = output_reshaped.shape
#         output_tensor = output_reshaped.view(B, C, H_out, W_out)

#         return output_tensor

#     def _pyramid_down(self, input, pad_mode='constant'):
#         if not len(input.shape) == 4: raise ValueError(f'Invalid shape {input.shape}')
#         img_pad = F.pad(input, (2,2,2,2), mode=pad_mode)
#         img_blur = self._apply_blur(img_pad) # 수정된 함수 호출
#         out = F.max_pool2d(img_blur, kernel_size=2, stride=2)
#         return out

#     def _pyramid_up(self, input, size, pad_mode='constant'):
#         if not len(input.shape) == 4: raise ValueError(f'Invalid shape {input.shape}')
#         # mode='nearest'는 ONNX 변환에 유리
#         img_up = F.interpolate(input, size=size, mode='nearest')
#         img_pad = F.pad(img_up, (2,2,2,2), mode=pad_mode)
#         img_blur = self._apply_blur(img_pad) # 수정된 함수 호출
#         return img_blur

#     def build_pyramid(self, input):
#         gp, lp = [input], []
#         if self.num_levels < 0:
#              # logger.warning("num_levels is less than 0, returning input as pyramid.") # 로깅 추가 가능
#              return [input]
#         current_level_tensor = input
#         for _ in range(self.num_levels):
#              # 다음 레벨 텐서 계산 중 에러 발생 가능성 고려
#              try:
#                  next_gp = self._pyramid_down(current_level_tensor)
#                  gp.append(next_gp)
#                  current_level_tensor = next_gp
#              except Exception as e:
#                  # logger.error(f"Error during pyramid downsampling: {e}")
#                  raise e # 또는 다른 오류 처리

#         for layer in range(self.num_levels):
#             curr_gp = gp[layer]
#             # size 인자는 (H, W) 튜플이어야 함
#             target_size = curr_gp.shape[2:]
#             # 다음 레벨 업샘플링 중 에러 발생 가능성 고려
#             try:
#                  next_gp_up = self._pyramid_up(gp[layer+1], size=target_size)
#             except Exception as e:
#                  # logger.error(f"Error during pyramid upsampling: {e}")
#                  raise e
#             # 뺄셈 전 shape 확인 (디버깅 시 유용)
#             # if curr_gp.shape != next_gp_up.shape:
#             #     logger.warning(f"Shape mismatch at level {layer}: {curr_gp.shape} vs {next_gp_up.shape}")
#             lp.append(curr_gp - next_gp_up)
#         lp.append(gp[self.num_levels])
#         return lp

#     def compose_pyramid(self, lp):
#         if not lp:
#              # logger.warning("compose_pyramid called with empty list.")
#              return None # 빈 리스트 입력 시 처리
#         rs = lp[-1]
#         for i in range(len(lp)-2, -1, -1):
#              target_size = lp[i].shape[2:]
#              try:
#                  rs_up = self._pyramid_up(rs, size=target_size)
#              except Exception as e:
#                  # logger.error(f"Error during pyramid upsampling in compose: {e}")
#                  raise e
#              # 덧셈 전 shape 확인
#              # if rs_up.shape != lp[i].shape:
#              #     logger.warning(f"Shape mismatch during compose at level {i}: {rs_up.shape} vs {lp[i].shape}")
#              rs = torch.add(rs_up, lp[i])
#         return rs

class LaplacianMaxPyramid(nn.Module):
    def __init__(self, num_levels) -> None:
        super().__init__()
        self.kernel = torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )/ 256.0
        self.num_levels = num_levels-1 # 总共有num_levels层，

    def _pyramid_down(self, input, pad_mode='constant'):
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        # blur 
        img_pad = F.pad(input, (2,2,2,2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        # downsample
        out = F.max_pool2d(img_blur, kernel_size=2, stride=2)# 使用max pooling取代下采样
        return out

    def _pyramid_up(self, input, size, pad_mode='constant'):
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        # upsample
        img_up = F.interpolate(input, size=size, mode='nearest', )
        # blur
        img_pad = F.pad(img_up, (2,2,2,2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        return img_blur

    def build_pyramid(self, input):
        gp, lp = [input], []
        for _ in range(self.num_levels):
            gp.append(self._pyramid_down(gp[-1]))
        for layer in range(self.num_levels):
            curr_gp = gp[layer]
            next_gp = self._pyramid_up(gp[layer+1], size=curr_gp.shape[2:])
            lp.append(curr_gp - next_gp)
        lp.append(gp[self.num_levels]) # 最后一层不是gp
        return lp

    def compose_pyramid(self, lp):
        rs = lp[-1]
        for i in range(len(lp)-2, -1, -1):
            rs = self._pyramid_up(rs, size=lp[i].shape[2:])
            rs = torch.add(rs, lp[i])
        return rs

class VolumeNorm(nn.Module):
    """
        Volume Normalization.
        CVN dims = (0,1);  SVN dims = (0,2,3)
    """
    def __init__(self, dims=(0,1) ):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(1,1,1,1))
        self.momentum = 0.1
        self.dims = dims
    def forward(self, x):
        if self.training:
            sample_mean = torch.mean(x, dim=self.dims, keepdim=True) 
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * sample_mean
            out = x - sample_mean
        else:
            out = x - self.running_mean
        return out

class AffineParamBlock(nn.Module):
    """
        Estimate `slog` and `t`.
    """
    def __init__(self, in_ch, out_ch=None, hidden_ch=None, ksize=7, clamp=2, vn_dims=(0,1)):
        super().__init__()
        if out_ch is None: 
            out_ch = 2*in_ch 
        if hidden_ch is None:
            hidden_ch = out_ch
        self.clamp = clamp
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=ksize, padding=ksize//2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=ksize, padding=ksize//2, bias=False),
        )
        nn.init.zeros_(self.conv[-1].weight.data)
        self.norm = VolumeNorm(vn_dims)
    def forward(self, input, forward_mode:bool):
        output = self.conv(input)
        _dlogdet, bias = output.chunk(2, 1)
        dlogdet = self.clamp * 0.636 * torch.atan(_dlogdet / self.clamp)  # soft clip
        dlogdet = self.norm(dlogdet)
        scale = torch.exp(dlogdet)
        return (scale, bias), dlogdet # scale * x + bias

class InvConv2dLU(nn.Module):
    """
        Invertible 1x1Conv with volume normalization.
    """
    def __init__(self, in_channel, volumeNorm=True):
        super().__init__()
        self.volumeNorm = volumeNorm
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(w_s.abs().log())
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def inverse(self, output):
        _, _, height, width = output.shape
        weight = self.calc_weight()
        inv_weight = torch.inverse(weight.squeeze().double()).float()
        input = F.conv2d(output, inv_weight.unsqueeze(2).unsqueeze(3))
        return input


    '''
    def calc_weight(self):
        if self.volumeNorm:
            w_s = self.w_s - self.w_s.mean() # volume normalization
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)
    '''
    
    def calc_weight(self):
        if self.volumeNorm:
            w_s_processed = self.w_s - self.w_s.mean() # 변수명 변경 (혼동 방지)
        else:
            w_s_processed = self.w_s
        
        # 원본 코드:
        # diag_matrix = torch.diag(self.s_sign * torch.exp(w_s_processed))

        # 수정 코드: 단위 행렬과 곱셈으로 대체
        diag_elements = self.s_sign * torch.exp(w_s_processed)
        N = diag_elements.shape[0] # 대각 요소 개수 (벡터 길이)
        # 단위 행렬 생성 (디바이스와 타입 일치)
        identity_matrix = torch.eye(N, device=diag_elements.device, dtype=diag_elements.dtype) 
        # 단위 행렬과 대각 요소 벡터 곱셈 (벡터를 (N, 1) 형태로 바꿔 브로드캐스팅)
        diag_matrix = identity_matrix * diag_elements.unsqueeze(1) 

        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + diag_matrix) # 수정된 diag_matrix 사용
        )
        return weight.unsqueeze(2).unsqueeze(3)



class FlowBlock(InvertibleModule):
    """
        @Paper Figure3(c) The proposed scale-wise pyramid coupling block.
    """
    def __init__(self, channel, direct, start_level, ksize, vn_dims):
        super().__init__()
        # assert direct in ['up', 'down']
        self.direct = direct
        self.start_idx = start_level
        self.affineParams = AffineParamBlock(channel, ksize=ksize, vn_dims=vn_dims)
        self.conv1x1 = InvConv2dLU(channel)

    def forward(self, inputs, logdets):
        # assert self.start_idx+1 < len(inputs)
        x0, x1 = inputs[self.start_idx: self.start_idx+2]
        logdet0, logdet1 = logdets[self.start_idx: self.start_idx+2]
        if self.direct == 'up':
            y10 = F.interpolate(x1, size=x0.shape[2:], mode='nearest') # interp first
            (scale0, bias0), dlogdet0 = self.affineParams(y10, forward_mode=True)
            z0, z1 = scale0*x0+bias0, x1
            z0 = self.conv1x1(z0) 
            dlogdet1 = 0
        else:
            (scale10, bias10), dlogdet10 = self.affineParams(x0, forward_mode=True)
            scale1, bias1, dlogdet1 = F.interpolate(scale10, size=x1.shape[2:], mode='nearest'),\
                                         F.interpolate(bias10, size=x1.shape[2:], mode='nearest'),\
                                             F.interpolate(dlogdet10, size=x1.shape[2:], mode='nearest') # interp after
            z0, z1 = x0, scale1*x1+bias1
            z1 = self.conv1x1(z1) 
            dlogdet0 = 0
        outputs = inputs[:self.start_idx]+(z0, z1)+inputs[self.start_idx+2:]
        out_logdets = logdets[:self.start_idx]+(logdet0+dlogdet0, logdet1+dlogdet1)+logdets[self.start_idx+2:]
        return outputs, out_logdets

    def inverse(self, outputs, logdets):
        # assert self.start_idx+1 < len(outputs)
        z0, z1 = outputs[self.start_idx: self.start_idx+2]
        logdet0, logdet1 = logdets[self.start_idx: self.start_idx+2]
        if self.direct == 'up':
            z0 = self.conv1x1.inverse(z0)
            z10 = F.interpolate(z1, size=z0.shape[2:], mode='nearest') # interp first
            (scale0, bias0), dlogdet0 = self.affineParams(z10, forward_mode=False)
            x0, x1 = (z0-bias0)/scale0, z1
            dlogdet1 = 0
        else:
            z1 = self.conv1x1.inverse(z1)
            (scale01, bias01), dlogdet01 = self.affineParams(z0, forward_mode=False)
            scale1, bias1, dlogdet1 = F.interpolate(scale01, size=z1.shape[2:], mode='nearest'),\
                                         F.interpolate(bias01, size=z1.shape[2:], mode='nearest'),\
                                             F.interpolate(dlogdet01, size=z1.shape[2:], mode='nearest') # interp after
            x0, x1 = z0, (z1-bias1)/scale1
            dlogdet0 = 0
        inputs = outputs[:self.start_idx]+(x0, x1)+outputs[self.start_idx+2:]
        in_logdets = logdets[:self.start_idx]+(logdet0-dlogdet0, logdet1-dlogdet1)+logdets[self.start_idx+2:]
        return inputs, in_logdets


class FlowBlock2(InvertibleModule):
    """
        @Paper Figure3(d) The reverse parallel and reparameterized of (c)-architecture.
    """
    def __init__(self, channel, start_level, ksize, vn_dims):
        super().__init__()
        self.start_idx = start_level
        self.affineParams = AffineParamBlock(in_ch=2*channel, out_ch=2*channel, ksize=ksize, vn_dims=vn_dims)
        self.conv1x1 = InvConv2dLU(channel)

    def forward(self, inputs, logdets):
        x0, x1, x2 = inputs[self.start_idx: self.start_idx+3]
        logdet0, logdet1, logdet2 = logdets[self.start_idx: self.start_idx+3]
        y01 = F.interpolate(x0, size=x1.shape[2:], mode='nearest')
        y21 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        affine_input = torch.concat([y01, y21], dim=1) # b, 2*ch, h, w
        (scale1, bias1), dlogdet1 = self.affineParams(affine_input, forward_mode=True)
        z0, z1, z2 = x0, scale1*x1+bias1, x2
        z1 = self.conv1x1(z1)
        outputs = inputs[:self.start_idx]+(z0, z1, z2)+inputs[self.start_idx+3:]
        out_logdets = logdets[:self.start_idx]+(logdet0, logdet1+dlogdet1, logdet2)+logdets[self.start_idx+3:]
        return outputs, out_logdets

    def inverse(self, outputs, logdets):
        z0, z1, z2 = outputs[self.start_idx: self.start_idx+3]
        logdet0, logdet1, logdet2 = logdets[self.start_idx: self.start_idx+3]
        z1 = self.conv1x1.inverse(z1)
        z01 = F.interpolate(z0, size=z1.shape[2:], mode='nearest')
        z21 = F.interpolate(z2, size=z1.shape[2:], mode='nearest')
        affine_input = torch.concat([z01, z21], dim=1) # b, 2*ch, h, w
        (scale1, bias1), dlogdet1 = self.affineParams(affine_input, forward_mode=False)
        x0, x1, x2 = z0, (z1-bias1)/scale1, z2
        inputs = outputs[:self.start_idx]+(x0, x1, x2)+outputs[self.start_idx+3:]
        in_logdets = logdets[:self.start_idx]+(logdet0, logdet1-dlogdet1, logdet2)+logdets[self.start_idx+3:]
        return inputs, in_logdets


class PyramidFlow(nn.Module):
    """
        PyramidFlow
        NOTE: resnetX=0 use 1x1 conv with #channel channel.
    """
    def __init__(self, resnetX, channel, num_level, num_stack, ksize, vn_dims, savemem=False):
        super().__init__()
        # assert num_level >= 2
        # assert resnetX in [18, 34, 0]
        self.channel = channel if resnetX==0 else 64
        self.num_level = num_level
        
        modules = []
        for _ in range(num_stack):
            for range_start in [0, 1]:
                if range_start == 1:
                    modules.append(FlowBlock(self.channel, direct='up', start_level=0, ksize=ksize, vn_dims=vn_dims))
                for start_idx in range(range_start, num_level, 2):
                    if start_idx+2 < num_level:
                        modules.append(FlowBlock2(self.channel, start_level=start_idx, ksize=ksize, vn_dims=vn_dims))
                    elif start_idx+1 < num_level:
                        modules.append(FlowBlock(self.channel, direct='down', start_level=start_idx, ksize=ksize, vn_dims=vn_dims))
        self.nf = SequentialNF(modules) if savemem else SequentialNet(modules)

        if resnetX != 0:
            # resnet = models.resnet18(pretrained=True, ) if resnetX==18 else models.resnet34(pretrained=True, )
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, ) if resnetX==18 else models.resnet34(weights=models.ResNet34_Weights.DEFAULT, )
            self.inconv = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            )# 1024->256
        else:
            self.inconv = SemiInvertible_1x1Conv(3, self.channel)

        self.pyramid = LaplacianMaxPyramid(num_level)

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        # assert h%(2**(self.num_level-1))==0 and w%(2**(self.num_level-1))==0
        with torch.no_grad():
            feat1 = self.inconv(imgs) # fix inconv/encoder
        pyramid = self.pyramid.build_pyramid(feat1)
        logdets = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid)
        pyramid_out, logdets_out = self.nf.forward(pyramid, logdets)
        return pyramid_out

    def inverse(self, pyramid_out):
        logdets_out = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid_out)
        pyramid_in, logdets_in = self.nf.inverse(pyramid_out, logdets_out)
        feat1 = self.pyramid.compose_pyramid(pyramid_in)
        if self.channel != 64:
            input = self.inconv.inverse(feat1)
            return input
        return feat1
