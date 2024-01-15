import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
#from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  
        
        
        return P3_out

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class Motion_coupling_Neck(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        self.weight = nn.ParameterList(torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True) for _ in range(num_frame))
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        # 参考帧分别与关键帧融合
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

    def forward(self, feats):
        f_feats = []
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))*self.weight[i] for i in range(self.num_frame-1)], dim=0)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=0))
        c_feat = self.conv_final(torch.cat([r_feat,c_feat], dim=1))
        f_feats.append(c_feat)
            
        return f_feats


class Network(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=10):
        super(Network, self).__init__()
        self.num_frame = num_frame
        self.backbone = Feature_Extractor(0.33,0.50) 

        #-----------------------------------------#
        #   尺度感知模块
        #-----------------------------------------#
        self.neck = Motion_coupling_Neck(channels=[128], num_frame=num_frame)
        #----------------------------------------------------------#
        #   head
        #----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")
        
        
        self.mapping0 = nn.Sequential(
                nn.Conv2d(128*num_frame, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU())
        self.SSTNet = SSTNet(input_dim=64, hidden_dim=[64], kernel_size=(3, 3), num_slices=1, num_layers=1)
        self.mapping1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=4, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU()) 
        self.mapping2 = nn.Sequential(
                nn.Conv2d(64*num_frame, 128*num_frame, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU(),
                nn.AdaptiveMaxPool2d(64))

        self.MSE = nn.MSELoss()
        
        
    def forward(self, inputs): #4, 3, 5, 512, 512
        feat = []
        for i in range(self.num_frame):
            feat.append(self.backbone(inputs[:,:,i,:,:]))
        """[b,128,32,32][b,256,16,16][b,512,8,8]"""
        
        if self.training:
            feat_S = torch.cat([self.mapping1(inputs[:,:,i,:,:]).unsqueeze(1) for i in range(self.num_frame)], 1) 
            lstm_output, _ = self.SSTNet(feat_S)
            motion_feat = lstm_output[-1]

            motion_loss = self.MSE(torch.cat(feat, 1), self.mapping2(torch.cat([motion_feat[:,i,:,:,:] for i in range(self.num_frame)], dim=1)))
  
        if self.neck:
            feat = self.neck(feat) #4, 128, 64, 64

        outputs  = self.head(feat)
        
        if self.training:
            return  outputs, motion_loss  
        else:
            return  outputs
            


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)  



class Cross_Slice_ConvLSTM_Node(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(Cross_Slice_ConvLSTM_Node, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 
        self.conv2 = nn.Conv2d(in_channels=4 * self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 
        self.motion = nn.Sequential(nn.Conv2d(in_channels=3 * self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias))

        self.mm1 = nn.Conv2d(2 * self.input_dim, 1 * self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.mm2 = nn.Conv2d(2 * self.input_dim, 1 * self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.past_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8,
                                               dropout=0.1, batch_first=True)
        self.future_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8,
                                               dropout=0.1, batch_first=True)
        self.pool = nn.AdaptiveMaxPool2d(32)
        

    def forward(self, input_tensor, input_head, all_state, cur_state, multi_head): 
        h_cur, c_cur = cur_state 

        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined) 
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
  
        m_h, m_c = multi_head

        combined2 = torch.cat([input_tensor, h_cur, input_head, m_h], dim=1) 
        combined_conv2 = self.conv2(combined2) 

        mm_i, mm_f, mm_o, mm_g = torch.split(combined_conv2, self.hidden_dim, dim=1)
        
        m_i = torch.sigmoid(mm_i+cc_i) 
        m_f = torch.sigmoid(mm_f+cc_f)
        m_o = torch.sigmoid(mm_o+cc_o)
        m_g = torch.tanh(mm_g+cc_g)
        
        c_next = m_f * c_cur + m_i * m_g       
        h_next = m_o * torch.tanh(c_next) 

        mh_feat = self.pool(m_h).flatten(start_dim=2,end_dim=3)
        input_tensor_feat =  self.pool(all_state[:,0,:,:,:]).flatten(start_dim=2,end_dim=3)
        input_head_feat = self.pool(all_state[:,-1,:,:,:]).flatten(start_dim=2,end_dim=3)
        
        mh_feat, _ = self.past_attention(input_tensor_feat, mh_feat, mh_feat) #4, 64, 1024
        input_feat, _ = self.future_attention(input_head_feat, mh_feat, mh_feat)
        
        mh_feat = mh_feat.view(all_state.shape[0],self.hidden_dim,32,32)  
        
        mh_feat = F.interpolate(mh_feat, size=[all_state.shape[3], all_state.shape[3]], mode='bilinear', align_corners=True)

        input_feat = input_feat.view(all_state.shape[0],self.hidden_dim,32,32)  
        input_feat = F.interpolate(input_feat, size=[all_state.shape[3], all_state.shape[3]], mode='bilinear', align_corners=True)

        motion1 = torch.cat([torch.sigmoid(mh_feat)+m_h, m_h],1) 
        motion1 = self.mm1(motion1)
        motion2 = torch.cat([torch.sigmoid(input_feat)+m_h, m_h],1)
        motion2 = self.mm2(motion2)

        motion_feat = torch.cat([input_head, motion1, motion2], 1)
        motion = self.motion(motion_feat)
        motion_i, motion_f, motion_g, motion_o = torch.split(motion, self.hidden_dim, dim=1)

        motion_i = torch.sigmoid(motion_i)
        motion_f = torch.sigmoid(motion_f)
        motion_o = torch.sigmoid(motion_o)
        motion_g = torch.tanh(motion_g)

        m_c_next = motion_f * m_c + motion_i * motion_g + c_next
        m_h_next = motion_o * torch.tanh(m_c_next) + h_next
        
        return h_next, c_next, m_h_next, m_c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
   
  

class SSTNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_slices, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(SSTNet, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.num_slices = num_slices
        cell_list = {}
       
        for i in range(0, self.num_slices):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            for j in range(0, self.num_layers):
                
                cell_list.update({'%d%d'%(i,j): Cross_Slice_ConvLSTM_Node(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias)}) 
                                          
        self.cell_list = nn.ModuleDict(cell_list)
        

    def forward(self, input_tensor, hidden_state=None):
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            
            deep_state = self._init_motion_hidden(batch_size=b,
                                             image_size=(h, w), t_len = input_tensor.shape[1])

        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        head_input = input_tensor
        
        input_deep_h = {}
        input_deep_c = {}
        
        for deep_idx in range(self.num_slices):   

            for layer_idx in range(self.num_layers):

                output_inner = []

                h, c  = hidden_state['%d%d'%(deep_idx,layer_idx)] 
                
                for t in range(seq_len): 

                    if deep_idx == 0:
                        m_h, m_c = deep_state['%d%d'%(layer_idx, t)] 
                    else:
                        m_h = input_deep_h['%d%d%d'%(deep_idx-1,layer_idx, t)]
                        m_c = input_deep_c['%d%d%d'%(deep_idx-1,layer_idx, t)]
                    
                    h, c, m_h, m_c = self.cell_list['%d%d'%(deep_idx,layer_idx)](input_tensor=cur_layer_input[:, t, :, :, :], input_head = head_input[:, t, :, :, :], all_state = head_input, cur_state=[h, c], multi_head=[m_h, m_c]) 
                    
                    output_inner.append(h)

                    input_deep_h.update({'%d%d%d'%(deep_idx,layer_idx,t): m_h}) 
                    input_deep_c.update({'%d%d%d'%(deep_idx,layer_idx,t): m_c}) 

                layer_output = torch.stack(output_inner, dim=1) 
                head_output = torch.stack(([input_deep_h['%d%d%d'%(deep_idx, layer_idx, t)] for t in range (seq_len)]), dim=1)

                cur_layer_input = layer_output
                head_input = head_output 
            
                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    
    def _init_hidden(self, batch_size, image_size):
        
        init_states = {}
        for i in range(0, self.num_slices):
            for j in range(0,self.num_layers):
                init_states.update({'%d%d'%(i,j): self.cell_list['%d%d'%(i,j)].init_hidden(batch_size, image_size)}) 
        return init_states
        
    def _init_motion_hidden(self, batch_size, image_size, t_len):
        
        init_states = {}
        for i in range(0,self.num_layers):
            for j in range(0,t_len):
                init_states.update({'%d%d'%(i,j): self.cell_list['00'].init_hidden(batch_size, image_size)}) 
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    



  
    
if __name__ == "__main__":
    
    from yolo_training import YOLOLoss
    net = Network(num_classes=1, num_frame=5)

    bs = 4
    a = torch.randn(bs, 3, 5, 512, 512)
    out = net(a)
    # for item in out:
    #     print(item.size())
        
    # yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])

    # target = torch.randn([bs, 1, 5]).cuda()
    # target = nn.Softmax()(target)
    # target = [item for item in target]

    # loss = yolo_loss(out, target)
    # print(loss)
