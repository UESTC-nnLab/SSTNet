#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.Network import Network

if __name__ == "__main__":
    input_shape = [512, 512]
    num_classes = 1
    phi         = 'l'
    
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = Network(num_classes, num_frame=5).to(device)
    summary(m, (3, 5,input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, 5,input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    # 计算FPS
    
    from utils.dataloader_for_DAUB import seqDataset, dataset_collate
    from torch.utils.data import DataLoader
    import time
    max_iter = 1000
    log_interval = 50
    num_warmup = 5
    pure_inf_time = 0
    fps = 0
    val_annotation_path = 'coco_val_DAUB.txt'
    val_dataset = seqDataset(val_annotation_path, input_shape[0], 5, 'val')
    gen_val     = DataLoader(val_dataset, shuffle = False, batch_size = 1, num_workers = 2, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate)
    # benchmark with 2000 image and take the average
    for i, data in enumerate(gen_val):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            Network(num_classes, num_frame=5).to('cuda')(data[0].to('cuda'))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    print("FPS:" ,fps)
