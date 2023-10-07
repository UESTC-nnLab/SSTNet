import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import time
import torch

# convert to RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    

# normalization
def preprocess(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

def augmentation(images, boxes,h, w, hue=.1, sat=0.7, val=0.4):
    # images [5, w, h, 3], bbox [:,4]
    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    filp = rand()<.5
    if filp:
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0,2]] = w - boxes[i][[2,0]]

    images      = np.array(images, np.uint8)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    for i in range(len(images)):
        hue, sat, val   = cv2.split(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
        dtype           = images[i].dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        images[i] = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)

    return np.array(images,dtype=np.float32), np.array(boxes,dtype=np.float32)



class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5 ,type='train'):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.txt_path = dataset_path
            self.aug = True
        else:
            self.txt_path = dataset_path
            self.aug = False
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                self.anno_idx.append(np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.transpose(preprocess(images),(3, 0, 1, 2))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        return images, box
    
    def get_data(self, index):
        image_data = []
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        label_data = self.anno_idx[index]  # 4+1
        for id in range(0, self.num_frame):
            img = Image.open(image_path +'%d.bmp' % max(image_id - id, 0)) 
            img = cvtColor(img)
            iw, ih = img.size
            
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            
            img = img.resize((nw, nh), Image.BICUBIC)  # 原图等比列缩放
            new_img = Image.new('RGB', (w,h), (128, 128, 128))  # 预期大小的灰色图
            new_img.paste(img, (dx, dy))  # 缩放图片放在正中
            image_data.append(np.array(new_img, np.float32))
            
            if len(label_data) > 0 and id == 0:
                np.random.shuffle(label_data)
                label_data[:, [0, 2]] = label_data[:, [0, 2]]*nw/iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]]*nh/ih + dy
                
                label_data[:, 0:2][label_data[:, 0:2]<0] = 0
                label_data[:, 2][label_data[:, 2]>w] = w
                label_data[:, 3][label_data[:, 3]>h] = h
                # discard invalid box
                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                label_data = label_data[np.logical_and(box_w>1, box_h>1)] 

                    
        image_data = np.array(image_data[::-1]) # 关键帧在后 # [5,w,h,3]
        label_data = np.array(label_data, dtype=np.float32) # [:,5]
        if self.aug is True:
            # image_data, label_data[:,:4] = augmentation(image_data,label_data[:,:4],h,w)
            pass
        return image_data, label_data
                    
def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

                
            
    
if __name__ == "__main__":
    train_dataset = seqDataset("/home/coco_val_IRDST.txt", 512, 5, 'test')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=dataset_collate)
    t = time.time()
    for index, batch in enumerate(train_dataloader):
        images, targets = batch[0], batch[1]
        print(index)
    print(time.time()-t)
    