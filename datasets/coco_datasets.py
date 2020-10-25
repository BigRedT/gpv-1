import os
import hydra
import numpy as np
import skimage.io as skio
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.detr_misc import collate_fn as detr_collate_fn
from .coco_generic_dataset import GenericCocoDataset


class CocoCaptioning(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)

class CocoDetection(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)

class CocoVqa(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)

class CocoClassification(GenericCocoDataset):
    def __init__(self,cfg,subset):
        super().__init__(cfg,subset)
    
    def read_image(self,image_subset,image_id,x1,y1,x2,y2):
        img_dir = os.path.join(self.cfg.image_dir,image_subset)
        img_path = os.path.join(
            img_dir,
            'COCO_'+image_subset+'_'+str(image_id).zfill(12)+'.jpg')
        img = skio.imread(img_path)[:,:,:3]
        H,W = img.shape[:2]
        x1,x2 = [min(max(0,int(z)),W) for z in [x1,x2]]
        y1,y2 = [min(max(0,int(z)),H) for z in [y1,y2]]
        img = img[y1:y2,x1:x2]
        original_image_size = img.shape[:2] # HxW
        resized_image_size = resize(img,(self.imh,self.imw),anti_aliasing=True)
        return resized_image_size, original_image_size
    
    def __getitem__(self,i):
        sample = self.samples[i]

        image_subset = sample['image']['subset']
        image_id = sample['image']['image_id']
        x,y,w,h = sample['boxes']
        img, original_image_size = self.read_image(
            image_subset,image_id,x,y,x+w,y+h)
        img = img.astype(np.float32)
        img = (img - 0.5)/0.25
        img = torch.as_tensor(img,dtype=torch.float32).permute(2,0,1)
        
        query = sample['query']

        targets = {'answer': sample['answer']}

        return img, query, targets

@hydra.main(config_path="../configs",config_name="test/coco_datasets")
def test_dataset(cfg):
    print(cfg.pretty())
    task_config_name = cfg.learning_datasets[cfg.dataset_to_test].task_config
    dataset = globals()[cfg.dataset_to_test](
        cfg.task_configs[task_config_name],'train')
    dataloader = dataset.get_dataloader(batch_size=2)
    for data in dataloader:
        import ipdb; ipdb.set_trace()


if __name__=='__main__':
    test_dataset()