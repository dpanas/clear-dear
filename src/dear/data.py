import os
import torch

import numpy as np
import pandas as pd

from PIL import Image
from torchvision import datasets, transforms


def _transform_pipeline( image_size, normalize, add_flips, crop_center= False):
    """
    Prep the pipeline needed, depending on the requirements, at minimum ToTensor, but also 
    optionally normalize, resize, augment.
    """
    trans_list = []
    if crop_center:
        trans_list.append( transforms.CenterCrop())
    if add_flips:
        trans_list.extend( [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    # always make sure dimensions are right and it is put to tensor
    # NOTE: the order of Resize and ToTensor matters! interpolation on int vs float
    trans_list.extend( [ transforms.Resize( (image_size, image_size)), transforms.ToTensor()])
    if normalize:
        trans_list.append( transforms.Normalize( ( 0.5, 0.5, 0.5), ( 0.5, 0.5, 0.5)))
    return transforms.Compose( trans_list)
    
def load_labels( label_path= None, labels_index= None, images_dir= None):
    if label_path is not None:
        df = pd.read_csv( label_path)
        print( f'Loaded labels from {label_path}, total of {len(df)} entries.')
        im_names = df['image'].values
        df_labels = df.drop('image',axis=1)
        if labels_index is None:
            labels_index_ = list( range( len( df_labels.columns)))
        elif type( labels_index[0]) == str:
            labels_index_ = [i for i,c in enumerate( df_labels.columns) if c in labels_index]
        else:
            labels_index_ = labels_index
        print( f'First entry: {im_names[0]}, with labels {df_labels.values[:,labels_index_][0,:]}')
        return im_names, df_labels.values[:, labels_index_]
    else:
        im_names = os.listdir( images_dir)
        print( f'Listed images in {im_names}, total of {len(im_names)}, assuming no labels.')
        return im_names, - np.ones( (len(im_names),1))

def make_dataloader(args, test_flag= False):

    test_loader = None
    train_loader = None
    if args.dataset == 'celeba':
        trans_f = _transform_pipeline( 
            args.image_size, normalize= True, add_flips= False, crop_center= True
        )
        train_set = datasets.CelebA(
            args.data_dir, split='train', download=False, transform= trans_f
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True, num_workers=4
        )
    
    else:
        train_set = DatasetLabelled( 
            args.data_dir, args.image_size, args.label_file, args.label_idx, 
            args.sup_prop, args.normalize, args.add_flips
        )
        if not test_flag:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size= args.batch_size, shuffle= True, drop_last= True, num_workers= 4
            )
        else:
            # for test we do not need shuffling or dropping:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size= args.batch_size, num_workers= 4
            )
            

    return train_loader, test_loader


def get_batch_no( data_loader, batch_no= 0, device= torch.device('cpu')):    

    for batch_idx, (x, label, im_name) in enumerate(data_loader):
        x = x.to(device)
        
        sup_flag = label[:, 0] != -1
        if sup_flag.sum() > 0:
            label_ = label[sup_flag, :].float()

        label = label.to(device)
        if batch_idx==batch_no:
            break
    print( f'In batch {batch_idx} there are {len(label_)} supervised.')
    return x, label, im_name

class DatasetLabelled(torch.utils.data.Dataset):
    """
    Inherit torch Dataset, but return not just image but also the label + optionally path.
    NOTE: expected that in `data_dir` there is at least `images/` subdirectory, optionally a labels
    """
    def __init__( 
        self, data_dir, image_size, label_file, labels_index, sup_prop=1., normalize= True, add_flips= False
    ):
        self.images_dir = os.path.join( data_dir, 'images')
        label_path = label_file
        if label_file is not None:
            if not os.path.exists( label_file):
                label_path = os.path.join( data_dir, label_file)
        #label_path = os.path.join( data_dir, label_file) if label_file is not None else label_file
        self.image_names, self.labels = load_labels( label_path, labels_index)                          
        self.transforms = _transform_pipeline( image_size, normalize, add_flips)
                          
        np.random.seed(2)
        self.n = len(self.image_names)
        self.available_label_index = np.random.choice(self.n, int(self.n * sup_prop), replace=0)
        
    def __getitem__(self, idx):        
        im_name = self.image_names[ idx]
        image = Image.open( os.path.join( self.images_dir, im_name)).convert('RGB')
        image_tensor = self.transforms( image)
        label_tensor = torch.tensor( self.labels[ idx].astype(float))
        if idx not in self.available_label_index:
            label_tensor = torch.zeros_like( label_tensor) -1        
        return image_tensor, label_tensor, im_name

    def __len__(self):
        return len(self.image_names)
