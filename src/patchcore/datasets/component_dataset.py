import os
from enum import Enum
import pickle
import PIL
import torch
from torchvision import transforms
import glob

_CLASSNAMES = [
    "instance_0",
    "instance_1",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class Componentdataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            classname,
            instancename,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TRAIN,
            train_val_split=1.0,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.instancemode=True
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.instancename=instancename
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.instance_per_class,self.instance_to_iterate=self.get_instance_data()
        self.imagesize = (3, imagesize, imagesize)
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.resize=resize
        self.transform_img = [
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
        ]

        self.transform_mask = transforms.Compose(self.transform_mask)

    def __getitem__(self, idx):
        if not self.instancemode:
            classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.transform_img(image)

            if self.split == DatasetSplit.TEST and mask_path is not None:
                mask = PIL.Image.open(mask_path)
                mask = self.transform_mask(mask)
            else:
                mask = torch.zeros([1, *image.size()[1:]])

            return {
                "image": image,
                "mask": mask,
                "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
                "image_name": "/".join(image_path.split("/")[-4:]),
                "image_path": image_path,
            }
        else:
            classname, anomaly, instance_path, mask_path = self.instance_to_iterate[idx]
            filepath=sorted(glob.glob(instance_path+'/*'))
            imgdict={}
            imgdict['bg'] = PIL.Image.open(filepath[0]).convert("RGB")
            for i in range(1,len(filepath)-1):
                imgpaths=sorted(glob.glob(filepath[i]+'/*'))
                imglist=[]
                instancenames=filepath[i].split('/')
                instancename=instancenames[len(instancenames)-1]
                for imgpath in imgpaths:
                    img=PIL.Image.open(imgpath).convert("RGB")
                    imglist.append(img)
                imgdict[instancename]=imglist
            c=filepath[len(filepath)-1]
            f_read=open(filepath[len(filepath)-1],'rb')
            relation=pickle.load(f_read)
            f_read.close()
            if self.split == DatasetSplit.TEST and mask_path is not None:
                mask = PIL.Image.open(mask_path)
                mask = self.transform_mask(mask)
            else:
                mask = torch.zeros([1, imgdict['bg'].size()[1:]])

            return {
                "image": imgdict,
                "relation":relation,
                "mask": mask,
                "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
            }

            pass



    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        bg_imgpaths_per_class={}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            bg_imgpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                if self.instancename != 'bg':
                    imgpaths_per_class[classname][anomaly] = sorted(glob.glob(anomaly_path+f'/*/{self.instancename}/*.jpg'))
                ##background
                else:
                    imgpaths_per_class[classname][anomaly] = sorted(glob.glob(anomaly_path+'/*/*.jpg'))

                '''
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                '''
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                    ##locodataset

                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x + '/000.png') for x in anomaly_mask_files
                    ]

                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        #data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        data_tuple.append(None)
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def get_instance_data(self):
        instance_per_class = {}
        maskpaths_per_class = {}
        bg_imgpaths_per_class={}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            instance_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            bg_imgpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                instance_per_class[classname][anomaly] = sorted(glob.glob(anomaly_path+'/*'))

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    ##locodataset
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x + '/000.png') for x in anomaly_mask_files
                    ]

                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(instance_per_class.keys()):
            for anomaly in sorted(instance_per_class[classname].keys()):
                for i, image_path in enumerate(instance_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return instance_per_class, data_to_iterate