import click
import contextlib
import logging
import os
import sys
import argparse
import numpy as np
import torch
import glob
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
import PIL
LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],"component_dataset": ["patchcore.datasets.component_dataset", "Componentdataset"]}

def main(args):
    methods=[]
    patchcore_tuple=patch_core(
    args.backbone_names,
    args.layers_to_extract_from,
    args.pretrain_embed_dimension,
    args.target_embed_dimension,
    args.preprocessing,
    args.aggregation,
    args.patchsize,
    args.patchscore,
    args.patchoverlap,
    args.anomaly_scorer_num_nn,
    args.patchsize_aggregate,
    args.faiss_on_gpu,
    args.faiss_num_workers,
    )

    sample_tuple=sampler(args.sample_name, args.percentage)

    dataset_tuple=dataset(
    args.dataset_name,
    args.data_path,
    args.subdatasets,
    args.train_val_split,
    args.batch_size,
    args.resize,
    args.imagesize,
    args.num_workers,
    args.augment,
    )
    methods.append(patchcore_tuple)
    methods.append(sample_tuple)
    methods.append(dataset_tuple)
    run(
        methods,
        args.results_path,
        args.gpu,
        args.seed,
        args.log_group,
        args.log_project,
        args.save_segmentation_images,
        args.save_patchcore_model,
    )




def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    methods = {key: item for (key, item) in methods}

    #### 运行结果储存路径 ####
    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    #### 定义traindataloder和testdataloader ####
    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize



            #### 定义下采样的方式 ####
            sampler = methods["get_sampler"](
                device,
            )
            #### 执行get_patchcore这个函数，完成整个patchcore的参数设置（比如，backbone,提取的层数，设备，输入尺寸，最后目标embed的维度，patchsize，knn的方法，knn搜索的个数） ####

            #### 往这里加  #####%%%%%%%

            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )

            #### 开始训练 ####
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                #### 计算训练集data的feature embeddings，存入memory bank，并且压缩，转化为coreset ####
                PatchCore.fit(dataloaders["training"])

            ########### 往这里加 ####### %%%%%%
            #### 开始测试 ####
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )

                #### 提取测试集的特征，比较得到异常得分和mask ####
                #### scores(166,) segmentations(166,64,64) ####
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)





            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            #### 对得到的162张图片的每个像素得分再做一次数据处理 结果是（162，224，224）####
            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            #segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            #### 测试集合内的好的照片与异常 ####
            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]





            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()



                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                #### 结果输出，mask是ground_truth,segmentation是计算的全图异常得分（后来转化为热力图） ####
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )





            #### 下面评估像素级别的 ####
            LOGGER.info("Computing evaluation metrics.")

            #### 图像级别的评估，即能否找出此图片是异常，通过输入的异常得分scores与既定打好的标签label ####
            #### 得到fpr，tpr，threshold，最终得到roc曲线，得到auroc得分 ####
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            #### 像素级别的评估，
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?

            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    '''
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )
    '''

def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            #### patchcore的超参数设置，具体见下面的英文 ####
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)

def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', action='store', type=str, required=True)
    parser.add_argument('--gpu', action='store', type=int,default=0,nargs='+', required=True)
    parser.add_argument('--seed', action='store', type=int,default=0, required=True)
    parser.add_argument('--log_group', action='store', type=str,default='group', required=True)
    parser.add_argument('--log_project', action='store', type=str,default='project', required=True)
    parser.add_argument('--save_segmentation_images', action='store_true')
    parser.add_argument('--save_patchcore_model', action='store_true')
    parser.add_argument('--backbone_names', action='store', type=str,nargs='+', required=True)
    parser.add_argument('--layers_to_extract_from', action='store', nargs='+',type=str,default=[], required=True)
    parser.add_argument('--pretrain_embed_dimension', action='store', type=int,default=1024, required=True)
    parser.add_argument('--target_embed_dimension', action='store', type=int,default=1024, required=True)
    parser.add_argument('--preprocessing', action='store', type=str,default='mean', required=True)
    parser.add_argument('--aggregation', action='store', type=str,default='mean', required=True)
    #### n_nearest_neighbours，knn中取最接近的n个得分 ####
    parser.add_argument('--anomaly_scorer_num_nn', action='store', type=int,default=5, required=True)
    parser.add_argument('--patchsize', action='store', type=int,default=3, required=True)
    parser.add_argument('--patchscore', action='store', type=str,default='max', required=True)
    parser.add_argument('--patchoverlap', action='store', type=float,default=0.0, required=True)
    parser.add_argument('--patchsize_aggregate', action='store',nargs='+', type=int,default=[],required=True)
    parser.add_argument('--faiss_on_gpu', action='store_true')
    parser.add_argument('--faiss_num_workers', action='store', type=int,default=8, required=True)
    parser.add_argument('--sample_name', action='store', type=str, required=True)
    parser.add_argument('--dataset_name', action='store', type=str, required=True)

    parser.add_argument('--percentage', action='store', type=float,default=0.1, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--subdatasets', action='store', type=str,nargs='+', required=True)
    #### train_val_split就训练集全是训练集，不用将数据按照比例分割为“测试集”和“训练集” ####
    parser.add_argument('--train_val_split', action='store', type=float, required=True)
    parser.add_argument('--batch_size', action='store', type=int,default=2, required=True)
    parser.add_argument('--num_workers', action='store', type=int,default=8, required=True)
    parser.add_argument('--resize', action='store', type=int,default=256, required=True)
    parser.add_argument('--imagesize', action='store', type=int,default=256, required=True)
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    main(args)
