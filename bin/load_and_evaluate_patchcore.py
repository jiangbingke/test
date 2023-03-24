import contextlib
import gc
import logging
import os
import sys
import argparse
import numpy as np
import torch
import glob
import PIL
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],"component_dataset": ["patchcore.datasets.component_dataset", "Componentdataset"]}

def main(args):
    patchcore_tuple=patch_core_loader(args.patch_core_paths, args.faiss_on_gpu, args.faiss_num_workers)

    dataset_tuple=dataset(
    args.dataset_name,
    args.data_path,
    args.subdatasets,
    args.batch_size,
    args.resize,
    args.imagesize,
    args.num_workers,
    args.augment,
    )
    methods=[]
    methods.append(patchcore_tuple)
    methods.append(dataset_tuple)
    run(
        methods,
        args.results_path,
        args.gpu,
        args.seed,
        args.save_segmentation_images,
    )

def run(methods, results_path, gpu, seed, save_segmentation_images):
    methods = {key: item for (key, item) in methods}

    os.makedirs(results_path, exist_ok=True)

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

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict_instance(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

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
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # Plot Example Images.
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

                patchcore.utils.plot_segmentation_images(
                    results_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            # Compute Image-level AUROC scores for all images.
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only for images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs], [masks_gt[i] for i in sel_idxs]
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

            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        patch_core_paths2=glob.glob(patch_core_paths[0]+'*')
        for patch_core_path in patch_core_paths2:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])

def dataset(
    name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


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
    parser.add_argument('--train_val_split', action='store', type=float, required=True)
    parser.add_argument('--batch_size', action='store', type=int,default=2, required=True)
    parser.add_argument('--num_workers', action='store', type=int,default=8, required=True)
    parser.add_argument('--resize', action='store', type=int,default=256, required=True)
    parser.add_argument('--imagesize', action='store', type=int,default=256, required=True)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--patch_core_paths', action='store', type=str, nargs='+',required=True)


    args = parser.parse_args()

    main(args)
