import kwarray
import ubelt as ub
import kwcoco
import kwimage
import kwutil
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.utils.data as torch_data


class KwcocoImageDataset(torch_data.Dataset):
    def __init__(self, coco_dset, processor, text_labels):
        self.coco_dset = coco_dset
        self.processor = processor
        self.text_labels = text_labels

    def __len__(self):
        return self.coco_dset.n_images

    def __getitem__(self, idx):
        image_id = self.coco_dset.dataset['images'][idx]['id']
        coco_img = self.coco_dset.coco_image(image_id)
        full_rgb = coco_img.imdelay().finalize()
        pil_img = Image.fromarray(full_rgb)

        inputs = self.processor(images=pil_img, text=self.text_labels, return_tensors="pt")
        return {
            'inputs': inputs,
            'pil_img_size': pil_img.size[::-1],
            'img_id': coco_img.img['id'],
            'coco_img': coco_img,
        }


def collate_fn(batch):
    # Leave each item as-is, no stacking. Just return as list of dicts.
    return batch


class HuggingFacePredictor:
    def __init__(self, model_id, device, classes, threshold=0.25, text_threshold=0.25, num_workers=4):
        self.device = torch.device(device)
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.num_workers = num_workers

        self.classes: list[str] = kwutil.Yaml.coerce(classes)
        self.text_labels : list[list[str]] = [self.classes]

        print('Loading model')
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def predict_coco(self, coco_dset):
        for c in self.classes:
            coco_dset.ensure_category(c)

        print('Loading dataset')
        dataset = KwcocoImageDataset(
            coco_dset=coco_dset,
            processor=self.processor,
            text_labels=self.text_labels,
        )

        print('Build dataloader')
        loader = torch_data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        print('Run inference:')
        for batch in ub.ProgIter(loader, desc='Running predictions', total=len(dataset)):
            dets = self.predict_item(batch[0])
            new_anns = list(dets.to_coco(image_id=dets.meta['image_id']))
            for ann in new_anns:
                catname = ann['category_name']
                ann['category_id'] = coco_dset.ensure_category(catname)
                coco_dset.add_annotation(**ann)

    def predict_item(self, item):
        inputs = item['inputs'].to(self.device)
        pil_img_size = item['pil_img_size']
        img_id = item['img_id']

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_img_size],
        )

        result = results[0]
        boxes = kwimage.Boxes(result["boxes"], 'ltrb').numpy()
        scores = kwarray.ArrayAPI.numpy(result['scores'])
        result_text_labels = result["text_labels"]

        valid_label_flags = np.array(
            [label in self.classes for label in result_text_labels], dtype=bool)

        # only take results with a label matching the prompt
        boxes = boxes.compress(valid_label_flags)
        scores = scores[valid_label_flags]
        result_text_labels = list(ub.compress(result_text_labels, valid_label_flags))

        classes = kwcoco.CategoryTree.coerce(self.classes)
        class_idxs = np.array([classes.node_to_idx[c] for c in result_text_labels])

        dets = kwimage.Detections(
            boxes=boxes.numpy(),
            scores=kwarray.ArrayAPI.numpy(scores),
            class_idxs=class_idxs,
            classes=classes,
        )
        dets.meta['image_id'] = img_id
        return dets
