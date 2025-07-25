#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class PredictGroundingDinoCLI(scfg.DataConfig):
    src = scfg.Value(None, help='input kwcoco dataset')
    dst = scfg.Value(None, help='output kwcoco dataset')
    model_id = scfg.Value("IDEA-Research/grounding-dino-tiny", help='huggingface model ID')
    device = scfg.Value('cuda:0', help='a torch device string or number')
    classes = scfg.Value('[foreground object]', help='A YAML list of text prompts')
    threshold = scfg.Value(0.25, help='Threshold to keep object detection predictions based on confidence score')
    text_threshold = scfg.Value(0.25, help='Score threshold to keep text detection predictions')
    num_workers = scfg.Value(4, help='Number of workers for dataloader')
    force_classname = scfg.Value(None, help='if specified, rename the classes to this for consistency over prompts')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> from shitspotter.other.predict_grounding_dino import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('shitspotter/tests/grounding_dino').ensuredir()
            >>> kwargs['src'] = dset.fpath
            >>> kwargs['dst'] = dpath / 'pred.kwcoco.zip'
            >>> kwargs['num_workers'] = 0
            >>> kwargs['device'] = 'cpu'
            >>> kwargs['classes'] = ['star', 'superstar', 'eff']
            >>> cls = PredictGroundingDinoCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        import kwcoco
        ub.Path(config.dst).parent.ensuredir()

        print('Reading dataset')
        src_dset = kwcoco.CocoDataset.coerce(config.src)

        dst_dset = src_dset.copy()
        dset = dst_dset
        dset.clear_annotations()
        dset.remove_categories(dset.categories())

        dset.reroot(absolute=True)
        dset._update_fpath(config.dst)

        import kwutil
        proc_context = kwutil.ProcessContext(
            name='shitspotter.other.predict_grounding_dino',
            config=kwutil.Json.ensure_serializable(dict(config)),
            track_emissions=True,
        )
        proc_context.start()
        print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

        if 0:
            old_run(config, dset)
        else:
            print('Importing HuggingFacePredictor')
            from shitspotter.other.coco_predict_grounding_dino import HuggingFacePredictor
            predictor = HuggingFacePredictor(
                model_id=config.model_id,
                device=config.device,
                classes=config.classes,
                threshold=config.threshold,
                text_threshold=config.text_threshold,
                num_workers=config.num_workers,
            )
            predictor.predict_coco(dset)

            if config.force_classname is not None:
                mapper = {catname: config.force_classname for catname in predictor.classes}
                dset.rename_categories(mapper)

        dset.dataset.setdefault('info', [])
        proc_context.stop()
        print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')
        dset.dataset['info'].append(proc_context.obj)
        dset.dump()
        import rich
        bundle_dpath = ub.Path(dset.fpath).parent.ensuredir()
        rich.print(f'Wrote to: [link={bundle_dpath}]{bundle_dpath}[/link]')


def old_run(config, dset):
    import kwarray
    import kwcoco
    import kwimage
    import kwutil
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = config.model_id
    device = torch.device(config.device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(device)

    classes: list[str] = kwutil.Yaml.coerce(config.classes)
    text_labels : list[list[str]] = [classes]

    for c in classes:
        dset.ensure_category(c)

    # needs more efficient loading
    for coco_img in ub.ProgIter(dset.images().coco_images, desc='predict images'):
        full_rgb = coco_img.imdelay().finalize()
        pil_img = Image.fromarray(full_rgb)

        inputs = processor(images=pil_img, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=config.threshold,
            text_threshold=config.text_threshold,
            target_sizes=[pil_img.size[::-1]]
        )
        assert len(results) == 1
        result = results[0]

        boxes = kwimage.Boxes(result["boxes"], 'ltrb')
        classes = kwcoco.CategoryTree.coerce(sorted(set(result["text_labels"])))
        class_idxs = np.array([classes.node_to_idx[c] for c in result["text_labels"]])

        dets = kwimage.Detections(
            boxes=boxes.numpy(),
            scores=kwarray.ArrayAPI.numpy(result['scores']),
            class_idxs=class_idxs,
            classes=classes,
        )

        new_anns = list(dets.to_coco(image_id=coco_img.img['id']))
        for ann in new_anns:
            ann['category_id'] = dset.ensure_category(ann['category_name'])
            dset.add_annotation(**ann)

    print(f'dset.cats={dset.cats}')
    dset.dump()

__cli__ = PredictGroundingDinoCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/other/predict_grounding_dino.py
        python -m shitspotter.other.predict_grounding_dino
    """
    __cli__.main()
