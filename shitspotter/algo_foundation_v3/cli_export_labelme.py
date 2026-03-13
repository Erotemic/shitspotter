"""
CLI for exporting LabelMe sidecars from prediction kwcoco outputs.
"""

import scriptconfig as scfg
import ubelt as ub

from shitspotter.algo_foundation_v3.kwcoco_adapter import export_predictions_to_labelme


class AlgoExportLabelmeCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='prediction kwcoco path')
    only_missing = scfg.Value(True, help='only create missing sidecars')
    score_thresh = scfg.Value(0.0, help='drop predictions below this score')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        written = export_predictions_to_labelme(
            pred_dataset=ub.Path(config.src).expand(),
            only_missing=config.only_missing,
            score_thresh=float(config.score_thresh),
        )
        print(ub.urepr([str(path) for path in written], nl=1))


__cli__ = AlgoExportLabelmeCLI


if __name__ == '__main__':
    __cli__.main()
