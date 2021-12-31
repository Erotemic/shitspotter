"""
Notes:
    The first generation of IPFS addresses

    QmWhKBAQ765YH2LKMQapWp7mULkQxExrjQKeRAWNu5mfBK shitspotter_dvc/data.kwcoco.json
    QmfYfUFRZivaJs3eyKM3x7kWGKRkwoCKU3FgeEr9ccVvy8 shitspotter_dvc/_cache
    QmXdQzqcFv3pky621txT5Z6k41gZR9bkckG4no6DNh2ods shitspotter_dvc/assets/_poop-unstructured-2021-02-06
    QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn shitspotter_dvc/assets/_trashed
    QmZ4vipXwH7f27VSjx3Bz4aLoeigL9T22sFADv5KCBTFW7 shitspotter_dvc/assets/poop-2020-12-28
    QmTHipghcRCVamWLojWKQy8KgamtRnPv9fL3dxxPv7VVZx shitspotter_dvc/assets/poop-2021-02-06
    QmZ3W4pXVkbhQKssWBhBgspeAB3U6GRGD85eff7BvAPNri shitspotter_dvc/assets/poop-2021-03-05
    QmZb6s53W34rmUJ2s5diw4ErhK3aLb5Td9MtML4u5wqMT5 shitspotter_dvc/assets/poop-2021-04-06
    QmbZrgM4jCJ8ccU9DLGewPkVBDH6pDVs4vdUUk1jeKyfic shitspotter_dvc/assets/poop-2021-04-19
    QmTexn6vX8vtAYiZYDq2YmHjoUnnJAAxEtyFPwXsqfvpKy shitspotter_dvc/assets/poop-2021-04-25
    QmXFyYBVqVVcKqcJuGzo3d9WTRxf4U4cZBmRaT6q52mqLp shitspotter_dvc/assets/poop-2021-05-11T000000
    QmcTkxhsA4QsWb9KJsLKGnWNyhf7SuMNhAmf55DiXqG8iU shitspotter_dvc/assets/poop-2021-05-11T150000
    QmNVZ6BGbTWd5Tw5s4E3PagzEcvp1ekxxQL6bRSHabEsG3 shitspotter_dvc/assets/poop-2021-06-05
    QmQAbQTbTquTyMmd27oLunS3Sw2rZvJH5p7zus4h1fvxdz shitspotter_dvc/assets/poop-2021-06-20
    QmRkCQkAjYFoCS4cEyiDNnk9RbcoQPafmZvoP3GrpVzJ8D shitspotter_dvc/assets/poop-2021-09-20
    QmYYUdAPYQGTg67cyRWA52yFgDAWhHDsEQX9yqED3tj4ZX shitspotter_dvc/assets/poop-2021-11-11
    QmYXXjAutQLdq644rsugp6jxPH6GSaP3kKRTC2jsy4FQMp shitspotter_dvc/assets/poop-2021-11-26
    QmQAufuJGGn7TDeiEE52k5SLPGrcrawjrd8S2AATrSSBvM shitspotter_dvc/assets/poop-2021-12-27
    QmfZZwoj1gwGPctBQW5Mkye3a8VuajFBCksHVJH7r9Wn3U shitspotter_dvc/assets
    QmRGxbcjYb7ndCzZ4fEBBk2ZR7MtU43f4SSDEeZp9vonx9 shitspotter_dvc

"""


def find_shit_coco_fpath(on_error='raise'):
    """
    Return the location of the shitspotter kwcoco file if it exists and is in a
    "standard" location.

    This assumes

    Ignore:
        ln -s /data/store/data/shit-pics/ $HOME/data/dvc-repos/shitspotter_dvc
        ln -s /data/data/shit-pics/ $HOME/data/dvc-repos/shitspotter_dvc
    """
    import ubelt as ub
    import os
    _default = ub.expandpath('$HOME/data/dvc-repos/shitspotter_dvc')
    dvc_dpath = os.environ.get('SHITSPOTTER_DVC_DPATH', _default)
    dvc_dpath = ub.Path(dvc_dpath)
    coco_fpath = dvc_dpath / 'data.kwcoco.json'

    if not coco_fpath.exists():
        if on_error == 'raise':
            raise Exception
        else:
            return None
    return coco_fpath


def open_shit_coco():
    """
    Shortcut to get "the" shitspotter dataset
    """
    import kwcoco
    coco_fpath = find_shit_coco_fpath()
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    return coco_dset
