🗑️📱💩 ShitSpotter 💩📱🗑️
=========================

.. 💩📱📷🤏🗑️🤌

.. .. |CircleCI| |Codecov| |Pypi| |Downloads| |ReadTheDocs|
.. .. +------------------+----------------------------------------------+
.. .. | Read the docs    | https://shitspotter.readthedocs.io           |
.. .. +------------------+----------------------------------------------+
.. .. | Github           | https://github.com/Erotemic/shitspotter      |
.. .. +------------------+----------------------------------------------+
.. .. | Pypi             | https://pypi.org/project/shitspotter         |
.. .. +------------------+----------------------------------------------+


The ``shitspotter`` module is where I will be work on the "shitspotter" poop-detection algorithm and dataset.
The primary goal of this work is to allow for the creation of a phone app that finds where your dog pooped,
because you ran to grab the doggy-bags you forgot, and now you can't find the damn thing.
Other applications can be envisioned, such as AR glasses that lets you know if you are about to walk into a steamer, 
or perhaps city governments could use this to more efficiently clean public areas. 

This module will contain an algorithm for training a pytorch network to detect poop in images, and a script
for detecting poop in unseen images given a pretrained model. 

The dataset currently contains 20GB of outdoor images taken with a phone. The general process of acquiring the dataset has been: 
1. My dog poops or I see a rogue poop, 
2. I take a "before" picture of the poop,
3. I pick up the poop, 
4. I take an "after" picture as a high-correlation negative, and 
5. I take a 3rd image of a different nearby area to get a lower-correlation-negative. 
The dataset is currently unannotated, but but before/after pairs will help with bootstrapping. 
Annotations and the data manifest will be managed using kwcoco.

Both the code and the dataset will be open source. 
The code will be published as it is written to this repo. 
The data and pretrained models will be made public on IPFS.


Recent Updates
==============

Check back for updates, but because this is a personal project, it might take
some time for it to fully drop.

* 2022-03-13 - Added more images and updated analysis
* 2021-12-30 - 
    - Found errors in the dataset stats, updating README.
    - Updated analytics to be updated as the dataset grows. 
    - Initial SIFT-based matching isnt as robust as I'd hoped.
    - First data is on IPFS, still need to open ports. ID of the root dataset is: QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG
* 2021-11-23 - Added annotation process overview and dataset sample.
* 2021-11-11 - Initial upload of data munging scripts.
* 2020-12-18 - Took the first picture.


Introduction
============

In Fall 2019, I was at the local dog park, and I found myself in a situation
where my dog pooped, but I had forgotten to bring bags with me. I walked to the
local bag station (thank you DGS), grabbed one, but then I couldn't find where
the poop was. The brown fallen leaves made it very difficult to find the poop.

This happened every so often. Often I would be able to find it, but I'm afraid
sometimes, I was unable to relocate the "object of interest". This got me
thinking, what if I had a phone app that could scan the area with the camera
and try to locate the poop? If I had a dataset, training a poop detection model
with today's deep learning methods should work pretty well.

Thus, on 2020-12-18, I took my first picture. My dog pooped, I took a picture,
I picked it up, and then I took an "after" picture. The idea is that I will
align the pictures (probably via computing local features like sift or some
deep variant and then estimating an affine/projective transform) and then take
a difference image. That should let me seed some sort of semi-automated
annotation process.

Then in 2021-05-11, one of my colleague suggested that I take a 3rd unrelated
picture to use as negative examples, so I took that suggestion and started
doing that. This is the process currently being used. The following figure
illustrates an example of one of these "triples".

.. image:: https://i.imgur.com/NnEC8XZ.jpg

Related Work
============

I was surprised to find that there does not seem to be much work on this problem in the outdoor setting.
Because none of the related work exactly meets my needs, I haven't looked too in depth into much of it,
it could be that some of these are more relevant than I've given them credit for. As time moves on
I'll continue to refine this section.

Apparently Roomba has an indoor poop dataset: https://www.engadget.com/irobot-roomba-j-7-object-poop-detection-040152887.html It would be interesting to combine the indoor / outdoor datasets, but we are more concerned about outdoor detection. Maybe Boston Dynamics and Roomba can take this dataset and do something interesting.

The MSHIT fake dog poop dataset: https://www.kaggle.com/mikian/dog-poop is similar to this domain, but not the real-deal. 
This may be relevant, but I have not looked too deeply into it yet.

There is Human Poop Classification: https://seed.com/poop/ and https://www.theverge.com/2019/10/29/20937108/poop-database-ai-training-photo-upload-first-mit but this is not our domain.

Detect Images of Dogs Pooping: https://colab.research.google.com/github/matthewchung74/blogs/blob/dev/Dog_Pooping_Dectron.ipynb 
Unfortunately, this is detecting the action, and not the consequence.

A Dog Poop DNA database could be used in conjunction with this work: https://www.bbc.com/news/uk-england-somerset-56324906

A 2019 Project by Neeraj Madan: https://www.youtube.com/watch?v=qGNbHwp0jM8 
This is the most similar thing to this project that I've seen so far. I have
not watched his entire video yet, but I may contact him so see if they're
interested in collaborating.

TACO: http://tacodataset.org/ 
The TACO dataset is Trash Annotations in Context. It could be the case that this data could be incorporated into the TACO dataset, although it does not currently contain a category for feces.

Other related links I haven't gone through well enough yet:

* https://getdiglabs.com/blogs/the-dig-labs-dish/computer-vision-and-dog-poop
* https://www.wired.co.uk/article/dog-poo-bin-cleanup
* https://www.reddit.com/r/robotics/comments/6p0rf0/can_i_use_opencv_to_get_my_robot_to_detect_dog/
* https://www.housebeautiful.com/lifestyle/kids-pets/a31289426/robot-picks-up-dog-poop/



Dataset Description
===================

The dataset contains a wide variety of image and background conditions that occur in update New York, including: seasonal changes, snow, rain, daytime, nighttime (some taken with flash, others taken with my phone's *night mode*), grass, concrete, etc...

Known dataset biases are:

* Geographic region: Most images were taken in Upstate New York climate.
* Sensor: Most images were taken with my Pixel 5. A few images were from my old Motorola Droid.
* Coordinate: Humans unconsciously center "objects of interest" in images they take. In some instances I tried to mitigate this bias, either by explicitly changing the center of the poop, or not looking at the screen when taking a snapshot.
* Me: I'm the only one taking pictures. I'm also fairly tall, so the images are all from my viewpoint. There are other "me" biases I may not be aware of.
* My Dogs: My two poop machines are fairly regular, and they have their own methods for times and places to make a dookie.
* Freshness: The shit I deal with is often fresh out of the oven. Although, I have picked up a decent number of abandoned stools from other dog owners in the area, some of these are quite old. And age of the sample does seem to have an impact on its appearance. New poops have a shine, while old ones are quite dull, and will start to break down. 

The following scatterplot illustrates trends in the space / time distribution of the images.

.. https://ipfs.io/ipfs/QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6
.. https://ipfs.io/ipfs/QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6/analysis/scat_scatterplot.png

.. image:: https://imgur.com/beP2PdK.png


A visualization of the cumulative number of images collected over time is as follows:

.. image:: https://i.imgur.com/1gL6EuI.png
   

The following figure is a hand-picked sample of 9 images from the dataset. Each of these images has poop in it. In some cases it's easy to spot. In other cases, it can be quite difficult. 

.. image:: https://i.imgur.com/QwFpxD1.jpg

Dataset Statistics:

* Most images only show a single poop, but other images have multiple.

 
### As of 2021-11-11 

(The counts for this date are wrong)

* I've collected 1935 pictures with "616" before/after/(maybe negative) groups of images.
* There are roughly 394 paired-groups and 222 triple-groups. (Based only on counts, grouping has not happened yet).

### As of 2021-12-30 

(These are more correct)

* As of 2021-12-30 I've collected 2088 pictures with "~728" before/after/(maybe negative) groups of images. (number of pairs is approximate, dataset not fully registered yet)
* There are roughly 394 paired-groups and 334 triple-groups. (Based only on counts, grouping has not happened yet).


### As of 2022-03-14 

* As of 2021-12-30 I've collected 2471 pictures with "~954" before/after/(maybe negative) groups of images. (number of pairs is approximate, dataset not fully registered yet)
* There are roughly 394 paired-groups and 560 triple-groups. (Based only on counts, grouping has not happened yet, there are 658 groups where the before / after images have been reported as registered by the matching algorithm).


Annotation Process
==================

To make annotation easier, I've taken before a picture before and after I clean up the poop. 
The idea is that I can align these images and use image-differencing to more quickly find the objects of interest in the image.
As you can see, it's not so easy to spot the shit, especially when there are leaves in the image.

.. image:: https://i.imgur.com/lZ8J0vD.png

But with a little patience and image processing, it's not to hard to narrow down the search.

.. image:: https://i.imgur.com/A6qlcNk.jpg

Scripts to produce these visualizations have been checked into the repo. Annotations and the image manifest will
be stored in the kwcoco json format.

The Algorithm
=============

Currently there is no algorithm checked into the repo. I need to start annotating the dataset first. 
Eventually there will be a `shitspotter.fit` and `shitspotter.predict` script for training and performing
inference on unseen images. My current plan for a baseline algorithm is a mobilenet backbone pretrained 
on imagenet and some single-stage detection / segmentation head on top of that.

Given kwcoco a formated detection dataset, we can also use off-the-shelf detection baselines
via netharn, mmdet, or some other library that accepts coco/kwcoco input manifests.


Downloading the Data
====================


This dataset will be made public once I figure out a way to host and manage it.
Currently the raw images live on my hard drive, and are backed up across 2 machines, each running RAID-10.
Lower res copies of the photos live on the cloud, but I'm planning on sharing the originals.

The dataset is currently 17G+GB, and is currently hosted on IPFS.  Currently
the data does not have any annotations, although I've started to build scripts
to make that process easier. 

Eventually I would like to host the data via DVC + IPFS, but fsspec needs an IPFS filesystem implementation first.
I may also look into git-annex as an alternative to DVC.

The licence for the software will be Apache 2. The license for the data will be
"Creative Commons Attribution 4.0 International".

In addition to these licenses please:

* Cite my work if you use it.
* If you annotate any of the images, contribute the annotations back. Picking up shit is a team effort.
* When asked to build something, particularly ML systems, think about the ethical implications, and act ethically.
* Pin the dataset on IPFS if you can.

Otherwise the data is free to use commercially or otherwise. 

Update 2022-02-31: Updated root CID: QmaPPoPs7wXXkBgJeffVm49rd63ZtZw5GrhvQQbYrUbrYL

Update 2021-12-30: Initial root CID: QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG

Update 2022-03-13: Initial root CID: QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6

The URL that can be viewed in a web browser: https://ipfs.io/ipfs/QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG 

IPFS addresses for the top-level dataset filesystem are:
.. code:: 

    QmRYp7JmvTwbHyCojYtxWYi3PrhjzT2LbKti5aBBw22Gpy shitspotter_dvc/data.kwcoco.json
    QmVArot1i19A7AnrQmXQHHkPoAuxRMQor4nSaJZ31WK3WG shitspotter_dvc/_cache
    QmdviphpTehw6nWBAFwAcd7L8AA9AMCAXoor7v7kcZx6wZ shitspotter_dvc/analysis
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
    QmZmJcYPPakvB4cYxjDWWt9Kq1pSgyYLHXK9b5h4wG7LSD shitspotter_dvc/assets/poop-2022-01-27
    QmSmMKCNtMTj4EVUfKzWfBKwwztSsDZhsjGGx4T17jtzfV shitspotter_dvc/assets/poop-2022-03-13-T152627
    QmZ4rxiP445CDWSuiXY2Tq4WfPeB48HzemHDwg1MX3gDqX shitspotter_dvc/assets
    QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6 shitspotter_dvc


Depsite the name, this is not yet a DVC repo.


Acknowledgements
================

I want to give thanks to the people and animals-that-think-they-are-people who
contributed to this project. My colleagues at Kitware have provided valuable
help / insight into project direction, dataset collection, problem formulation,
related research, discussion, and memes.

I want to give special thanks to my two poop machines, without whom this project would not be possible.

.. image:: https://i.imgur.com/MWQVs0w.jpg

.. image:: https://i.imgur.com/YUJjWoh.jpg

.. |Pypi| image:: https://img.shields.io/pypi/v/shitspotter.svg
   :target: https://pypi.python.org/pypi/shitspotter

.. |Downloads| image:: https://img.shields.io/pypi/dm/shitspotter.svg
   :target: https://pypistats.org/packages/shitspotter

.. |ReadTheDocs| image:: https://readthedocs.org/projects/shitspotter/badge/?version=release
    :target: https://shitspotter.readthedocs.io/en/release/

.. # See: https://ci.appveyor.com/project/jon.crall/shitspotter/settings/badges
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/master?svg=true
   :target: https://ci.appveyor.com/project/jon.crall/shitspotter/branch/master

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/utils/shitspotter/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/utils/shitspotter/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/utils/shitspotter/badges/master/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/utils/shitspotter/commits/master

.. |CircleCI| image:: https://circleci.com/gh/Erotemic/shitspotter.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/shitspotter

.. |Travis| image:: https://img.shields.io/travis/Erotemic/shitspotter/master.svg?label=Travis%20CI
   :target: https://travis-ci.org/Erotemic/shitspotter

.. |Codecov| image:: https://codecov.io/github/Erotemic/shitspotter/badge.svg?branch=master&service=github
   :target: https://codecov.io/github/Erotemic/shitspotter?branch=master
