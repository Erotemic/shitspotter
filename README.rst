💩💩💩 ShitSpotter 💩💩💩
=========================

.. .. |CircleCI| |Codecov| |Pypi| |Downloads| |ReadTheDocs|
.. .. +------------------+----------------------------------------------+
.. .. | Read the docs    | https://shitspotter.readthedocs.io           |
.. .. +------------------+----------------------------------------------+
.. .. | Github           | https://github.com/Erotemic/shitspotter      |
.. .. +------------------+----------------------------------------------+
.. .. | Pypi             | https://pypi.org/project/shitspotter         |
.. .. +------------------+----------------------------------------------+


The ``shitspotter`` module is where I will be publishing my work on the "shitspotter" poop-detection algorithm.
The data will be made public as soon as I figure out a hosting situation.




Recent Updates
==============

Check back for updates, but because this is a personal project, it might take
some time for it to fully drop.

* 2021-11-23 - Added annotation process overview and dataset sample.
* 2021-11-11 - Initial upload of data munging scripts.
* 2020-12-18 - Took the first picture.


Introduction
============

In Fall 2019, I was at the local dog park, and I found myself in a situation
where my dog pooped, but I had forgotten to bring bags with me. I walked to the
local bag station (thank you DGS), grabed one, but then I couldn't find where
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
doing that.

Dataset Description
===================

The dataset contains a wide variety of image and background conditions that occur in update New York, including: seasonal changes, snow, rain, daytime, nighttime (some taken with flash, others taken with my phone's *night mode*), grass, concrete, etc...

Known dataset biases are:

* Geographic region: Most images were taken in Upstate New York climate.
* Sensor: Most images were taken with my Pixel 5. A few images were from my old Motorola Droid.
* Coordinate: Humans unconciously center "objects of interest" in images they take. In some instances I tried to mitigate this bias, either by explicitly changing the center of the poop, or not looking at the screen when taking a snapshot.
* Me: I'm the only one taking pictures. I'm also fairly tall, so the images are all from my viewpoint. There are other "me" biases I may not be aware of.
* My Dogs: My two poop machines are fairly regular, and they have their own methods for times and places to make a dookie.
* Freshness: The shit I deal with is often fresh out of the oven. Although, I have picked up a decent number of abandoned stools from other dog owners in the area, some of these are quite old. And age of the sample does seem to have an impact on its appearance. New poops have a shine, while old ones are quite dull, and will start to break down. 

The following scatterplot illustrates trends in the space / time distribution of the images.

.. image:: https://i.imgur.com/LXvcqGW.png

The following figure is a hand-picked sample of 9 images from the dataset. Each of these images has poop in it. In some cases it's easy to spot. In other cases, it can be quite difficult. 

.. image:: https://i.imgur.com/QwFpxD1.jpg

Dataset Statistics:

* Most images only show a single poop, but other images 
* As of 2021-11-11 I've collected 1935 pictures with "798" before/after/(maybe negative) groups of images. There are roughtly 457 paired-groups and 333 triple-groups. (Based only on counts, grouping has not happened yet).


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


Downloading the Data
====================

This dataset will be made public once I figure out a way to host and manage it.
Currently the raw images live on my hard drive, and are backed up across 2 machines, each running RAID-10.
Lower res copies of the photos live on the cloud, but I'm planning on sharing the originals.

The dataset is currently 20+GB, so I'm planning to use IPFS (or some P2P solution) to handle data distribution.
Curently the data does not have any annotations, although I've started to build scripts to make that process
easier. 

Eventually I would like to host the data via DVC + IPFS, but fsspec needs an IPFS filesystem implementation first.

If you are in urgent need of any of the data, feel free to contact me (make an issue or email me).
I can pin what I have on IPFS, but I was planning on organizing the data a bit before I did that,
but I'm willing 

Officially the licence will be Apache 2 because that's what I use for everything.
It annoys me that I have to put licenses on things. These are the terms I care about:

* Cite my work if you use it.
* If you annotate any of the images, contribute the annotations back. Picking up shit is a team effort.
* When asked to build something, particularly ML systems, think about the ethical implications, and act ethically.
* Pin the dataset on IPFS if you can.

Otherwise the data is free to use commercially or otherwise. 


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
