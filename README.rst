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


This ``shitspotter`` repo is where I am building the "shitspotter" poop-detection algorithm and dataset.
The primary goal of this work is to allow for the creation of a phone app that finds where your dog pooped,
because you ran to grab the doggy-bags you forgot, and now you can't find the damn thing.
Other applications can be envisioned, such as AR glasses that lets you know if you are about to walk into a steamer,
or perhaps city governments could use this to more efficiently clean public areas.

This module will contain an algorithm for training a pytorch network to detect poop in images, and a script
for detecting poop in unseen images given a pretrained model.

The dataset currently contains 25GB of outdoor images taken with a phone. The general process of acquiring the dataset has been:
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

* 2023-04-16 - More ground based photos. One "after" photo contains a positive example I didn't see in the background.
* 2023-03-11 - 305 new images. Many of these images are taken from a close up ground angle. I will continue to collect more in this way.
* 2023-01-01 - Another batch of leafy images.
* 2022-11-23 - We are thankful for more images 🦃
* 2022-09-19 - Added more images (With an indoor triple! wow! Thanks sick dog!)
* 2022-07-17 - Added more images
* 2022-06-20 - Added more images, starting transition to V1 CIDS
* 2022-04-02 - Added more images and updated analysis (Over 1000 Poop Images 🎉)
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

SnapCrap: An app to report poop on the streets of San Francisco
https://medium.com/@miller.stowe/snapcrap-why-i-built-an-app-to-report-poop-on-the-streets-of-san-francisco-aac12382a7ce

Other related links I haven't gone through well enough yet:

* https://getdiglabs.com/blogs/the-dig-labs-dish/computer-vision-and-dog-poop
* https://www.wired.co.uk/article/dog-poo-bin-cleanup
* https://www.reddit.com/r/robotics/comments/6p0rf0/can_i_use_opencv_to_get_my_robot_to_detect_dog/
* https://www.housebeautiful.com/lifestyle/kids-pets/a31289426/robot-picks-up-dog-poop/



Dataset Description
===================

The dataset contains a wide variety of image and background conditions that occur in upstate New York, including: seasonal changes, snow, rain, daytime, nighttime (some taken with flash, others taken with my phone's *night mode*), grass, concrete, etc...

Known dataset biases are:

* Geographic region: Most images were taken in Upstate New York climate.
* Sensor: Most images were taken with my Pixel 5. A few images were from my old Motorola Droid.
* Coordinate: Humans unconsciously center "objects of interest" in images they take. In some instances I tried to mitigate this bias, either by explicitly changing the center of the poop, or not looking at the screen when taking a snapshot.
* Me: I'm the only one taking pictures. I'm also fairly tall, so the images are all from my viewpoint. There are other "me" biases I may not be aware of.
* My Dogs: My two poop machines are fairly regular, and they have their own methods for times and places to make a dookie.
* Freshness: The shit I deal with is often fresh out of the oven. Although, I have picked up a decent number of abandoned stools from other dog owners in the area, some of these are quite old. And age of the sample does seem to have an impact on its appearance. New poops have a shine, while old ones are quite dull, and will start to break down.

The following scatterplot illustrates trends in the space / time distribution of the images.

.. .. image:: https://ipfs.io/ipfs/bafybeibnofjvl7amoiw6gx4hq5w3hfvl3iid2y45l4pipcqgl5nedpngzi/analysis/scat_scatterplot.png
.. image:: https://i.imgur.com/tL1rHPP.png
.. .. image:: https://imgur.com/DeUesAC.png
.. .. image:: https://imgur.com/q6XzSKa.png
.. .. image:: https://i.imgur.com/ne3AeC4.png


A visualization of the cumulative number of images collected over time is as follows:

.. .. image:: /analysis/images_over_time.png
.. image:: https://i.imgur.com/TfDYWOV.png
.. .. image:: https://imgur.com/vrAzrfj.png
.. .. image:: https://imgur.com/C2X1NCt.png
.. .. image:: https://i.imgur.com/ppPXo6X.png


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


Further updates will be added to this table. The number of images is total
images (including after and negatives). The (estimated) number of groups is
equal to the number of images with poop in them. And number of registered
groups is the number of groups the before / after pair had a successful
registration via the SIFT+RANSAC algorithm.


+-------------+----------+---------------------+-----------------------+
| Date        | # Images | # Estimated Groups  | # Registered Groups   |
+=============+==========+=====================+=======================+
| 2021-11-11  |  1935    |   ~616              | N/A                   |
+-------------+----------+---------------------+-----------------------+
| 2021-12-30  |  2088    |   ~728              | N/A                   |
+-------------+----------+---------------------+-----------------------+
| 2022-03-14  |  2471    |   ~954              | 658                   |
+-------------+----------+---------------------+-----------------------+
| 2022-04-02  |  2614    |  ~1002              | 697                   |
+-------------+----------+---------------------+-----------------------+
| 2022-04-16  |  2706    |  ~1033              | 722                   |
+-------------+----------+---------------------+-----------------------+
| 2022-06-20  |  2991    |  ~1127              | 734?                  |
+-------------+----------+---------------------+-----------------------+
| 2022-07-17  |  3144    |  ~1179              | 823                   |
+-------------+----------+---------------------+-----------------------+
| 2022-09-19  |  3423    |  ~1272              | 892                   |
+-------------+----------+---------------------+-----------------------+
| 2022-11-23  |  3667    |  ~1353              | 959                   |
+-------------+----------+---------------------+-----------------------+
| 2023-01-01  |  3800    |  ~1397              | 998                   |
+-------------+----------+---------------------+-----------------------+
| 2023-03-03  |  4105    |  ~1498              | 1068                  |
+-------------+----------+---------------------+-----------------------+
| 2023-04-16  |  4286    |  ~1559              | 1094                  |
+-------------+----------+---------------------+-----------------------+


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

Given kwcoco a formatted detection dataset, we can also use off-the-shelf detection baselines
via netharn, mmdet, or some other library that accepts coco/kwcoco input manifests.


Downloading the Data
====================


This dataset will be made public once I figure out a way to host and manage it.
Currently the raw images live on my hard drive, and are backed up across 2 machines, each running RAID-10.
Lower res copies of the photos live on the cloud, but I'm planning on sharing the originals.

The dataset is currently 26G+GB, and is currently hosted on IPFS.  Currently
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

The URL that can be viewed in a web browser: https://ipfs.io/ipfs/bafybeicjvjt2abdj7e5mpwq27itxi2u6lzcegl5dgw6nqe22363vmdsnru

Current IPFS addresses for the top-level dataset filesystem are:

.. code::

    bafybeiefroexxylbcka54ahtdngfvebvochmjndpj5xbliuldmp7wf4eqq - shitspotter_dvc/data.kwcoco.json
    bafybeibnapjzxiwlkveapyhqfu6mzul73rea7axh6b4wcmkt7q42brgkde - shitspotter_dvc/analysis
    bafybeialkwcgpkneapnixeit2uzyjehnjn27vwbaynjnlryl5flsf3q6fi - shitspotter_dvc/assets/poop-2020-12-28/
    bafybeig2wlncnixclod2hecd7e7zoemvdhwoxuzi5xtnomottb27kx4jzq - shitspotter_dvc/assets/poop-2021-02-06/
    bafybeifrkr2grtiuhm4uwuqri25h67dsfmsrwtn3q7xpfaeetqlwukgoum - shitspotter_dvc/assets/poop-2021-03-05/
    bafybeigspol3oqllgushdujw3dgzlnrgb5ywy42i3gtk5g2h7px3r25w6q - shitspotter_dvc/assets/poop-2021-04-06/
    bafybeibhtymsnsmdqukjf2emde6zlxapzczom7a4pauqv65pfducb4v2pu - shitspotter_dvc/assets/poop-2021-04-19/
    bafybeiecpxpodwxrmmkiyxef6222hobnr6okq35ecdcvlrt2wa4pduqpua - shitspotter_dvc/assets/poop-2021-04-25/
    bafybeicozcsoutdeh37nvv4a6xjuwha2s5yqssepg4x6rysvcaqyptluhm - shitspotter_dvc/assets/poop-2021-05-11T000000/
    bafybeig6v5abxioluw7zmk6mxzsg4xumhphkr64jqznjc2pgilhhg453b4 - shitspotter_dvc/assets/poop-2021-05-11T150000/
    bafybeiecdgnasyccutesze6odoyg2uhqkzc4hy25imbls2szpbwmsqsggm - shitspotter_dvc/assets/poop-2021-06-05/
    bafybeia5v47nt7m5dlw6ozfptreu6oxjdypjbbod3zhwx26hducphkg2em - shitspotter_dvc/assets/poop-2021-06-20/
    bafybeigo4ffpewvp23v6pa65durazqtzov7rpqucg6w3723bkolnhi2xwu - shitspotter_dvc/assets/poop-2021-09-20/
    bafybeie4sspgtzv5upgmj2gx43jcebnvmjpcnlehwpteyzbm7rm22buuli - shitspotter_dvc/assets/poop-2021-11-11/
    bafybeialx3qohukvftxoluss4iakdqiqjnsfyxu7qmcotbjci76zxy4wym - shitspotter_dvc/assets/poop-2021-11-26/
    bafybeigmpdet6dhtvkauisv7oyxvbcqlhiac4a6pgi7phn3zak37txireu - shitspotter_dvc/assets/poop-2021-12-27/
    bafybeiewsg5b353s26r566aw756y5h5omnjei3xllzv7sldesmthu6p5bi - shitspotter_dvc/assets/poop-2022-01-27/
    bafybeiapgukq36wxd3b23io3io5iry2jpu6ojy4pdc5wqry5ouy3s7q65u - shitspotter_dvc/assets/poop-2022-03-13-T152627/
    bafybeiba5k3iauqu4ayul4yozapadlpiehezwow63lm3r26hgk4eqrrjki - shitspotter_dvc/assets/poop-2022-04-02-T145512/
    bafybeic3amh4klgs3aantyqgd7lti2vhnnmutbcfddtvw2572ynlldkpua - shitspotter_dvc/assets/poop-2022-04-16-T135257/
    bafybeicyotgcgufq2nsewvk2ph4xchgbnltd7t2j334lqgvc4jdnxrw5by - shitspotter_dvc/assets/poop-2022-05-26-T173650/
    bafybeieddszhqi6fzrpnn2q2ab74hva4gwnx5bcdnvh7cwwrnf7ikyukru - shitspotter_dvc/assets/poop-2022-06-08-T132910/
    bafybeigss3h3p6pnsw7bgfevs77lv6duzhzi7fmuiyf5qtujafqanrrjsi - shitspotter_dvc/assets/poop-2022-06-20-T235340/
    bafybeih6qtza2vnrdvemlhuezfhoom6wh2457mnwmlw7sg4ncgstl35zsa - shitspotter_dvc/assets/poop-2022-07-16-T215017/
    bafybeigvu4k5w2eflpkmucaas3p4yb7mhdbpmcdsmysbpfa54biiy4vvya - shitspotter_dvc/assets/poop-2022-09-19-T153414/
    bafybeihnmt5pxlx5ywatlzvabtbebh6sspkaiithobqwjvrvfvzynl6oyy - shitspotter_dvc/assets/poop-2022-11-23-T182537/
    bafybeibx2oarr3liqrda4hd7xlw643vbd5nxff2b44blzccw7ekw6gbwv4 - shitspotter_dvc/assets/poop-2023-01-01-T171030/
    bafybeibky4jj4hhmlwuifx52fjdurseqzkmwpp4derwqvf5lo2vakzrtoe - shitspotter_dvc/assets/poop-2023-03-11-T165018/
    bafybeifj7uidepqz2wbumajacy2oacn7c7cuh6zwnduovn4xyszdpiodoe - shitspotter_dvc/assets/poop-2023-04-16-T175739/

    bafybeia37l74odvbshidnvk4vndr2jl2kfze6uls4jzfa5jzhxvcqkx7hq - shitspotter_dvc/assets
    bafybeic2ehnqled363zqimtbqbonagw6atgsyst5cqbm3wec6cg3te5ala - shitspotter_dvc

Despite the name, this is not yet a DVC repo.


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
