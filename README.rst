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


This ``shitspotter`` repo is where I am building the "ShitSpotter" poop-detection algorithm and dataset.
The primary goal of this work is to allow for the creation of a phone app that finds where your dog pooped,
because you ran to grab the doggy-bags you forgot, and now you can't find the damn thing.
Other applications can be envisioned, such as AR glasses that lets you know if you are about to walk into a steamer,
or perhaps city governments could use this to more efficiently clean public areas.

This module will contain an algorithm for training a pytorch network to detect poop in images, and a script
for detecting poop in unseen images given a pretrained model.

The dataset currently contains 32GB of outdoor images taken with a phone. The general process of acquiring the dataset has been:
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


Downloading
===========

All data is publicly hosted on IPFS and is free to use under "Creative Commons Attribution 4.0 International".

We use IPFS because it supports content addressable storage (CAS).  CAS has a
lot of benefits. Among these are: data duplication, simpler updates to "living
datasets", and verification of download success.  To learn more see the
Wikipedia article on:

* `content addressable storage <https://en.wikipedia.org/wiki/Content-addressable_storage>`_,
* `IPFS <https://en.wikipedia.org/wiki/InterPlanetary_File_System>`_.

The `IPFS CID <https://docs.ipfs.tech/concepts/content-addressing/>`_ (Content Identifier) for the most recent dataset is:

.. code::

    bafybeihuem7qz2djallypbb6bo5z7ojqnjz5s4xj6j3c4w4aztqln4tbzu

The dataset can be viewed in a webbrowser through an IPFS gateway:
https://ipfs.io/ipfs/bafybeihuem7qz2djallypbb6bo5z7ojqnjz5s4xj6j3c4w4aztqln4tbzu

If you have an IPFS node, please help keep this dataset alive and available by pinning it.

Sometimes IPFS can be slow, especially for the latest data. However, older URLs can be faster to access: e.g.
https://ipfs.io/ipfs/QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG



Recent Updates
==============

Check back for updates, but because this is a personal project, it might take
some time for it to fully drop.

* 2024-01-31 - First update of 2024. New images are being added to the validation split. New CID is ``bafybeibxxrs3w7iquirv262ctgcwgppgvaglgtvcabb76qt5iwqgwuzgv4``.
* 2023-12-31 - Last update of 2023. We also welcome a new content contributor: Roady. Details will be added in the acknowledgements. New CID is ``bafybeihuem7qz2djallypbb6bo5z7ojqnjz5s4xj6j3c4w4aztqln4tbzu``.
* 2023-12-20 - More images and many more annotations. Data collected next year (2024) will be part of the validation set. New CID is ``bafybeifkufkmmx3qxbvxe5hbskxr4gijkevcryxwp3mys2pqf4yjv2tobu``.
* 2023-11-17 - More images and annotations. New CID is ``bafybeie275n5f4f64vodekmodnktbnigsvbxktffvy2xxkcfsqxlie4hrm``.
* 2023-10-19 - A few new images, the last images from Bezoar, who passed away today.
* 2023-10-15 - The next phase of the project - annotation and training - has begun. Also 82 new images.
* 2023-08-22 - 182 new images.
* 2023-07-01 - Another batch of 300 photos. I also realized that if I could ID which dog made which poop, I could do a longiturdinal study.
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
    - Initial SIFT-based matching isn't as robust as I'd hoped.
    - First data is on IPFS, still need to open ports. ID of the root dataset is: ``QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG``
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


The name "ShitSpotter" is an homage to my earlier work: `HotSpotter <https://github.com/Erotemic/hotspotter>`_, which later became `IBEIS <https://github.com/Erotemic/ibeis>`_ This is work on individual animal identification, particularly Zebras. This work is continued by `WildMe <https://www.wildme.org/>`_.

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
This is the most similar thing to this project that I've seen so far.
He enumerates many reasons why it is beneficial to remove dog waste from our
environment, and considers many applications for a dog poop detector. He has a
dataset of 100 dog poop images and used FasterRCNN as a baseline dataset.
I have reached out to him to see if he is interested in collaborating.

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
* Me: I'm ~the only one~ the main person taking pictures. I'm also fairly tall, so the images are all from my viewpoint. There are other "me" biases I may not be aware of.
* My Dogs: My two poop machines are fairly regular, and they have their own methods for times and places to make a dookie.
* Freshness: The shit I deal with is often fresh out of the oven. Although, I have picked up a decent number of abandoned stools from other dog owners in the area, some of these are quite old. And age of the sample does seem to have an impact on its appearance. New poops have a shine, while old ones are quite dull, and will start to break down.

The following scatterplot illustrates trends in the space / time distribution of the images.

.. .. image:: https://ipfs.io/ipfs/bafybeibnofjvl7amoiw6gx4hq5w3hfvl3iid2y45l4pipcqgl5nedpngzi/analysis/scat_scatterplot.png
.. image:: https://i.imgur.com/78EfIpl.png
.. .. image:: https://i.imgur.com/tL1rHPP.png
.. .. image:: https://imgur.com/DeUesAC.png
.. .. image:: https://imgur.com/q6XzSKa.png
.. .. image:: https://i.imgur.com/ne3AeC4.png


A visualization of the cumulative number of images collected over time is as follows:

.. .. image:: /analysis/images_over_time.png
.. image:: https://i.imgur.com/lQCNvNn.png
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


+-------------+----------+---------------------+-----------------------+-----------------------+
| Date        | # Images | # Estimated Groups  | # Registered Groups   | # Annotated Images    |
+=============+==========+=====================+=======================+=======================+
| 2021-11-11  |  1935    |   ~616              | N/A                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2021-12-30  |  2088    |   ~728              | N/A                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-03-14  |  2471    |   ~954              | 658                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-04-02  |  2614    |  ~1002              | 697                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-04-16  |  2706    |  ~1033              | 722                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-06-20  |  2991    |  ~1127              | 734?                  | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-07-17  |  3144    |  ~1179              | 823                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-09-19  |  3423    |  ~1272              | 892                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2022-11-23  |  3667    |  ~1353              | 959                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-01-01  |  3800    |  ~1397              | 998                   | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-03-03  |  4105    |  ~1498              | 1068                  | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-04-16  |  4286    |  ~1559              | 1094                  | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-07-01  |  4594    |  ~1662              | 1154                  | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-08-22  |  4776    |  ~1723              | 1197                  | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-09-22  |  4899    |  ~1764              | 1232                  | 0                     |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-10-15  |  4981    |  ~1790              | 1255                  | 362                   |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-10-20  |  5019    |  ~1804              | 1266                  | 430                   |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-11-17  |  5141    |  ~1845              | 1304                  | 919                   |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-12-20  |  5249    |  ~1881              | 1337                  | 1440                  |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2023-12-31  |  5330    |  ~1908              | 1360                  | 1440                  |
+-------------+----------+---------------------+-----------------------+-----------------------+
| 2024-01-31  |  5533    |  ~1975              | 1411                  | 1964                  |
+-------------+----------+---------------------+-----------------------+-----------------------+


For further details, see the `Datasheet <DATASHEET.md>`_.


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


Update: 2023-10-15

The before/after annotation process is unfortunately not robust enough to
generate annotations. This additional structure is still of interest for
defining change detection problems or other processing, but bootstrapping the
annotation process is harder than originally anticipated.

In lieu of difference-image annotations, annotations are being added with an AI assisted annotation tool: `labelme <https://github.com/wkentaro/labelme>`_. This tool leverages the `Segment Anything Model (SAM) <https://segment-anything.com/>`_, which does a good job at finding poop polygon boundaries from a single click. This process is not perfect, and annotations are corrected when they are incorrectly generated. In some difficult cases the SAM model is unable to segment the object of interest at all.

The following is a screenshot of the annotation tool with two easy cases and
one harder case that SAM struggled with on the top.

.. image:: https://i.imgur.com/3lmXgww.png


The labelme annotations are kept in their original form as sidecar json files
to the original images. However, when the dataset is updated, these annotations
are converted and stored in the top-level kwcoco dataset.


The Algorithm
=============

Currently there is no algorithm checked into the repo. I need to start annotating the dataset first.
Eventually there will be a ``shitspotter.fit`` and ``shitspotter.predict`` script for training and performing
inference on unseen images. My current plan for a baseline algorithm is a mobilenet backbone pretrained
on imagenet and some single-stage detection / segmentation head on top of that.

Given kwcoco a formatted detection dataset, we can also use off-the-shelf detection baselines
via netharn, mmdet, or some other library that accepts coco/kwcoco input manifests.

Update: 2023-10-15

The `geowatch <https://gitlab.kitware.com/computer-vision/geowatch>`_ framework
is being used to train initial models on the small set of annotations.


Initial train and validation batches look like this:

.. image:: https://i.imgur.com/Nfk8XbE.jpg


.. image:: https://i.imgur.com/YHfl0Wd.jpg


An example prediction from an initial model on a full validation image is:

.. image:: https://i.imgur.com/ya4jnAO.jpg


Clearly there is still more work to do, but training a deep network is an art,
and I have full confidence that a high quality model is possible. The training
batches are starting to fit the data, but the validation batches shows that
there is still a clear generalization gap, but this is only the very start of
training and the hyper-parameters are untuned.


The current train validation split is defined in the ``make_splits.py`` file.
Only "before" images with annotations are currently considered. The "after"
images and "negative" will be taken into account when they are properly
associated with the "before" images in the kwcoco metadata. The early images
before 2021 are used for validation, whereas everything else is used for
training. Contributor data is also currently held out and can serve as a test
set once annotations are placed.


Data Management
===============

The full resolution dataset is public and hosted on IPFS.

Despite the name, this is not yet a DVC repo.  Eventually I would like to host
the data via DVC + IPFS, but fsspec needs a mature IPFS filesystem
implementation first. I may also look into git-annex as an alternative to DVC.

The licence for the software will be Apache 2. The license for the data will be
"Creative Commons Attribution 4.0 International".

In addition to these licenses please:

* Cite the work if you use it.
* If you annotate any of the images, contribute the annotations back. Picking up shit is a team effort.
* When asked to build something, particularly ML systems, think about the ethical implications, and act ethically.
* Pin the dataset on IPFS if you can.

Otherwise the data is free to use commercially or otherwise.

The URL that can be viewed in a web browser: https://ipfs.io/ipfs/bafybeigovcysmghsyab6ia3raycsebbc32kea2k4qoxcsujmp52hzpsghy

Current IPFS addresses for the top-level dataset filesystem are:

.. temp



.. code::

    bafybeieydez2b6tksq5c26l4quhx5475et5ttvipuc7hs6n5khaolomilm - shitspotter_dvc/assets/_contributions
    bafybeidap2man4erddpk74ql253cutjeqisxoeu5mtaal52hpjbwrdy3fy - shitspotter_dvc/assets/_horse-poop-2022-05-26
    bafybeidmcwo5lugzs5pjdwp3rvhgorz6zzw2of6s3surdnth5yz4hkxt2m - shitspotter_dvc/assets/_poop-unstructured-2021-02-06
    bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354 - shitspotter_dvc/assets/_trashed
    bafybeigl4v7dlltjmyvujoo563wf6uoj7pqrbudkatar7h4zagqbe73hd4 - shitspotter_dvc/assets/_unstructured
    bafybeieony6ygiipdp324ibuqhdggefsaa7ykqrxuxoqgobnvhpkqhq2gi - shitspotter_dvc/assets/poop-2020-12-28
    bafybeiddzhnsovxx76pgb65p7kekfmlz4i6afqsdrbdnazs3h6cxhosr3i - shitspotter_dvc/assets/poop-2021-02-06
    bafybeifrkr2grtiuhm4uwuqri25h67dsfmsrwtn3q7xpfaeetqlwukgoum - shitspotter_dvc/assets/poop-2021-03-05
    bafybeigspol3oqllgushdujw3dgzlnrgb5ywy42i3gtk5g2h7px3r25w6q - shitspotter_dvc/assets/poop-2021-04-06
    bafybeibshwnzyerfheehpt7qhw7jojjjrb5g2a74yvpwqm2wcadpyjjzny - shitspotter_dvc/assets/poop-2021-04-19
    bafybeiecpxpodwxrmmkiyxef6222hobnr6okq35ecdcvlrt2wa4pduqpua - shitspotter_dvc/assets/poop-2021-04-25
    bafybeigzkx5xxju2rbj5zai3o7vppwqbjso7tj23q77deqymjsf7trubzu - shitspotter_dvc/assets/poop-2021-05-11T000000
    bafybeiasq55mc6nba3akml5c4niupbpfbyqtzcm2kjv7klgorllm5e3qna - shitspotter_dvc/assets/poop-2021-05-11T120000-notes.txt
    bafybeig6v5abxioluw7zmk6mxzsg4xumhphkr64jqznjc2pgilhhg453b4 - shitspotter_dvc/assets/poop-2021-05-11T150000
    bafybeiecdgnasyccutesze6odoyg2uhqkzc4hy25imbls2szpbwmsqsggm - shitspotter_dvc/assets/poop-2021-06-05
    bafybeia5v47nt7m5dlw6ozfptreu6oxjdypjbbod3zhwx26hducphkg2em - shitspotter_dvc/assets/poop-2021-06-20
    bafybeigo4ffpewvp23v6pa65durazqtzov7rpqucg6w3723bkolnhi2xwu - shitspotter_dvc/assets/poop-2021-09-20
    bafybeibrw7je4zmoartzrpq5vbvg7klim5gr5j3q44doeb3tbxkkboftvi - shitspotter_dvc/assets/poop-2021-11-11
    bafybeid7yfx6u4yacxpnmzg5vhwh7e47lga5oj3tpmdup3omo6s7yx54ee - shitspotter_dvc/assets/poop-2021-11-26
    bafybeicedyv5dfy5x6yb2vw5quliajx2emrusssnev2v3qz3xdm7h6fsyy - shitspotter_dvc/assets/poop-2021-12-27
    bafybeiewsg5b353s26r566aw756y5h5omnjei3xllzv7sldesmthu6p5bi - shitspotter_dvc/assets/poop-2022-01-27
    bafybeiapgukq36wxd3b23io3io5iry2jpu6ojy4pdc5wqry5ouy3s7q65u - shitspotter_dvc/assets/poop-2022-03-13-T152627
    bafybeiba5k3iauqu4ayul4yozapadlpiehezwow63lm3r26hgk4eqrrjki - shitspotter_dvc/assets/poop-2022-04-02-T145512
    bafybeic3amh4klgs3aantyqgd7lti2vhnnmutbcfddtvw2572ynlldkpua - shitspotter_dvc/assets/poop-2022-04-16-T135257
    bafybeicyotgcgufq2nsewvk2ph4xchgbnltd7t2j334lqgvc4jdnxrw5by - shitspotter_dvc/assets/poop-2022-05-26-T173650
    bafybeieddszhqi6fzrpnn2q2ab74hva4gwnx5bcdnvh7cwwrnf7ikyukru - shitspotter_dvc/assets/poop-2022-06-08-T132910
    bafybeigss3h3p6pnsw7bgfevs77lv6duzhzi7fmuiyf5qtujafqanrrjsi - shitspotter_dvc/assets/poop-2022-06-20-T235340
    bafybeih6qtza2vnrdvemlhuezfhoom6wh2457mnwmlw7sg4ncgstl35zsa - shitspotter_dvc/assets/poop-2022-07-16-T215017
    bafybeigvu4k5w2eflpkmucaas3p4yb7mhdbpmcdsmysbpfa54biiy4vvya - shitspotter_dvc/assets/poop-2022-09-19-T153414
    bafybeid6guu5vv5zj467bkxpt3zkg2mn45q7kxab5tteps7hzpiuyam7mi - shitspotter_dvc/assets/poop-2022-11-23-T182537
    bafybeibx2oarr3liqrda4hd7xlw643vbd5nxff2b44blzccw7ekw6gbwv4 - shitspotter_dvc/assets/poop-2023-01-01-T171030
    bafybeibky4jj4hhmlwuifx52fjdurseqzkmwpp4derwqvf5lo2vakzrtoe - shitspotter_dvc/assets/poop-2023-03-11-T165018
    bafybeifj7uidepqz2wbumajacy2oacn7c7cuh6zwnduovn4xyszdpiodoe - shitspotter_dvc/assets/poop-2023-04-16-T175739
    bafybeihhbwe6mtkts7335e2wdr3p4mo5impx3niqbcavvqh3l3rknpbuti - shitspotter_dvc/assets/poop-2023-07-01-T160318
    bafybeiez6f2nwubarmduko73uclgitsaagvdov4s5oexcwltw5dosjhq4m - shitspotter_dvc/assets/poop-2023-08-22-T202656
    bafybeihurilrwce7rxr7o3iqdf227o74cfk23ilv2nleoj5hd6wx5iapz4 - shitspotter_dvc/assets/poop-2023-09-22-T180825
    bafybeihsxlzwr45jvxzhq7vst6zirykdm4ufbmapxidl5bs4ncyfo7nmja - shitspotter_dvc/assets/poop-2023-10-15-T193631
    bafybeiew5srmawar4qjkj3iohhg7i7fnc24ik3ym5is5y4d7ftho47puoq - shitspotter_dvc/assets/poop-2023-10-19-T212018
    bafybeicqdlnupmpn54ehiqfqwhiwejh5sl5dizqsb2gsr6rk6aszszu2ue - shitspotter_dvc/assets/poop-2023-11-16-T154909
    bafybeiboaujmbfrmopu4qguc6klv2s7ubxq3z4fka2u3d5m6i7waykonuy - shitspotter_dvc/assets/poop-2023-12-19-T190904
    bafybeieyi3erbwzu5couwg4lrgr3xynq4xwtsoho3md6rhr6qfn5icl2vu - shitspotter_dvc/assets/poop-2023-12-19-T190904

    bafybeigdoerp75pscmq7veoi2quh7p5oc3doeklepyscymrvt4hlcl4yvu - shitspotter_dvc/assets
    bafybeibxxrs3w7iquirv262ctgcwgppgvaglgtvcabb76qt5iwqgwuzgv4 - shitspotter_dvc


Acknowledgements
================

I want to give thanks to the people and animals-that-think-they-are-people who
contributed to this project.  My colleagues at
`Kitware <https://www.kitware.com/>`_ have provided valuable help / insight into
project direction, dataset collection, problem formulation, related research,
discussion, and memes.

I would also like to thank the several people that have contributed their own
images in the contributions folder (More info on contributions will be added
later).

I want to give special thanks to my first two poop machines - Honey and Bezoar
- who inspired this project. Without them, ShitSpotter would not be possible.

.. Image of Honey And Bezoar
.. image:: https://i.imgur.com/MWQVs0w.jpg


.. Multiple Images of Honey And Bezoar
.. image:: https://i.imgur.com/YUJjWoh.jpg


Honey - (~2013 - ) - Adopted in 2015 and named for a
`sweet concoction <https://en.wikipedia.org/wiki/Honey>`_ of insect
regurgitation and enzymes, Honey is often called out for her resemblance to a
fox and is notable for her eagerness for attention and outgoing personality.
DNA analysis indicates that she is part Boxer, Beagle, German Shepard, and
Golden Retriever.  Honey's likes include: breakfast, sniffing stinky things,
digging holes, sleeping on soft things, viciously shaking small furry objects,
and whining for absolutely no reason.  Honey's dislikes include: baths, loud
noises, phone calls, and arguments.  Honey came to us from Ohio as a fearful
dog, but has always been open to trusting new people.  She has grown into an
intelligent and willful dog with a scrappy personality.

.. An Image of Honey
.. image:: https://i.imgur.com/gUzwgCT.jpg
   :height: 400px
   :align: left
.. bafybeihuhrp6wtle5wuhsgcgf6bp7w4ol4pft7y2pcplylzly7gfag74lm bafybeic5a4kjrb37tdmc6pzlpcxe2x6hc4kggemnqm2mcdu4tmrzvir6vm/Contributor-Honey.jpg


Bezoar - (~2018 - 2023-10-19) - Adopted in 2020 and named for a
`calcified hairball <https://en.wikipedia.org/wiki/Bezoar>`_, Bezoar was an
awkward and shy dog, but grew into a curious and loving sweetheart.  Her DNA
test indicated she was part Stafford Terrier, Cane Corso, Labrador Retriever,
German Shepard, and Rhodesian Ridgeback.  Bezoar's likes included: breakfast, a
particular red coco plush, boops (muzzle nudges), chasing squirrels, and
running in the park, Bezoar's dislikes included: baths, sudden movements, rainy
weather, and coming inside before she is ready.  Bezoar came to us from Alabama
with bad heartworm and experienced a host of health problems through her life.
In 2022 she was diagnosed with rare form of osteosarcoma in her nose, which is
an aggressive bone cancer, but she had a rare progression and lived a quality
life for over a year and a half without significant tumor growth.  Sadly, in
October 2023, rapid growth resumed and she was euthanized while surrounded by
her close friends and family.  To say she will be missed is an understatement;
there are no words that can describe my grief or the degree to which she
enriched my life.  I take comfort in knowing that she may be in part
immortalized through her contributions to this dataset.

.. An Image of Bezoar
.. image:: https://i.imgur.com/Z3TCZ47.jpg
   :height: 400px
   :align: left
.. bafybeibr33vb5m3ytovwputzai2vka2sjovmguktyk7yjp3emvtoihp7he bafybeic5a4kjrb37tdmc6pzlpcxe2x6hc4kggemnqm2mcdu4tmrzvir6vm/Contributor-Bezoar.jpg



Roady - (2016-04-29 - ) - Adopted in 2023, Roady is the newest addition to the
pack, and his bio will be updated as we learn more about him. He is an
Australian Cattle Dog (Blue Healer) with tons of energy and tons of poopy.
Roadies likes include: fetching the ball, staring deeply into eyes, pets, and invading personal space.
Roadies dislikes include: dropping the ball, others not paying attention to him, and spinach.

.. An Image of Roady
.. image:: https://i.imgur.com/yaZi5bO.jpg
   :height: 400px
   :align: left


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
