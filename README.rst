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


This ``shitspotter`` repo is where I am building the "ShitSpotter" (or
"ScatSpotter" in a formal setting) poop-detection algorithm and dataset.  The
primary goal of this work is to allow for the creation of a phone app that
finds where your dog pooped, because you ran to grab the doggy-bags you forgot,
and now you can't find the damn thing.  Other applications can be envisioned,
such as AR glasses that lets you know if you are about to walk into a steamer,
or perhaps city governments could use this to more efficiently clean public
areas.

This module will contain an algorithm for training a pytorch network to detect poop in images, and a script
for detecting poop in unseen images given a pretrained model.

The dataset currently contains mostly outdoor images taken with a phone. The general process of acquiring the dataset has been:
1. My dog poops or I see a rogue poop,
2. I take a "before" picture of the poop,
3. I pick up the poop,
4. I take an "after" picture as a high-correlation negative, and
5. I take a 3rd image of a different nearby area to get a lower-correlation-negative.
New data is added roughly each month, and each new set of data adds about 1GB
to the dataset size.
Most of the dataset is unannotated with segmentation polygons.
Annotations and the data manifest are managed using kwcoco.

The code and the dataset are open source with permissive licensing.
The data and pretrained models are distributed via IPFS, BitTorrent, and
centralized mechanisms.


Major Milestones and Goals
==========================

This following is the high level status of the project.

- ☑ Image collection started (2020-12-18)
- ☑ Image collection published (2021-12-30)
- ☑ Dataset has enough images to train models (2023-03-03)
- ☑ Dataset has enough annotations to train models (2023-11-17)
- ☑ Baseline models are trained (2024-07-27)
- ☑ Scientific paper about dataset `published on arxiv <https://www.arxiv.org/abs/2412.16473>`_ (2024-12-21)
- ☑ Scientific paper about dataset peer-reviewed (2025-07-30ish)
- ☑ Efficient models for phones are trained (2025-07-30ish)
- ☐ Phone application is developed (unreviewed external prototype: https://github.com/mkorzunowicz/poopdetector/releases/tag/v1.1.0)
- ☐ Phone application is available for download and usage
- ☐ Phone application is release on app stores for free
- ☐ Phone application demonstrates value.


Introduction
============

In Fall 2019, I was at the local dog park, and I found myself in a situation
where my dog pooped, but I had forgotten to bring bags with me. I walked to the
local bag station (thank you D.G.S.), grabbed one, but then I couldn't find where
the poop was. The brown fallen leaves made it very difficult to find the poop.

This happened every so often. Usually I would be able to find it, but there
were times I was completely unable to find the "object of interest". This got
me thinking, what if I had a phone app that could scan the area with the camera
and try to locate the poop? If I had a dataset, training a poop detection model
with today's deep learning methods should work pretty well.

Thus, on 2020-12-18, I took my first picture. My dog pooped, I took a picture,
I picked it up, and then I took an "after" picture. The idea is that I will
align the pictures (probably via computing local features like sift or some
deep variant and then estimating an affine/projective transform) and then take
a difference image. That should let me seed some sort of semi-automated
annotation process.

Then in 2021-05-11, one of my colleagues suggested that I take a 3rd unrelated
picture to use as negative examples, so I took that suggestion and started
doing that. This is the process currently being used. The following figure
illustrates an example of one of these "triples".

.. image:: https://i.imgur.com/NnEC8XZ.jpg


The name "ShitSpotter" is an homage to my earlier work: `HotSpotter <https://github.com/Erotemic/hotspotter>`_, which later became `IBEIS <https://github.com/Erotemic/ibeis>`_ This is work on individual animal identification, particularly Zebras. After 2017, this work was continued in Wildbook by `WildMe <https://www.wildme.org/>`_, which merged with `Conservation X Labs <https://www.conservationxlabs.com/>`_ in 2024.


Downloading the Data
====================

All data is publicly hosted on IPFS and is free to use under
"Creative Commons Attribution 4.0 International" `(CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`_.

We use IPFS because it supports content addressable storage (CAS).  CAS has a
lot of benefits. Among these are: data duplication, simpler updates to "living
datasets", and verification of download success.  To learn more see the
Wikipedia article on:

* `content addressable storage <https://en.wikipedia.org/wiki/Content-addressable_storage>`_,
* `IPFS <https://en.wikipedia.org/wiki/InterPlanetary_File_System>`_.


The following `IPNS <https://docs.ipfs.tech/concepts/ipns/>`_ address should always point to the latest version of the dataset:
`/ipns/k51qzi5uqu5dje1ees96dtsoslauh124drt5ajrtr85j12ae7fwsfhxb07shit <https://ipfs.io/ipns/k51qzi5uqu5dje1ees96dtsoslauh124drt5ajrtr85j12ae7fwsfhxb07shit>`_.

This should resolve to the most recent `IPFS CID <https://docs.ipfs.tech/concepts/content-addressing/>`_ (Content Identifier) of:
`/ipfs/bafybeidfxayyacq4jbbhjcxbxumqlspmsmkj52nq2ns4vfew5udgysocoy <https://ipfs.io/ipfs/QmQonrckXZq37ZHDoRGN4xVBkqedvJRgYyzp2aBC5Ujpyp?redirectURL=bafybeidfxayyacq4jbbhjcxbxumqlspmsmkj52nq2ns4vfew5udgysocoy&autoadapt=0&requiresorigin=0&web3domain=0&immediatecontinue=1&magiclibraryconfirmation=0>`_.

This can be viewed in a webbrowser through an IPFS gateway:

If you have an IPFS node, please help keep this dataset alive and available by pinning it.

Sometimes IPFS can be slow, especially for the latest data. However, older CIDs can be faster to access: e.g.
`/ipfs/QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG <https://ipfs.io/ipfs/QmQonrckXZq37ZHDoRGN4xVBkqedvJRgYyzp2aBC5Ujpyp?redirectURL=QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG&autoadapt=0&requiresorigin=0&web3domain=0&immediatecontinue=1&magiclibraryconfirmation=0>`_.


The the goal is to make IPFS the main distribution mechanism. However, it is
still relatively new technology and until all of the kinks are worked out, the
dataset will be mirrored on a centralized Girder server:
https://data.kitware.com/#user/598a19658d777f7d33e9c18b/folder/66b6bc34f87a980650f41f90

.. .. OLD and broken (its a dead link!): https://data.kitware.com/#user/598a19658d777f7d33e9c18b/folder/65d6c52fb40ab0fa6c57909b

Unlike IPFS, which (ideally) gives seamless access to the data, the centralized
storage has the upload from each update grouped in its own zipfile. If
annotations for one of these folders changes, the entire zipfile will be
reuploaded, so there will be no mechanism for version control.


While we are interested in IPFS as a distribution mechanism, we recognize it is
a newer technology and we also periodically seed versions of the dataset using
bittorrent:

https://academictorrents.com/details/ee8d2c87a39ea9bfe48bef7eb4ca12eb68852c49

We also have the dataset on `hugging face
<https://huggingface.co/datasets/erotemic/shitspotter>`__, which is currently
offering the best download speeds.


Models are also available on huggingface: https://huggingface.co/erotemic/shitspotter-models

Recent Updates
==============

Check back for updates, but because this is a personal project, it might take
some time for it to fully drop.

* 2025-07-04 - Releasing new data on IPFS. The growth seems to be increasing. Will take 7-9 more years to get 30k images.
* 2025-04-20 - The number of images is now over 9000! The dataset is now `mirrored on hugging face <https://huggingface.co/datasets/erotemic/shitspotter>`__.
* 2025-03-09 - Bunch of new images, with somewhat of a domain shift. The detectron model is good at annotating new images, but still not good enough. More work to be done.
* 2024-12-31 - It is the end of 2024, lots has changed: new varied images, new privacy policy, and new contributions. Happy new year 🎊, all new 2025 images will go into the train set.
* 2024-09-16 - It's not part of a triple (I did not have a bag with me) but the dataset now has an international poop.
* 2024-07-03 - Happy 4th 🎆, my dogs are shitting themselves.
* 2024-06-15 - Small image drop. Working on writeup. Training new models.
* 2024-05-21 - Slowing down release cycles. Still collecting images at roughly the same rate. CIDs for recent and previous releases are now in the CID table.
* 2024-03-30 - This includes recent models that have been performing reasonably well.
* 2024-02-29 - Going to change this year to be 1/3 validation, next update will have a new split. Will also rework this README eventually.
* 2024-02-22 - Added centralized Girder download link to increase accessibility of the data with an ok-ish pretrained model.
* 2024-01-31 - First update of 2024. New images are being added to the validation split.
* 2023-12-31 - Last update of 2023. We also welcome a new content contributor: Roadie. Details will be added in the acknowledgements.
* 2023-12-20 - More images and many more annotations. Data collected next year (2024) will be part of the validation set.
* 2023-11-17 - More images and annotations.
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

Related Work
============

I was surprised to find that there does not seem to be much work on this problem in the outdoor setting.
Because none of the related work exactly meets my needs, I haven't looked too in depth into much of it,
it could be that some of these are more relevant than I've given them credit for. As time moves on
I'll continue to refine this section.

Apparently Roomba has an indoor poop dataset: https://www.engadget.com/irobot-roomba-j-7-object-poop-detection-040152887.html It would be interesting to combine the indoor / outdoor datasets, but we are more concerned about outdoor detection. Maybe Boston Dynamics and Roomba can take this dataset and do something interesting.

The MSHIT fake dog poop dataset: https://www.kaggle.com/mikian/dog-poop is similar to this domain, but not the real-deal.
THe dataset consists of 3.89GB of real images with fake poop (e.g. plastic
poop) in controlled environments.

There is Human Poop Classification: https://seed.com/poop/ and https://www.theverge.com/2019/10/29/20937108/poop-database-ai-training-photo-upload-first-mit but this is not our domain.

Detect Images of Dogs Pooping: https://colab.research.google.com/github/matthewchung74/blogs/blob/dev/Dog_Pooping_Dectron.ipynb
Unfortunately, this is detecting the action, and not the consequence.

Calab Olson trained a dog-pose recognition network to detect when a specific dog was pooping.
https://github.com/calebolson123/DogPoopDetector
https://calebolson.com/blog/2022/01/14/dog-poop-detector.html
https://www.youtube.com/watch?v=uWZu3rnj-kQ

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
It is now defunct and no longer available.

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
.. image:: https://i.imgur.com/aPvRJ3q.png
.. .. image:: https://i.imgur.com/78EfIpl.png
.. .. image:: https://i.imgur.com/tL1rHPP.png
.. .. image:: https://imgur.com/DeUesAC.png
.. .. image:: https://imgur.com/q6XzSKa.png
.. .. image:: https://i.imgur.com/ne3AeC4.png


A spatial visualization of where the majority of images were taken is as follows:


.. .. image:: https://ipfs.io/ipfs/<HEAD>/analysis/maps/map_0000.png
.. image:: https://i.imgur.com/Guz019L.png

A visualization of the cumulative number of images collected over time is as follows:

.. .. image:: /analysis/images_over_time.png
.. image:: https://i.imgur.com/KkrKx7e.png
.. .. image:: https://i.imgur.com/lQCNvNn.png
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


+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| Date        | # Images | # Estimated Groups  | # Registered Groups   | # Annotated Images    | CID                                                          |
+=============+==========+=====================+=======================+=======================+==============================================================+
| 2021-11-11  | 1935     | ~616                | N/A                   | 0                     | -                                                            |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2021-12-30  | 2088     | ~728                | N/A                   | 0                     | QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG               |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-03-14  | 2471     | ~954                | 658                   | 0                     | QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6               |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-04-02  | 2614     | ~1002               | 697                   | 0                     | QmfStoay5rjeHMEDiyuGsreXNHsyiS5kVaexSM2fov216j               |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-04-16  | 2706     | ~1033               | 722                   | 0                     | -                                                            |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-06-20  | 2991     | ~1127               | 734?                  | 0                     | bafybeihltrtb4xncqvfbipdwnlxsrxmeb4df7xmoqpjatg7jxrl3lqqk6y  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-07-17  | 3144     | ~1179               | 823                   | 0                     | bafybeihi7v7sgnxb2y57ie2dr7oobigsn5fqiwxwq56sdpmzo5on7a2xwe  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-09-19  | 3423     | ~1272               | 892                   | 0                     | bafybeiedk6bu2qpl4snlu3jmtri4b2sf476tgj5kdg2ztxtm7bd6ftzqyy  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2022-11-23  | 3667     | ~1353               | 959                   | 0                     | bafybeibnofjvl7amoiw6gx4hq5w3hfvl3iid2y45l4pipcqgl5nedpngzi  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-01-01  | 3800     | ~1397               | 998                   | 0                     | bafybeihicisq66veupabzpq7gutxd2sikfe43jvtirield4wlnznpanj24  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-03-03  | 4105     | ~1498               | 1068                  | 0                     | bafybeicjvjt2abdj7e5mpwq27itxi2u6lzcegl5dgw6nqe22363vmdsnru  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-04-16  | 4286     | ~1559               | 1094                  | 0                     | bafybeic2ehnqled363zqimtbqbonagw6atgsyst5cqbm3wec6cg3te5ala  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-07-01  | 4594     | ~1662               | 1154                  | 0                     | bafybeiflkm37altah2ey2jxko7kngquwfugyo4cl36y7xjf7o2lbrgucbi  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-08-22  | 4776     | ~1723               | 1197                  | 0                     | bafybeiczi4pn4na2iw7c66bpbf5rdr3ua3grp2qvjgrmnuzqabjjim4o2q  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-09-22  | 4899     | ~1764               | 1232                  | 0                     | bafybeieahblb6aafomi72gnheu3ihom7nobdad4t6jcrrwhd5eb3wxkrgy  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-10-15  | 4981     | ~1790               | 1255                  | 362                   | bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-10-20  | 5019     | ~1804               | 1266                  | 430                   | bafybeigovcysmghsyab6ia3raycsebbc32kea2k4qoxcsujmp52hzpsghy  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-11-17  | 5141     | ~1845               | 1304                  | 919                   | bafybeie275n5f4f64vodekmodnktbnigsvbxktffvy2xxkcfsqxlie4hrm  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-12-20  | 5249     | ~1881               | 1337                  | 1440                  | bafybeifkufkmmx3qxbvxe5hbskxr4gijkevcryxwp3mys2pqf4yjv2tobu  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2023-12-31  | 5330     | ~1908               | 1360                  | 1440                  | bafybeihuem7qz2djallypbb6bo5z7ojqnjz5s4xj6j3c4w4aztqln4tbzu  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-01-31  | 5533     | ~1975               | 1411                  | 1964                  | bafybeibxxrs3w7iquirv262ctgcwgppgvaglgtvcabb76qt5iwqgwuzgv4  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-02-29  | 5771     | ~2054               | 1479                  | 1964                  | bafybeia2gphecs3pbrccwopg63aka7lxy5vj6btcwyazf47q6jlqjgagru  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-03-30  | 6019     | ~2137               | 1549                  | 2133                  | bafybeibw5xqmdiycd7vw5qqdf3ceidjbq3cv4taalkc3ruu3qeqmqdy6sm  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-05-21  | 6373     | ~2255               | 1640                  | 2252                  | bafybeidle54us5cdwpzzis4h52wjmtsk643gprx7nvvtd6g26mxq76kfjm  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-06-15  | 6545     | ~2313               | 1684                  | 2311                  | bafybeia44hiextgcpjfvglib66gxziaf7jkvno63p7h7fsqkxi5vpgpvay  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-07-03  | 6648     | ~2347               | 1711                  | 2346                  | bafybeiedwp2zvmdyb2c2axrcl455xfbv2mgdbhgkc3dile4dftiimwth2y  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-09-16  | 7108     | ~2500               | 1824                  | 2501                  | bafybeibn3kmmz3ytrlmt2pwbifvcwv7veddoeuabtifgvztetilnav2gom  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2024-12-31  | 8291     | ~2894               | 2108                  | 2898                  | bafybeie2nfp6km4x63ldpysnje4qaggijnh5jilgawjcdnahoddvxln3xm  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2025-03-09  | 8726     | ~3040               | 2200                  | 3046                  | bafybeihsd6rwjha4kbeluwdjzizxshrkcsynkwgjx7fipm5pual6eexax4  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2025-04-20  | 9175     | ~3189               | 2316                  | 3198                  | bafybeia2uv3ea3aoz27ytiwbyudrjzblfuen47hm6tyfrjt6dgf6iadta4  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+
| 2025-07-04  | 9790     | ~3394               | 2444                  | 3406                  | bafybeidfxayyacq4jbbhjcxbxumqlspmsmkj52nq2ns4vfew5udgysocoy  |
+-------------+----------+---------------------+-----------------------+-----------------------+--------------------------------------------------------------+



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


Update 2024-03-31: Recent results from model ``shitspotter_from_v027_halfres_v028-epoch=0179-step=000720-val_loss=0.005.ckpt.pt`` have been quite good. These have quantiatively been measured against the ``vali_imgs228_20928c8c.kwcoco.zip`` variant of the validation dataset. The precision recall and ROC curves for pixelwise binary poop/no-poop classification are:


.. image:: https://i.imgur.com/rgGjAda.png

And the corresponding threshold versus F1, G1, and MCC is:

.. image:: https://i.imgur.com/vay6TEP.png

Qualitatively some cherry-picked success cases in challenging images look like:


.. image:: https://i.imgur.com/oWPg4CE.jpeg

There still are false positives and false negatives in some of the more
challenging images, but the algorithm is now accurate enough where it can be
used, and it will continue to improve.


Data Management
===============

The full resolution dataset is public and hosted on IPFS.

Despite the name, this is not yet a DVC repo.  Eventually I would like to host
the data via DVC + IPFS, but fsspec needs a mature IPFS filesystem
implementation first. I may also look into git-annex as an alternative to DVC.

The licence for the software will be Apache 2. The license for the data is
"Creative Commons Attribution 4.0 International".

In addition to these licenses please:

* Cite the work if you use it.
* If you annotate any of the images, contribute the annotations back. Picking up shit is a team effort.
* When asked to build something, particularly ML systems, think about the ethical implications, and act ethically.
* Pin the dataset on IPFS or seed it on BitTorrent if you can.

Otherwise the data is free to use commercially or otherwise.

The URL that can be viewed in a web browser: https://ipfs.io/ipfs/bafybeigovcysmghsyab6ia3raycsebbc32kea2k4qoxcsujmp52hzpsghy

Current IPFS addresses for each top-level asset group are:

.. temp



.. code::

    bafybeidvihxq3wlaeymbxeeogefvmvcyaz6cjhshnrcd4zqa4ndogxx2n4 - shitspotter_dvc/assets/_contributions
    bafybeifmudpqd7hyc3ahzq6onjjcbkfddsolgndhycdnc6g3ah452uonpm - shitspotter_dvc/assets/_horse-poop-2022-05-26
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
    bafybeicxiansxev6cipgp4lyykcfwregg3zlzlz2w4udpiggoyig7fsq3i - shitspotter_dvc/assets/poop-2024-03-30-T213537
    bafybeia4cjh42u6wa3eykb5kow3qpvh5otae34ksbs7t6t2xs7nnrzwrly - shitspotter_dvc/assets/poop-2024-05-21-T133127
    bafybeie4xnm4ba2nevrouz3drn5oanl4a34lxfxek743wyspwz4drone3i - shitspotter_dvc/assets/poop-2024-06-15-T163943
    bafybeieg7n6rkrdudzsyqe3e4kanvscdk7qyd3sf5qubwvldfung2cozh4 - shitspotter_dvc/assets/poop-2024-07-03-T144034
    bafybeihghjiuil27tzk3td43d6y44liivi4q3jemmp3c2vpfm4zirikoke - shitspotter_dvc/assets/poop-2024-09-16-T130352
    bafybeibwijhponhdmw5wixkm5tvptmbh2vnusjnxhd7qch7mplnyk2hbzq - shitspotter_dvc/assets/poop-2024-10-16-T223026
    bafybeiallyvshbkuxlnjp4nlc4tk5mlphqejp27vv6ewekpwpzbebavcf4 - shitspotter_dvc/assets/poop-2024-11-22-T195205
    bafybeihgnfs6hku3xlqa7fnoqujkmq2ezk7lvidw32dvgjhsmk7wpk72cy - shitspotter_dvc/assets/poop-2024-12-30-T212347
    bafybeidvaphwcib2qezdcey4cj3a2r7r7oxskl56yaccgdi75pdou4ggmm - shitspotter_dvc/assets/poop-2025-03-08-T224918
    bafybeieyl6yzi6cyz3minjyvmz53ydbpxmljxs5gytv6cwu6ci7tmwyvjq - shitspotter_dvc/assets/poop-2025-04-20-T172113
    bafybeicz7kxvmxojmu33pskfrkglosz3tndgsmpzz6cmmcxynaau5xzfeu - shitspotter_dvc/assets/poop-2025-07-03-T230656



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


Honey - (~2013 - ) - Adopted in June 2015, Honey is often called out for her
resemblance to a fox and is notable for her eagerness for attention and
outgoing personality.  DNA analysis indicates that she is part Boxer, Beagle,
German Shepherd, and Golden Retriever.  Honey's likes include: breakfast,
sniffing stinky things, digging holes, sleeping on soft things, viciously
shaking small furry objects, and whining for absolutely no reason.  Honey's
dislikes include: baths, loud noises, phone calls, and arguments.  Honey came
to us from Ohio as a fearful dog, but has always been open to trusting new
people.  She has grown into an intelligent and willful dog with a scrappy
personality.

.. An Image of Honey
.. image:: https://i.imgur.com/gUzwgCT.jpg
   :height: 400px
   :align: left
.. bafybeihuhrp6wtle5wuhsgcgf6bp7w4ol4pft7y2pcplylzly7gfag74lm bafybeic5a4kjrb37tdmc6pzlpcxe2x6hc4kggemnqm2mcdu4tmrzvir6vm/Contributor-Honey.jpg


Bezoar - (~2018 - 2023-10-19) - Adopted in July 2020 and named for a
`calcified hairball <https://en.wikipedia.org/wiki/Bezoar>`_, Bezoar was an
awkward and shy dog, but grew into a curious and loving sweetheart.  Her DNA
test indicated she was part Stafford Terrier, Cane Corso, Labrador Retriever,
German Shepherd, and Rhodesian Ridgeback.  Bezoar's likes included: breakfast, a
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



Roadie - (2016-04-29 - ) - Adopted in December 2023, Roadie is an energetic
blue heeler who is not afraid to voice his opinions. His DNA test indicates he
is 60% Australian Cattle Dog mixed with 20% Border Collie and small percents of
Husky and Spaniel.  Roadie's likes include: fetching the ball, getting
different people to throw the ball, dropping the ball and picking it back up
before someone can take it, staring deeply into eyes, pets, and invading
personal space. Did I mention he likes the ball? Roadie's dislikes include:
dropping the ball, steep staircases, and spinach. Roadie was originally from
Texas, but came to us after his aging owners could no longer take care of him.
Thusfar he has proven an excellent contributor to this project, pooping far
more frequently than the other dogs and in novel locations that bolster dataset
diversity.

.. An Image of Roadie
.. image:: https://i.imgur.com/DYdkt75.jpeg
   :height: 400px
   :align: left

.. .. An Image of Roadie
.. .. image:: https://i.imgur.com/yaZi5bO.jpg
..   :height: 400px
..   :align: left

Contributing
============

Please contribute! The quickest way is with the `Google Form for ShitSpotter Image Contributions <https://docs.google.com/forms/d/e/1FAIpQLSfqH1555hynVUwh0O0526svPOaS0NnWiR15n68sgr7DExB6TQ/viewform?usp=sf_link>`_.

Alternatively, you can send me an image via email to: ``crall.vision@gmail.com``.

When you contribute an image:

* Make sure you are ok with it being released for free under: `(CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`_
* Let me know how to give you credit.
* Let me know if you want time / GPS camera metadata to be removed from the images.

Guide to taking an image:

Upload an image with poop in it. The poop need not be centered in the image. It could be close up, or far away. It should be visible, but it need not be obvious. The idea is that it could be difficult to see and we want to test if a machine learning algorithm can find it. The only requirement is that if a human looks at it carefully, they can tell there is poop in it.



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
