# Datasheet: ShitSpotter

Template: [JRMeyer/markdown-datasheet-for-datasets](https://github.com/JRMeyer/markdown-datasheet-for-datasets) based on [Datasheets for Datasets by Gebru et al](https://arxiv.org/abs/1803.09010).

Author: Jon Crall

Organization: Kitware


## Motivation

*The questions in this section are primarily intended to encourage dataset creators to clearly articulate their reasons for creating the dataset and to promote transparency about funding interests.*

1. **For what purpose was the dataset created?** Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

	There are several reasons. 

    1. To provide a training / validation data for poop detection / segmentation networks. 

    2. To experiment with distribution of datasets over IPFS.

    3. To provide an interesting, challenge, and humorous problem / benchmark.

2. **Who created this dataset (e.g. which team, research group) and on behalf of which entity (e.g. company, institution, organization)**?

	The primary author - Jon Crall - collected most images. 
    Several contributions have been made by other people: friends, familly, acquaintances, colleagues. 

    There is no entity this is on behalf of.

3. **What support was needed to make this dataset?** (e.g. who funded the creation of the dataset? If there is an associated grant, provide the name of the grantor and the grant name and number, or if it was supported by a company or government agency, give those details.)

    There is currently no funding. All effort is based on volunteer time.

4. **Any other comments?**

    The main reason Jon Crall started collecting the dataset was because 
    he wasn't able to find where his dogs in the fall because of all the leafs.

    See the README.


## Composition

*Dataset creators should read through the questions in this section prior to any data collection and then provide answers once collection is complete. Most of these questions are intended to provide dataset consumers with the information they need to make informed decisions about using the dataset for specific tasks. The answers to some of these questions reveal information about compliance with the EU’s General Data Protection Regulation (GDPR) or comparable regulations in other jurisdictions.*

1. **What do the instances that comprise the dataset represent (e.g. documents, photos, people, countries)?** Are there multiple types of instances (e.g. movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

    The exact answer to this question may change as the dataset grows.

    Mostly dog feces. Feces from other species are represented, but not well
    supported by labels yet. Specifically, there are images of horse and racoon
    poop. It is possible that some of the images labeled as dog poop are
    actually feces from another species. If another animal pooped in 
    a public area and a contributor came across it, it may be mistaken for 
    dog poop. It would be interesting of some of these cases could be
    identified.

2. **How many instances are there in total (of each type, if appropriate)?**

    The answer to this will change as the dataset grows. Thus, it is important
    to answer this question programatically. We provide the command and a recent 
    output with a timestamp of its generation.

```
cd $HOME/code/shitspotter/shitspotter_dvc
date
kwcoco stats data.kwcoco.json 

  n_anns  n_cats  n_imgs  n_tracks  n_videos
    4386       3    6648         0         0

```

1986 Poop Annotations. @ Sun Dec 10 07:20:05 PM EST 2023


3. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g. geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g. to cover a more diverse range of instances, because instances were withheld or unavailable).

	No, it is a sample of stool, mostly from Albany, NY. 

4. **What data does each instance consist of?** "Raw" data (e.g. unprocessed text or images) or features? In either case, please provide a description.

	Each annotation is a polygon

5. **Is there a label or target associated with each instance?** If so, please provide a description.

	Currently, polygons are only labeled as poop. This may change in the future.

6. **Is any information missing from individual instances?** If so, please provide a description, explaining why this information is missing (e.g. because it was unavailable). This does not include intentionally removed information, but might include, e.g. redacted text.

    Yes, the identity of the dog that pooped was not recorded - and sometimes
    is unavailable. 
    

7. **Are relationships between individual instances made explicit (e.g. users' movie ratings, social network links)?** If so, please describe how these relationships are made explicit.

	No. I don't think that applies here.

8. **Are there recommended data splits (e.g. training, development/validation, testing)?** If so, please provide a description of these splits, explaining the rationale behind them.

    Yes, we are currently suggesting that data from 2021, 2022, 2023 are in the
    training set. Data from 2020 is used for validation. Data from 2024 is
    split between training and validation. On the nth day of the year, images
    are in the validation set if n % 3 == 0 else they are in the training set.

9. **Are there any errors, sources of noise, or redundancies in the dataset?** If so, please provide a description.

	Yes, some images were not taken according to the "before,after,negative" protocol.

10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g. websites, tweets, other datasets)?** If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g. licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

	Yes, it is a self-contained dataset.

11. **Does the dataset contain data that might be considered confidential (e.g. data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** If so, please provide a description.

	No, I don't think so.

12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** If so, please describe why.

	Yes. Some people might find poop offensive, and viewing it may cause anxiety.

13. **Does the dataset relate to people?** If not, you may skip the remaining questions in this section.

	Mostly no. Sometimes people do appear in photos incidentally. 

14. **Does the dataset identify any subpopulations (e.g. by age, gender)?** If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

    No.

15. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** If so, please describe how.

    Possibly, but it would be very difficult.

16. **Does the dataset contain data that might be considered sensitive in any way (e.g. data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** If so, please provide a description.

    No. I don't think that images of poop qualify as sensitive. 

17. **Any other comments?**

    Everybody poops.


## Collection

*As with the previous section, dataset creators should read through these questions prior to any data collection to flag potential issues and then provide answers once collection is complete. In addition to the goals of the prior section, the answers to questions here may provide information that allow others to reconstruct the dataset without access to it.*

1. **How was the data associated with each instance acquired?** Was the data directly observable (e.g. raw text, movie ratings), reported by subjects (e.g. survey responses), or indirectly inferred/derived from other data (e.g. part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.

	Images were labeled with with both manual and AI-assited (SegmentAnythingModel) polygons.

2. **What mechanisms or procedures were used to collect the data (e.g. hardware apparatus or sensor, manual human curation, software program, software API)?** How were these mechanisms or procedures validated?

	A phone camera (details are recorded in metadata) and the LabelMe segmentation tool. 

3. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g. deterministic, probabilistic with specific sampling probabilities)?**

	It is a subset of the set of all possible images of dog poop. It is not a subset or generated from some other dataset, these are all original phone images taken for the purpose of constructing this dataset.

4. **Who was involved in the data collection process (e.g. students, crowdworkers, contractors) and how were they compensated (e.g. how much were crowdworkers paid)?**

	All work was volunteer work.

5. **Over what timeframe was the data collected?** Does this timeframe match the creation timeframe of the data associated with the instances (e.g. recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. Finally, list when the dataset was first published.

    Work started on 2020-11-12. Collection is ongoing. The git repo was launched on 2021-11-11.

7. **Were any ethical review processes conducted (e.g. by an institutional review board)?** If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.

	No. This project started organically.

8. **Does the dataset relate to people?** If not, you may skip the remainder of the questions in this section.

	Mostly no. Some people do appear in the images.

9. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g. websites)?**

	No

10. **Were the individuals in question notified about the data collection?** If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.

	No

11. **Did the individuals in question consent to the collection and use of their data?** If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

	No

12. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).

	Na

13. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g. a data protection impact analysis) been conducted?** If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

	No

14. **Any other comments?**

	No


## Preprocessing / Cleaning / Labeling

*Dataset creators should read through these questions prior to any pre-processing, cleaning, or labeling and then provide answers once these tasks are complete. The questions in this section are intended to provide dataset consumers with the information they need to determine whether the “raw” data has been processed in ways that are compatible with their chosen tasks. For example, text that has been converted into a “bag-of-words” is not suitable for tasks involving word order.*

1. **Was any preprocessing/cleaning/labeling of the data done (e.g. discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** If so, please provide a description. If not, you may skip the remainder of the questions in this section.

    All data is provided as recieved. 

2. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g. to support unanticipated future uses)?** If so, please provide a link or other access point to the "raw" data.

    Yes, the data is stored in its original form as given by the phone.

3. **Is the software used to preprocess/clean/label the instances available?** If so, please provide a link or other access point.

    Software on the phone may include post processing. I'm unaware of what these methods are.

4. **Any other comments?**

    Several attributes like precomputed homographies between the "before/after" images are
    provided in the IPFS distribution as a lightweight cache, and the shitspotter codebase 
    contains 


## Uses

*These questions are intended to encourage dataset creators to reflect on the tasks  for  which  the  dataset  should  and  should  not  be  used.  By  explicitly highlighting these tasks, dataset creators can help dataset consumers to make informed decisions, thereby avoiding potential risks or harms.*

1. **Has the dataset been used for any tasks already?** If so, please provide a description.

	No

2. **Is there a repository that links to any or all papers or systems that use the dataset?** If so, please provide a link or other access point.

    Currently there are none that I know of, but the main README will be
    updated with this information: https://github.com/Erotemic/shitspotter

3. **What (other) tasks could the dataset be used for?**

    There is a potential to classify different types of poop from different
    species.  Images contain other content such as local park scenery, grass,
    leafs. Additional annotations could be placed on those objets for other tasks.

4. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g. stereotyping, quality of service issues) or other undesirable harms (e.g. financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?

    There is a bias towards poops from certain individuals. There is a long
    tailed distribution of identity of the pooper. Some poops are older than
    others, that distribution is unlabeled, but a human annotator may be able
    to guess the ages (or individual animal in some cases). It's also not 100%
    clear if all images are dog poop. Most certainly are, but some may not be. 

5. **Are there tasks for which the dataset should not be used?** If so, please provide a description.

	Nothing comes to mind.

6. **Any other comments?**

    The motivating use case is to build a phone application that can help dog
    owners find lost poops. There are several other use cases I can imagine, some
    more elaborate than others.

    * Automatic waste cleanup robots
    * "Smart glasses" augmented reality to warn you before you step in poop.
    * As suplemental data for species identification from images of feces.


## Distribution

*Dataset creators should provide answers to these questions prior to distributing the dataset either internally within the entity on behalf of which the dataset was created or externally to third parties.*

1. **Will the dataset be distributed to third parties outside of the entity (e.g. company, institution, organization) on behalf of which the dataset was created?** If so, please provide a description.

    It will be freely available as long as someone is willing to host it.

2. **How will the dataset will be distributed (e.g. tarball on website, API, GitHub)?** Does the dataset have a digital object identifier (DOI)?

	No DOI yet. It is being made available via IPFS, BitTorrent, and centralized means.

3. **When will the dataset be distributed?**

	I update about once a month.

4. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

    All data is free to use under "Creative Commons Attribution 4.0 International".

5. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

    No

6. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

	No

7. **Any other comments?**

	No


## Maintenance

*As with the previous section, dataset creators should provide answers to these questions prior to distributing the dataset. These questions are intended to encourage dataset creators to plan for dataset maintenance and communicate this plan with dataset consumers.*

1. **Who is supporting/hosting/maintaining the dataset?**

    Currently, the main author hosts an IPFS server. Their employer's IPFS
    server also pins the information, and other entities may be pinning it.

2. **How can the owner/curator/manager of the dataset be contacted (e.g. email address)?**

	Github Issue for this project.

3. **Is there an erratum?** If so, please provide a link or other access point.

	No.

4. **Will the dataset be updated (e.g. to correct labeling errors, add new instances, delete instances)?** If so, please describe how often, by whom, and how updates will be communicated to users (e.g. mailing list, GitHub)?

	Roughly monthly with a push to github.

5. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g. were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** If so, please describe these limits and explain how they will be enforced.

	Na.

6. **Will older versions of the dataset continue to be supported/hosted/maintained?** If so, please describe how. If not, please describe how its obsolescence will be communicated to users.

    Possibly, as long as the IPFS CIDs remain alive. At the time of writing all
    versions of the dataset should still be available. 

7. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.

    Yes. Take it and do what you want: ideally something cool and good. It
    would be nice to throw us a citation though.

8. **Any other comments?**

    No
