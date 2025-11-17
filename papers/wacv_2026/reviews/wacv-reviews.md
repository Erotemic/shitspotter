
Meta Review Round 2 of Submission2595 by Area Chair aPn3
Meta Review Round 2by Area Chair aPn301 Nov 2025, 14:56 (modified: 09 Nov 2025, 01:09)Authors, Reviewers Submitted, Area Chairs, Program ChairsRevisions
Metareview:
This paper was reviewed by 3 experienced researchers and received ratings of R, WR, WA (Reject, Weak Reject, Weak Accept). Some concerns included: limited novelty (the area chairs didn't focus on this, it's an application paper, though maybe emphasizing that could be helpful, e.g. last comment in major weaknesses from 1mTn); inconsistencies in evaluation; limited impact to the field (generalization unclear); and questions about data collection methodology.

A prevailing concern was the limited variety in the images collected: similar state (mostly fresh), limited number of subjects (most from 3 specific dogs), clearly limited # of species (not clear how many, but mostly from 3 individuals), limited geographically (mostly urban, single city), limited in scale ("relatively small" from favorable reviewer MGwr).

The primary motivation being a phone app to assist dog owners in cleaning up after their dogs seems of questionable utility -- presumably the owners can see it themselves? Applications like automatic/robotic clean up or wildlife research feel more compelling, but this dataset doesn't feel particularly well-suited to those. In the area chair panel's discussion, additional questions such as "why limit to just dog scat?" arose.

The area chair panel recommends that this paper be rejected, but do want to recognize the efforts of the authors. There are parts already that could be integrated with more work into a solid paper. Moreover, WACV is a good venue for a dataset like this. However, the authors need to more effectively convince their audience of what compelling problems it addresses or is useful for and probably dramatically increase the variety of the data collected. Authors are encouraged to closely consider all the reviewers' feedback in revising their paper.

Final Recommendation: Reject in round 2


----

Official review
Official Review Round 2by Reviewer 1mTn21 Oct 2025, 20:26 (modified: 09 Nov 2025, 01:09)Program Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 1mTnRevisions
Paper Summary:
This paper introduces a new dataset designed for animal feces detection, specifically focusing on dog feces. The main contribution lies in the creation and presentation of this dataset. The paper is clearly written and easy to follow, and the authors provide a detailed description of the dataset construction process. The procedure is lengthy and labor-intensive, the effort invested in developing this dataset is commendable. The dataset has 9,296 images and 6,594 polygon masks, collected using a “before/after/negative” (BAN) protocol to supply realistic hard negatives.

Paper Strengths:
The paper provides a well-structured and easy-to-understand description of the dataset creation process.

The dataset is diverse and includes a substantial number of annotated images.

The authors also evaluate several existing models on the proposed dataset and compare their results using different methods, which adds value to the work.

Major Weaknesses:
The authors do not provide sufficient details about the accuracy and reliability of the annotations. It remains unclear how the annotation process was conducted . For example, how many annotators were involved, what validation steps were taken, or how consistency between annotators was ensured. The paper does not explain how the correctness of the annotations was verified, which raises concerns about the dataset’s quality and reliability.

Scene Diversity and Capture Method: The paper does not clarify whether multiple images were captured from different angles of the same scene (if not why?). This limits research on view‑point robustness. Additionally, the data is mostly one city, mostly one phone, and many images are from three specific dogs which introduces bias e.g. device‑/locale‑specific.

Although the dataset contribution can be useful, the paper does not introduce any new methodological innovation or solution for the feces detection problem. The work is primarily focused on dataset creation, without proposing novel detection algorithms or improvements to existing ones.

PErhaps the biggest problem is the inconsistent baseline settings used. VIT‑seg used full resolution, while Mask R‑CNN, DINO, and YOLO used resized inputs. In addition, YOLO and DINO lack pixel‑level scores since they don’t output masks. This makes method‑to‑method comparisons less clean.

There is no new detection method. The novelty is the dataset and a practical distribution study, not algorithms; this is fine for an applications/data track, but should be clearly stated.

Minor Weaknesses:
Please see above

Round 2 Recommendation: 1: Reject
Round 2 Justification:
Reject, for the reasons explained above.

Confidence Level: 4 - High Confidence: The reviewer has strong expertise in the area. They are highly familiar with the relevant literature and can critically evaluate the paper.


-----


Review of ScatSpotter
Official Review Round 2by Reviewer dK8b21 Oct 2025, 08:01 (modified: 09 Nov 2025, 01:09)Program Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer dK8bRevisions
Paper Summary:
This paper presents a dataset focused on the detection and segmentation of dog poop in images of outdoor scenes. The authors benchmark several standard object detection and segmentation models (Mask R-CNN, YOLOv9, Grounding DINO, and ViT-sseg) on the dataset to establish baseline performance. The authors also benchmark the efficiency of centralized versus decentralized dataset distribution mechanisms (Hugging Face, IPFS, and BitTorrent) in order to address data accessibility concerns.

While the dataset is carefully annotated and publicly accessible, and while I acknowledge that considerable effort went into collecting the data, annotating them, and running baselines, I feel that the task itself is standard (object detection/segmentation) and limited to a single class (“dog poop”), which restricts generalizability and research impact. The paper fits the Applications track thematically, but its overall contribution is somewhat marginal in novelty and scope.

Paper Strengths:
I believe that the paper strengths are the following:

The dataset appears well organized, documented, and reproducible, with public hosting and code references leaving no doubt regarding reproducibility of experiments.
Several baselines both from classic convolutional-based detection methods as well as transformer-based segmentation methods and even zero-shot foundation models have been established on the dataset.
The discussion on dataset distribution (centralized vs decentralized) is somewhat novel and relevant to open-science and reproducibility concerns, although a bit tangentially related to a computer vision conference theme.
The writing and experimental methodology are generally clear, and the supplementary provides detailed information on aspects of data collection and baseline experiments.
Major Weaknesses:
I believe that the paper major weaknesses are the following:

The research novelty is very limited. The task itself is standard, the baselines are also standard, there is no attempt to improve the performance of baselines, and the domain of the data (dog poop) is very narrow.
The impact is low. The dataset contains just a single object class, which, granted, can help in certain applications, but does not seem like it will open new research directions or address an existing open problem in computer vision. This may also be seen by looking at the related work section of the paper, in which I could not locate a prior published paper on this particular data domain, except a couple of open data competitions. Openly distributing such data may be useful e.g. for a kaggle competition, but I don't think this necessarily means that a peer reviewed paper can be published on this.
I have concerns regarding the data collection methodology (e.g. what does the BAN protocol actually offer here is not really evaluated anywhere), the fact that the test split is very small, the somewhat arbitrary way in which splits were created (why by year?) the fact that most data comes from the same source (some dog/s and area/s), and there is some confusion regarding what version of the dataset was actually used for the experiments.
The motivation and the data domain risk being perceives as trivial or humorous, which can undermine its suitability for WACV.
Minor Weaknesses:
I don't have many minor weaknesses to list, but the following stand out for me:

The dataset name and framing might benefit from more formal language to avoid unintended humor.

Figure 4 has some problems. In 4a) there is a nan in the legend. In 4b) it seems that the top part of the plot was cropped out and reattached due to the y axis range. Perhaps use log scale?

There is some ambiguity regarding if the task is semantic segmentation or object detection. Polygonal masks are mentioned as annotation but most methods are object detection methods. Perhaps split into two tracks and evaluate more methods (there are several, including foundation ones like SAM and follow-ups) on the semantic segmentation track ?

Round 2 Recommendation: 2: Weak Reject
Round 2 Justification:
I think that while the paper appears technically sound and the dataset itself is well-documented, the paper’s contribution is marginal in both novelty and impact. The dataset addresses a very narrow, single-class domain with limited general applicability. The exploration of decentralized data distribution is mildly interesting but I don't think it is enough on its own.

If the authors wish to strengthen the work, they could reframe the dataset in a broader scientific context (e.g., bio-waste detection, outdoor contamination monitoring), add additional classes or tasks, and clarify the segmentation objectives. As it stands, I believe the paper’s novelty and significance fall below the acceptance threshold for WACV.

Confidence Level: 4 - High Confidence: The reviewer has strong expertise in the area. They are highly familiar with the relevant literature and can critically evaluate the paper.


----

``ScatSpotter'' --- A Dog Poop Detection Dataset
Official Review Round 2by Reviewer MGwr19 Oct 2025, 15:52 (modified: 09 Nov 2025, 01:09)Program Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer MGwrRevisions
Paper Summary:
The paper proposes a dataset for dog poop detection, including 9000 images and 6000 annotated polygons containing poop. The dog poop can be difficult to detect, as it is often small relative to the scene and can be visually variable, making it a challenge. The paper is well written and the dataset curation and design decisions are clearly stated.

Paper Strengths:
This appears to be the largest current dog-poop-focused dataset, and the authors do a good job of motivating why dog poop detection might be of interest in different applications.

Figure 2 is quite interesting, it definitely does a nice job in emphasizing the distributioal differences across these datasets.

The paper includes several useful analyses of the dataset, which helps readers get a sense of how the data is distributed over time, visual similarity, and positive/negative

The protocol for splitting the data is robust and removes much of the potential for overfitting.

Major Weaknesses:
The dataset is still relatively small, and was primarily captured from one city with (presumably) a bias towards the specific poop of the researcher’s dog. Additionally, there is likely photographer bias in the data (as can be seen in Figure 2, mostly centered and larger objects are more centered). Ad a result, it isn’t clear how generalizeable results on this data might be to some of the applications metioned, e.g. automated cleanup or wildlife research.

Minor Weaknesses:
Overall, the paper is easy to read, no minor weaknesses

Round 2 Recommendation: 5: Weak Accept
Round 2 Justification:
The dataset is novel and complementary to past datasets, and despite being relatively small, presents an interesting challenge for CV models (small, often camouflaged objects of interest), and is complementary to existing datasets. The work is presented clearly, and testing protocols are robust. I would recommend the paper for weak acceptance.

Confidence Level: 4 - High Confidence: The reviewer has strong expertise in the area. They are highly familiar with the relevant literature and can critically evaluate the paper.
