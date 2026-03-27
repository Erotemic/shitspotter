ScatSpotter Presentation Outline (10 minutes)

* 0. Blunt Opening: (0:30–0:45)

    * <state exactly what I did>: I've been taking pictures of dog poop for the past 5 years.
    * <Greet audience>: Hi.
    * <Introduce self>: I'm Jon Crall.
    * <state high level takeaway>: And I drew polygons on them.
    * <state high level takeaway>: And I trained models on those.
    * <state high level takeaway>: And measured how good I could do on a test and validation dataset.
    * <state high level takeaway>: And I uploaded it to the internet.
    * <joke-but-also-real>: And I wasn't paid to do it, I actually had to pay to be here.
    * This is ScatSpotter: a high-res dataset, baselines that reach about 0.70 Box AP when tuned, and foundational zero-shot doesn’t solve it.

* 1. Main Contributions: (0:30–0:45)

    Goal: set the table; no details yet.

    Dataset: high-res phone images + polygon labels + BAN collection protocol

    Baselines: ViT, YOLO, MaskRCNN, Grounding Dino foundation detector (zero-shot vs tuned)

    Distribution study: centralized vs decentralized, with reproducibility-minded identifiers

* 2. Motivation (0:45–1:00)

    * story of why I started doing this

    * Why it’s hard (one sentence): small + amorphous + confusers + lighting

    * Why it matters: this is a proxy for “waste in the wild” / small-object cluttered detection

    * self-funded
        - This is why a few techniques may feel out of date, I only work on this sporatically 
        - If you want to fund this work or related work (general AI for waste in vision), lets talk.

    * where I want this to go: free phone app

    * <goal: convince the audience this problem is worth studying> 

* 3. Dataset Description (2:30–3:00)

    * Talk about what the dataset is 
    * Talk about how I built it
    * Talk about my methodology
    * Talk about if there are errors / what is annotated known
    * Talk about potential applications of the dataset beyond what I want

    * Brief points:
        * privacy/ethics
        * train / test / validation splits

    * Justify test dataset size, discuss roboflow

    * <goal: understand what I did and what I made>

        3.1 What the dataset is

        Single-class polygons in full-res images (phone camera scale)

        Key shape: mostly verified negatives (because of BAN)

        3.2 How you built it (the “method” slide)

        BAN protocol: Before / After / Negative

        Before = positive

        After = counterfactual negative from the same viewpoint

        Negative = nearby confuser scene

        Why BAN matters:

        scales “known negatives”

        creates hard negatives

        enables future change-detection/contrastive angles

        3.3 What’s annotated / errors / caveats (keep it short and confident)

        Polygons are human-verified (some SAM-assisted initialization)

        Confuser labels exist but are not exhaustive (don’t overpromise)

        Known issue class: occasional missed instances; future cleaning via better models

        3.4 Splits + test set size (20–30 seconds max)

        Val: held-out author images

        Test: photographer-disjoint contributor set (small but realistic generalization)

        Roboflow mention (one sentence):

        “External sources exist, but QC/licensing/consistency make them tricky as a benchmark—so this test is curated and clean.”

* 4. Trained Models & Observations (2:30–3:00)

    * Starting with VIT

    * YOLO-v9 Model

    * DINO-v2 Zero Shot vs Tuned performance (and why I'm not using this in my annotation pipeline yet)

    * <goal: convey comparative performance>

        Goal: convey comparative performance with one plot and one qualitative grid.

        4.1 Setup (one sentence)

        “We compared a segmenter, classical detectors, and a foundation detector.”

        4.2 What to highlight (don’t list every model equally)

        Pretraining helps across families

        Tuned foundation detector is best

        Zero-shot is not enough

        4.3 Your model story (clean ordering)

        ViT-sseg: why you tried it (dense masks → boxes for metrics)

        YOLO-v9: strong baseline; pretrained vs scratch matters

        GroundingDINO: zero-shot vs tuned gap is the key insight

        Include your reason for not using it in annotation yet:

        “Prompt sensitivity + workflow/engineering time + wanting stability in the pipeline”

        (This feels practical, not defensive.)

        4.4 Qualitative failure modes (30 seconds)

        Show 2–3 examples: leaf/stick confusers, low light, closeups

        “These are the cases the phone app needs to handle.”

* 5. Dataset distribution & Observations (1:15–1:30)

    * IPFS is motivated by the fact that I'm unfunded. I have very limited institutional support on this.

    * Convince people that dataset size in GB is interesting, people who don't manage these sort of models / datasets on a day-to-day don't understand the importance of the file size. 

    * Discuss scaling of such a dataset.

    * Mention that IPFS seems to have gotten better since I did the experiments. Peer finding is much less of a problem

    * <goal: Convince them IPFS content addressed data is interesting.>

    Why distribution is part of the work:

    “If we want open benchmarks, we need reliable ways to share tens-of-GB datasets.”

    Centralized vs decentralized framing:

    Centralized is fast (CDNs), but URLs rot / change

    Content addressing gives integrity and long-term reference

    “Self-funded projects benefit from low-maintenance, verifiable distribution.”

    “At this size you start to feel friction: storage, transfer time, versioning, reproducibility.”

    “IPFS has improved since these experiments—peer discovery is less painful now.”

    “CIDs are citations (dois) for dataset bits: if the bits change, the ID changes.”

* 6. Future (0:45–1:00)

    * Discuss the phone app
    * Discuss future foundational models SAM3 Dino-V3

    ---

    Goal: end with a direction and an ask.

    Phone app path:

    what needs to be true (robust in clutter/low light, low false positives, mobile constraints)

    Data:

    grow contributor test set (more phones, cities, seasons)

    Models:

    revisit with next-gen foundation models (SAM3 / DINOv3, etc.)

    emphasize “small-object + hard negative” methods

    Call to action:

    contribute data / try new models / fund “vision for waste in the wild”

    “Thanks. I’m looking for collaborators who can sponsor waste-related vision
    work through Kitware—think a short starter sprint—with open-source
    deliverables. If your organization funds pilots like that, I’d love to talk
    after the session.”
