# SITE - Serial Imaging of the Tumor and microEnvironment
_Authors: Vaibhav Murthy, Jeremy Copperman, Alexander Davies_

The Serial Imaging of Tumour and microEnvironment (SITE) platform is a system designed for live-cell, ex vivo imaging of cancer cell dynamics within their native tissue architecture and integrates organoid–tissue explant cultures with fluorescent signalling biosensors and advanced live-cell microscopy, enabling simultaneous measurement of cell behaviours, physical cell–cell interactions, and quantitative signalling activity.

The experimental framework is paired with a computational pipeline that extracts high-precision single-cell time-series data and analyzes cell fate and signalling trajectories within complex tissue microenvironments. Together, these components form a fully integrated experimental and computational platform for studying cancer cell dynamics in physiologically relevant contexts.

To showcase the capabilities of the SITE platform, we used it to experimentally model basal-like breast cancer in both primary and metastatic tissue environments. We developed tissue-specific variants of SITE—MammaSITE for the mammary gland and LungSITE for the lung—to study tumour cell signalling and tumour–host interactions within their respective physiological contexts. Using basal-like breast cancer cells expressing an ERK biosensor, we tracked signalling dynamics across these distinct tissue settings over time. This allowed us to investigate early tumour initiation in the mammary gland, the processes underlying dissemination to the lung, and the coordinated evolution of signalling and tumour–host interactions in both models.

We describe the development and implementation of the SITE platform in our paper: **Serial Imaging of Tumor and microEnvironment (SITE) platform for live-cell ex vivo modeling of primary and metastatic cancer dynamics**  
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.09.08.674915v1)

![SITE_Pipeline_Complete_Figure](https://github.com/user-attachments/assets/82dc0812-0cd1-4f6c-9840-a99c5b6ad67a)


## Advantages of SITE

The SITE platform offers several key advantages for studying cancer cell behaviour in tissue environments:

-  **Native tissue context:** Preserves the structural, mechanical, and cellular complexity of the tumour microenvironment, enabling more physiologically relevant observations than traditional 2D or spheroid culture systems.

-  **Live, long-term imaging:** Supports continuous ex vivo imaging to capture dynamic processes such as signalling fluctuations, cell fate changes, migration, and tumour–host interactions over extended timescales.

-  **Integrated signalling readouts:** Combines tissue explant culture with fluorescent biosensors to report real-time activity of pathways like ERK at single-cell resolution.

-  **Multi-modal data fusion:** Couples advanced microscopy with a computational pipeline that extracts high-precision single-cell trajectories from complex 3D tissues.

-  **End-to-end experimental + computational workflow:** Provides a unified system for generating, quantifying, and analysing dynamic cancer cell behaviour within organ-specific microenvironments, facilitating deeper mechanistic insight and hypothesis generation.


## Usage of SITE 

The following vignettes contain the explanation on how to perform the basic steps of extraction of features from ND2 files using SITE. 
-  [SITE Image Processing and Feature Extraction](./Image_processing_and_feature_extraction/SITE_Image_Processing_and_Feature_Extraction.ipynb)

Following completion of the SITE Image Processing and Feature Extraction pipeline, the data can be used to derive trajectories, extract experimental metadata and high-dimensional single-cell features, ERK C/N signaling dynamics and model spatiotemporal coordination and boundary libraries and contact-niche states and can be found below. 
- [SITE Analysis pipeline](./Analysis_pipeline/SITE_Analysis_pipeline.ipynb)

## References
- **Serial Imaging of Tumor and microEnvironment (SITE) platform for live-cell ex vivo modeling of primary and metastatic cancer dynamics**. Vaibhav Murthy, Rawan Makkawi, Francis Anderson, Carol Halsey, Elise Manalo-Hall, Sanjay Srikanth, Gabriella Tangkilisan, Ting Zheng, James McGann, Marc R. Birtwistle, Jeremy Copperman, Alexander E. Davies. bioRxiv 2025.09.08.674915; doi: https://doi.org/10.1101/2025.09.08.674915

SITE Pipeline was built upon and using the [celltraj](https://github.com/cancer-dynamics/celltraj) toolset 
