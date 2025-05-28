# Image to Text Captioning NLP Project

## About

This project presents a comprehensive deep learning pipeline designed to automatically generate scientifically accurate and context-aware textual descriptions for astronomical images, specifically those captured by the Hubble Space Telescope. The system is built around state-of-the-art transformer-based multi-modal models that bridge the gap between visual input and natural language generation. The goal is to support applications in scientific documentation, data cataloging, and educational dissemination by producing human-readable captions aligned with domain-specific knowledge in astronomy.

By combining computer vision, natural language processing (NLP), and vision-language alignment techniques, this project showcases how artificial intelligence can be used to enhance the interpretability and accessibility of large-scale space imagery archives. The models used—BLIP, T5, and CLIP—are strategically chosen to form a modular and extensible system capable of adapting to diverse datasets and scientific domains.

## Project Overview

At its core, this project tackles the challenge of image captioning within a domain-specific scientific context, where general-purpose captioning models often fall short due to hallucinated or vague outputs. The proposed solution is an end-to-end captioning pipeline consisting of three key stages:

1. Caption Generation using a fine-tuned BLIP model with a Vision Transformer (ViT-G/14) encoder. BLIP is responsible for producing initial image captions based on visual features extracted from raw telescope images.

2. Language Refinement using a pretrained T5 transformer model (specifically, google/flan-t5-base). This stage improves the readability, fluency, and scientific clarity of the captions by eliminating generic or incorrect phrases and restructuring outputs in a more human-understandable form.

3. Semantic Reranking using CLIP (Contrastive Language–Image Pre-training). CLIP computes a similarity score between the visual embedding of the input image and each candidate caption. The most semantically aligned caption is selected as the final output.


To further improve scientific accuracy, the pipeline integrates Named Entity Recognition (NER) using spaCy. This step extracts astronomical object identifiers—such as NGC or UGC designations—from expert-written reference texts. These entities, along with metadata fields such as object category and constellation, are injected into caption prompts to guide the model toward more accurate and context-specific outputs.

The system is evaluated using both language generation metrics (BLEU, ROUGE-L, METEOR) and vision-language alignment metrics (CLIP similarity scores). The results show that prompt conditioning, NER, and reranking significantly improve the quality and accuracy of generated captions, making the system well-suited for scientific use.


## Key Components

1. BLIP (Bootstrapped Language Image Pretraining): A multi-modal vision-language model that leverages a ViT encoder to generate initial image captions. It serves as the foundation of the pipeline by mapping visual features to text.

2. T5 (Text-to-Text Transfer Transformer): A large language model trained on instruction-tuned data. In this pipeline, it is used to refine captions produced by BLIP, enhancing grammatical structure, factual correctness, and domain fluency.

3. CLIP (Contrastive Language–Image Pre-training): CLIP evaluates how well a caption semantically aligns with an image. It is used here to rerank multiple candidate captions generated via beam search, selecting the one that best reflects the image content.

4. NER (Named Entity Recognition) and Metadata Injection: spaCy’s NER pipeline is used to extract celestial object names and scientific keywords from existing descriptions. These are embedded into the model prompts to reduce hallucination and anchor the output in real astronomical context.

## Dataset

The primary dataset used in this project is the ESA Hubble Dataset, which is publicly available on the Hugging Face Hub under the identifier Supermaxman/esa-hubble [https://huggingface.co/datasets/Supermaxman/esa-hubble]. This dataset is specifically curated for tasks involving space science, and includes the following components:

High-resolution astronomical images captured by the Hubble Space Telescope

Expert-written reference descriptions written by scientists and curated by ESA

Metadata fields including:

1. Object name or identifier (e.g., NGC 1300, UGC 12158)

2. Object category (e.g., galaxy, nebula, star cluster)

3. Constellation name

4. Distance (where applicable)

Before training and inference, all images are preprocessed and resized to 224×224 pixels to match the input requirements of the BLIP and CLIP models. Metadata is optionally included in the captioning prompt to steer model behavior and encourage generation of accurate scientific terms and references.

This dataset allows the system to generate grounded and informative descriptions that closely resemble those written by human experts, thereby improving the utility of AI in astronomy and scientific imaging.

## References
 
[1] Kinakh, V., Belousov, Y., Schaerer, D., Quétant, G., Voloshynovskiy, S., Drozdova, M., & Holotyak, T. (2024). Hubble Meets Webb: Imageto-Image Translation in Astronomy. Sensors, 24(4), 1151. https://doi.org/10.3390/s24041151 

[2] Alam, M. T., Imam, R., Guizani, M., & Karray, F. (2024). AstroSpy: On detecting fake images in astronomy via joint image-spectral representations. arXiv. https://arxiv.org/abs/2407.06817  

[3] Reale-Nosei, G., Amador-Domínguez, E., & Serrano, E. (2024). From vision to text: A comprehensive review of natural image captioning in medical diagnosis and radiology report generation. Medical Image Analysis, 97, 103264. https://doi.org/10.1016/j.media.2024.103264  

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,  A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need  (arXiv:1706.03762).arXiv. https://doi.org/10.48550/arXiv.1706.03762  

[5] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale (arXiv:2010.11929). arXiv. https://doi.org/10.48550/arXiv.2010.11929  

[6] Sinha, G. R., Thakur, K., & Vyas, P. (2017). Research impact of astronomical image processing. International Journal of Luminescence and Applications, 7(3–4), 503–506. 
