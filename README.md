# EpicDiffusion: Enhancing Video Diffusion Models for Storytelling

## Abstract

Diffusion models are proficient at generating images from textual prompts. Recent advancements have enabled encoding video information in diffusion models, leading to the creation of temporally consistent videos. However, there is still significant exploration needed to translate passages from novels and epic literature into animated illustrations. This project aims to enhance diffusion models to better generate animated illustrations from novel passages. We discuss the issues in diffusion models, integration with large language models (LLMs) for prompt generation, and a custom architecture to support text-to-video alignment. We introduce a new LLM-guided prompt weighting method to generate spatially and temporally consistent videos for fictional passages. Furthermore, we perform qualitative and quantitative evaluations on these images and videos across various action categories.

## Problem Statement

The objective of EpicDiffusion is to address the current limitations of video diffusion models in capturing the nuanced narrative and thematic elements essential for epic and historic fiction. Despite their success in a broad range of tasks, these models often fall short in generating contextually rich visuals for specific storytelling genres, resulting in outputs that may lack depth or relevance. EpicDiffusion seeks to develop a bespoke video generation model that:

- Enhances the generative capabilities of diffusion models to accurately interpret and visualize complex narrative content.
- Advances the technology while respecting the source material’s artistic and historical integrity, providing a tool for generating personalized, engaging visuals.

## Motivation

1. **Enhanced Reading Engagement**: Integration of visual elements into epic and historic fiction can significantly enrich the reader’s experience. By generating dynamic visuals, EpicDiffusion aims to bridge traditional reading experiences with contemporary digital storytelling methods.
2. **Personalized Video Graphics**: Tailoring visuals for book covers and illustrations to individual preferences or specific narrative elements can transform reading into an interactive experience, enhancing enjoyment and emotional connection to the story.

## Technical Background

### Diffusion Models

A diffusion model is a generative model that generates data based on what it is trained on. The process involves three main steps:

1. **Forward Process (Diffusion or Noising)**: Adding Gaussian noise to the input data.
2. **Reverse Process (Denoising)**: Removing the noise to reconstruct the original data.
3. **Generation**: Generating new data by starting with random noise and denoising it.

### Prompt Weighting

Prompt weighting is used to control the influence of different concepts within a text prompt on the generated image by adjusting the scale of text embeddings corresponding to each concept.

### FreeU

FreeU enhances the generation quality of image and video models by re-weighting contributions from the U-Net’s skip connections and backbone feature maps.

### Related Work

- **I2VGenXL Diffusion Model**: Addresses challenges in semantic accuracy, clarity, and spatiotemporal continuity in video generation.
- **LLM-Grounded Diffusion**: Guides text-to-image diffusion models to handle complex prompts involving numeracy and spatial reasoning.
- **Timesformer**: A state-of-the-art video understanding model using spatial and temporal attention mechanisms.
- **Kinetics 400 Dataset**: Standard dataset in video understanding tasks, containing around 650k videos across 400 categories.

## Technical Implementation

### Infrastructure

EpicDiffusion uses ASU Sol Supercomputing resources, particularly the Nvidia A100 GPU, for generating Stable Diffusion images and videos.

### Datasets

To test the efficiency of our approach, we used fictional passages from works like Aesop’s fables and Adventures of Tom Sawyer. For evaluation, we generated synthetic data across different action categories from the Kinetics-400 dataset.

### Architecture

1. Use GPT-4 to dissect the given passage into sections (Character, Action, Setting) using the Chain-of-thought method.
2. Generate an effective prompt with weights using Chain-of-thought.
3. Configure the model using Guiding-Scale and Inference-Steps parameters.
4. Add a FreeU Mixin to the diffusion pipeline to tweak the skip connection contributions from the U-Net.
5. Generate the short video based on the image and the action using the I2VGenXL model.

## Evaluation Plan

### Evaluating T2I Model

- **Fantasy Fiction**: Evaluated qualitatively with manual inspection.
- **Historical Fiction**: Evaluated qualitatively with manual inspection.
- **Contemporary**: Evaluated using CLIP score.

### Evaluating I2V Model

Evaluated using Timesformer to compute the action classes from 30 videos across 3 classes and compute the overall accuracy from the ground truth.

## Results

### Semantic Composability of T2I Model

Images generated with LLM prompt weighting showed improved focus on key elements from the passages.

### Temporal Consistency of I2V Model

Generated videos were consistent with the expected passages.

## Conclusion

### Observations

- LLM enhanced semantic alignment.
- A balanced guidance scale of 12.5 provided optimal performance.
- 200 denoising steps for text-to-image and 100 for text-to-video were used for optimal image quality and generation time.

### Future Work

- Explore LLM enhanced prompt weighting for negative prompts.
- Broaden the range of classes for CLIP score evaluation.
- Investigate appropriate video benchmarking methods.

## References

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
2. Zhang, S., Wang, J., Zhang, Y., Zhao, K., Yuan, H., Qin, Z., Wang, X., Zhao, D., & Zhou, J. (2023). I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145.
