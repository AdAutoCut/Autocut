# Autocut

AutoCut: End-to-end advertisement video editing based on multimodal discretization and controllable editing.

This repository is under active development. Code and released resources will be available soon.



## Dataset

We release URL files for part of the AutoCut dataset.

The released files are:

* `data/autocut_data_url.csv`: URLs to the processed AutoCut data.
* `data/video_urls_testing_set_filtered.csv`: URLs to the raw testing-set videos.

The video files are not stored directly in this repository. The CSV files only provide URLs for downloading the corresponding data. Each URL can be accessed directly in a browser or downloaded with `wget` / `curl`.

The released URLs are provided for academic research and reproducibility purposes only. Please do not redistribute the downloaded video files or use them for commercial purposes unless you have obtained the necessary rights.


## Model

The AutoCut model is available on Hugging Face:  
[MiltonZ/AutoCut_embsft](https://huggingface.co/MiltonZ/AutoCut_embsft)

After downloading:

- Place the dataset files under: `llm_factory_test/data/`  
- Place the model checkpoint under: `llm_factory_test/saves/`

---

To reproduce the main experiment for the AutoCut Model, first download the released datasets and the AutoCut model checkpoint, and place them in the corresponding directories described above. Then run the following inference script:

```bash
python llm_factory_test/baselines/_vllm_infer_atc.py