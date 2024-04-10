
# Diffusion-KTO: Aligning Diffusion Models by Optimizing Human Utility

<p align="center">
    <img src="assets/teaser.png", width=50%> <br>
</p>


This repository contains the official code and checkpoints used in the paper "[Aligning Diffusion Models by Optimizing Human Utility](https://arxiv.org/abs/2404.04465)"


## Abstract
We present Diffusion-KTO, a novel approach for aligning text-to-image diffusion models by formulating the alignment objective as the maximization of expected human utility. Since this objective applies to each generation independently, Diffusion-KTO does not require collecting costly pairwise preference data nor training a complex reward model. Instead, our objective requires simple per-image binary feedback signals, e.g. likes or dislikes, which are abundantly available. After fine-tuning using Diffusion-KTO, text-to-image diffusion models exhibit superior performance compared to existing techniques, including supervised fine-tuning and Diffusion-DPO, both in terms of human judgment and automatic evaluation metrics such as PickScore and ImageReward. Overall, Diffusion-KTO unlocks the potential of leveraging readily available per-image binary signals and broadens the applicability of aligning text-to-image diffusion models with human preferences.


## Checkpoint will be available soon

## Datasets

We used a processed version of [Pick-a-Pick](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) with binary label instead of pair-wise preference. The processed annotation `train_processed_v3.csv` can be found at [Huggingface](https://huggingface.co/datasets/jacklishufan/pick-a-pick-kto-processed).

Please download the original Pick-a-Pick dataset and preprocessed annotation and organize the data folder in the following manner:

```
<root of pick-a-pick>
-train_processed_v3.csv
-train
--- 17a0ce90-9e1c-4366-82c7-aa5977b06375.jpg
-- ... 
```

## Training

``` 
bash exps/example.sh 
```

## Citation
```
@misc{li2024aligning,
      title={Aligning Diffusion Models by Optimizing Human Utility}, 
      author={Shufan Li and Konstantinos Kallidromitis and Akash Gokul and Yusuke Kato and Kazuki Kozuka},
      year={2024},
      eprint={2404.04465},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
