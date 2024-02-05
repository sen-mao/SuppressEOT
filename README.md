# Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models (ICLR'24)</sub>

![Random Sample](./docs/supresseot_results.png)

**Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models**<br>

**Abstract**: *The success of recent text-to-image diffusion models is largely due to their capacity to be guided by a complex text prompt, which enables users to precisely describe the desired content. However, these models struggle to effectively suppress the generation of undesired content, which is explicitly requested to be omitted from the generated image in the prompt. In this paper, we analyze how to manipulate the text embeddings and remove unwanted content from them. We introduce two approaches, which we refer to as *soft-weighted regularization* and *inference-time text embedding optimization*. The first regularizes the text embedding matrix and effectively suppresses the undesired content. The second method aims to further suppress the unwanted content generation of the prompt, and encourages the generation of desired content. We evaluate our method quantitatively and qualitatively on extensive experiments, validating its effectiveness.*
## Requirements
The codebase is tested on 
* Python 3.8
* PyTorch 1.12.1
* Quadro RTX 3090 GPUs (24 GB VRAM) with CUDA version 11.7

environment or python libraries:

```
pip install -r requirements.txt
```

## Suppression for generated image
```
python suppress_content_w_eot.py  --type 'Generated-Image' \
                                  --prompt "A man without glasses" --seed 2 \
                                  --token_indices "[[4],]" \
                                  --alpha '(1.,)' --cross_retain_steps '(.3,)'
```

## Suppression for real image
```
python suppress_content_w_eot.py  --type 'Real-Image' \
                                  --prompt "A man with a beard wearing glasses and a beanie in blue shirt" \
                                  --token_indices "[[5],[7],[10],]" \
                                  --alpha '(1.,)' --cross_retain_steps '(.2,.3,.4,)' --max_step_to_erase 20
```

