# Official Implementations "Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models" (ICLR'24)</sub>



[//]: # (**Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models**<br>)
<hr />

**Abstract**: The success of recent text-to-image diffusion models is largely due to their capacity to be guided by a complex text prompt, which enables users to precisely describe the desired content. However, these models struggle to effectively suppress the generation of undesired content, which is explicitly requested to be omitted from the generated image in the prompt. In this paper, we analyze how to manipulate the text embeddings and remove unwanted content from them. We introduce two approaches, which we refer to as **soft-weighted regularization** and **inference-time text embedding optimization**. The first regularizes the text embedding matrix and effectively suppresses the undesired content. The second method aims to further suppress the unwanted content generation of the prompt, and encourages the generation of desired content. We evaluate our method quantitatively and qualitatively on extensive experiments, validating its effectiveness. Furthermore, our method is generalizability to both the pixel-space diffusion models (i.e. DeepFloyd-IF) and the latent-space diffusion models (i.e. Stable Diffusion).

<hr />

## 🛠️ Method Overview
<span id="method-overview"></span>

![Random Sample](./docs/overview.jpg)

Overview of the proposed method. (a) We devise  a negative target embedding matrix $\boldsymbol\chi$: $\boldsymbol\chi = [\boldsymbol{c}^{NE},\boldsymbol{c}^{EOT}\_0, \cdots, \boldsymbol{c}^{EOT}\_{N-{|\boldsymbol{p}|-2}}]$.  We perform SVD for the embedding matrix $\boldsymbol\chi=\textbf{\emph{U}}{\boldsymbol\Sigma}{\textbf{\emph{V}}}^T$. We introduce a soft-weight regularization  for each largest eigenvalue. Then  we recover the embedding matrix $\hat{\boldsymbol\chi}=\textbf{\emph{U}}{\hat{\boldsymbol\Sigma}}{\textbf{\emph{V}}}^T$. (b) We propose inference-time text embedding optimization (ITO).  We align the attention maps of both $\boldsymbol{c}^{PE}$ and  $\boldsymbol{\hat{c}}^{PE}$, and widen  the ones of  both $\boldsymbol{c}^{NE}$ and $\boldsymbol{\hat{c}}^{NE}$.

## 💻 Requirements
The codebase is tested on 
* Python 3.8
* PyTorch 1.12.1
* Quadro RTX 3090 GPUs (24 GB VRAM) with CUDA version 11.7

environment or python libraries:

```
pip install -r requirements.txt
```


## 🎊 Suppression for real image
```shell
python suppress_eot_w_nulltext.py  --type Real-Image \
                                   --prompt "A man with a beard wearing glasses and a hat in blue shirt" \
                                   --image_path "./example_images/A man with a beard wearing glasses and a hat in blue shirt.jpg" \
                                   --token_indices "[[4,5],[7],[9,10],]" \
                                   --alpha "[1.,]" --cross_retain_steps "[.2,]"
```
![Random Sample](./docs/supresseot_results.png)

You can use **NPI** (Negative Prompt Inversion) for faster inversion of the real image, but it may lead to a certain degree of degradation in editing quality:
```shell
python suppress_eot_w_nulltext.py  --type Real-Image --inversion NPI\
                                   --prompt "A man with a beard wearing glasses and a hat in blue shirt" \
                                   --image_path "./example_images/A man with a beard wearing glasses and a hat in blue shirt.jpg" \
                                   --token_indices "[[4,5],[7],[9,10],]" \
                                   --alpha "[1.,]" --cross_retain_steps "[.2,]"
```



## 🪄 Additional application

### 1️⃣ Generating subjects for generated image ([Attend-and-Excite](https://arxiv.org/abs/2301.13826) similar results ()

```
python suppress_eot_w_nulltext.py  --type Generated-Image \
                                   --prompt "A painting of an elephant with glasses" --seed 16 \
                                   --token_indices "[[7],]" \
                                   --alpha "[-0.001,]" --cross_retain_steps "[.2,]" --iter_each_step 0
```
![Random Sample](./docs/generating_subjects.jpg)

### 2️⃣ Adding subjects for real image ([GLIGEN](https://arxiv.org/abs/2301.07093) similar results)

```
python suppress_eot_w_nulltext.py  --type Real-Image \
                                   --prompt "A car near trees with raining" \
                                   --image_path "./example_images/A car near trees.jpg" \
                                   --token_indices "[[6],]" \
                                   --alpha "[-0.001,]" --cross_retain_steps "[.2,]"
```
![Random Sample](./docs/adding_subjects.jpg)


### 3️⃣ Replacing subject in the real image with another

```
python suppress_eot_w_nulltext.py  --type Real-Image \
                                   --prompt "A car near trees with raining" \
                                   --image_path "./example_images/A car near trees.jpg" \
                                   --token_indices "[[6],]" \
                                   --alpha "[-0.001,]" --cross_retain_steps "[.2,]"
```
![Random Sample](./docs/adding_subjects.jpg)

## Contact
Should you have any questions, please contact senmaonk@gmail.com


**Acknowledgment:** This code is based on the [P2P, Null-text](https://github.com/google/prompt-to-prompt) repository. 