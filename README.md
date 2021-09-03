# Introduction
This repo contains the code for the paper "What’s Hidden in a One-layer Randomly Weighted Transformer?" at EMNLP 2021 (short) main conference. 
We demonstrate that one-layer randomly weighted Transformer contain subnet-works that can achieve impressive performance **without ever modifying the weight values**  on  machine  translation  tasks. To find subnetworks for one-layer randomly weighted Transformer, we apply different binary masks to the same weight matrix to generate different layers. Hidden in a one-layer randomly weighted Transformer`wide/wider`, we find subnetworks can achieve **29.45/17.29** BLEU on IWSLT14/WMT14.  Using a fixed pre-trained embedding  layer, the previously found subnetworks are smaller than, but can match **98%/92%** (34.14/25.24 BLEU) the performance of a trained Transformer`small/base` on IWSLT14/WMT14. 

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.0)
# pip install fairseq==0.10.0
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Training 
We have the pre-processed data for IWSLT/WMT and the pre-trained encoder/decoder embedding at [data](https://drive.google.com/drive/u/0/folders/1JwaCMLaX2Amz5DftJ_nCr3glAqGTBdsV). 

To train a one-layer randomly weighted Transformer on IWSLT with one GPU, you can run the script below:
```
OUTPUT_PATH=
DATA_PATH=data-bin/iwslt14.tokenized.de-en/
prune_ratio=0.5
share_mask=layer_weights
init=kaiming_uniform
_arch=masked_transformer_iwslt_de_en/masked_transformerbig_iwslt_de_en

python train.py ${DATA_PATH} --seed 1 --fp16 --no-progress-bar \
        --max-epoch 55 --save-interval 1 \
        --keep-last-epochs 5 \
        --arch ${_arch} \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 --eval-bleu \
        --eval-bleu-args '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}' \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric {encoder_embed} {decoder_embed} \
        --share-mask ${share_mask} --clip-norm 0.5 \
        --mask-layernorm-type masked_layernorm \
        --prune-ratio ${prune_ratio} --mask-init standard --prune-method super_mask --mask-constant 0. \
        --init ${init} --scale-fan --share-decoder-input-output-embed \
        --save-dir ${OUTPUT_PATH} | tee -a ${OUTPUT_PATH}/train_log.txt
```


To train a one-layer randomly weighted Transformer on WMT with 8 GPUs, you can run the script below:
```
OUTPUT_PATH=
DATA_PATH=data-bin/wmt14_en_de_joined_dict/
prune_ratio=0.5
share_mask=layer_weights
_arch=masked_transformer_wmt_en_de/masked_transformer_wmt_en_de_big
init=kaiming_uniform

python train.py ${DATA_PATH} \
    --seed 1 --dropout 0.2 --no-progress-bar --fp16 \
    --arch ${_arch} --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 \
    --lr 1e-3 --update-freq 1 --log-interval 50 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 4096 --save-dir ${OUTPUT_PATH} --distributed-world-size 8 --distributed-port 61024 \
    --ddp-backend no_c10d --keep-interval-updates 20 --keep-last-epochs 10 --max-epoch 100 \
    --share-mask ${share_mask} \
    --mask-layernorm-type masked_layernorm \
    --prune-ratio ${prune_ratio} --mask-init standard --prune-method super_mask --mask-constant 0. \
    --init ${init} --scale-fan --share-decoder-input-output-embed  | tee -a ${OUTPUT_PATH}/train_log.txt
```

You can add the ``--encoder-embed-path data/iwslt_embed/encoder_embed.txt --decoder-embed-pat data/iwslt_embed/decoder_embed.txt`` with pre-trained word embeddings. 

# Testing 
To test on IWSLT, you can run the following script
```
BEAM_SIZE=5
python fairseq_cli/generate.py ${DATA_PATH} \
        --path ${OUTPUT_PATH}/checkpoint_best.pt \
        --batch-size 128  --beam ${BEAM_SIZE} \
        --lenpen 1.0 --remove-bpe --log-format simple \
        --source-lang de --target-lang en \
    > {RESULT_PATH}/res.txt"
```
To test on WMT, you can run the following script
```
python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints 10 --output ${OUTPUT_PATH}/averaged_model.pt
python fairseq_cli/generate.py ${DATA_PATH} \
    --path ${OUTPUT_PATH}/averaged_model.pt \
    --beam 5 --remove-bpe \
    > ${RESULT_PATH}/res.txt
```

# Related Papers
* What’s Hidden in a Randomly Weighted Neural Network?
* Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask
* The Lottery Ticket Hypothesis for Pre-trained BERT Networks
* PLAYING THE LOTTERY WITH REWARDS AND MULTIPLE LANGUAGES: LOTTERY TICKETS IN RL AND NLP
* When BERT Plays the Lottery, All Tickets Are Winning

# Citation
This repo has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:
```
@article{shen2021whats,
  title={What’s Hidden in a One-layer Randomly Weighted Transformer?},
  author={Shen, Sheng and Yao, Zhewei and Kiela, Douwe and Keutzer, Kurt and Mahoney, Michael W},
  booktitle={EMNLP},
  year={2021}
}
```