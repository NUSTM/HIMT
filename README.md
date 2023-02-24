# HIMT
The code for our paper Hierarchical Interactive Multimodal Transformer for Aspect-Based Multimodal Sentiment Analysis


## Usage
We use [Faster-RCNN](https://github.com/peteanderson80/bottom-up-attention) to extract image features and corresponding semantic labels. You can download from [Baidu Netdist](https://pan.baidu.com/s/1F3rI0oSA2GTvToXlhXAsmQ) with the extraction code kbtf. Before you run the bash file, you need to change the parameter **--img_path** to your save path.

For the textual input of each sample, please refer to [TomBERT](https://github.com/jefferyYu/TomBERT) to download them from the **absa_data** folder.

## Training and Inference
```
bash run_himt.sh
```

## Acknowledgements

- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.
