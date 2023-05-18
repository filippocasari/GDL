# Project GDL - AEGNN
@authors: Filippo Casari, Alessandro De Grandi, Anthony Bugatto
## Setup
We suggest to run our code with a linux OS. \
Create a conda environment by running:
```bash
conda env create --file=environment.yml
```

## Preprocessing
To run the preprocessing script:
```bash
AEGNN_DATA_DIR=$(pwd)/data python3 scripts/preprocessing.py --dataset "ncaltech101" --num-workers 4 --gpu 0
```
In case you want to work on Ncars dataset, substitute Ncaltech with 'ncars'
**AEGNN_DATA_DIR** is the dir in which the dataset should be. You can change this env variable if your dataset is elsewhere. Change the number of workers according with your hardware.
Alternatively, use experiments jupyter notebook that includes also dataset splitting, and some useful plots for accuracy per event. 
## Training
```bash
AEGNN_DATA_DIR=$(pwd)/data AEGNN_LOG_DIR=$(pwd)/data/log python scripts/train.py graph_res --task recognition --dataset "ncaltech101" --gpu 0 --batch-size 8 --max-epochs 20 --num-workers 8
```
As before, you can change the name of the dataset you are working on as well as the batch size and number of epochs. 
The authors used for Ncaltech a batch size of 16 and 64 for Ncars. If your GPU memory runs out, you may want to reduce the batch size or use:
```python
trainer_kwargs["accumulate_grad_batches"] = 2
```
that is set true by default within the code. 
## Flops Evaluation

```bash
PYTHONPATH=$(pwd) AEGNN_DATA_DIR=$(pwd)/data/storage python evaluation/flops.py
```
## Accuracy per event
To run the accuracy per event, you have to pass as args the path of your trained model, the device you're using, the dataset, and the batch size. We added the last argument to the script to avoid running out of memory. 

```bash
python evaluation/accuracy_per_events.py /home/ale/Downloads/GDL/data/log/checkpoints/ncaltech101/recognition/20230505200347/epoch\=19-step\=8199.pt --device cuda --dataset ncaltech101 --bsize 8
```
Suggestion: for Ncars use as batch size 64. 
## Plot flops
We created a script for plotting number of flops per layer. You can run:
```bash
python example_plot_flops.py
```
