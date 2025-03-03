




# Environment

Pillow==9.5.0
python-version==0.0.2
scikit-image==0.19.3
tensorflow==2.11.0

Please run the following line to install enviroment
```python

pip install -r requirements.txt

```

# How to try



## dataset 
We use the synthetic method from [Zheng et. al] (https://github.com/q-zh/absorption)

[place 365](https://github.com/CSAILVision/places365)

[RID](https://github.com/USTCPCS/CVPR2018_attention)

[SIR2](https://www.dropbox.com/scl/fi/qgg1whla1jb3a9cgis18l/SIR2.zip?rlkey=kmhrc2uk63be2s9hzr43gc3hm&e=1&st=cfsh8sol&dl=0)

[Nature](https://github.com/JHL-HUST/IBCLN)

[Zhang et. al](https://github.com/ceciliavision/perceptual-reflection-removal)

[ERRNET](https://github.com/Vandermode/ERRNet)


```python

Data_root/
         -train/
               -syn
               -t
               -r
               -estimate.txt
               ⋮
         -test/
               -syn
               -r
               -t
               ⋮
 ⋮

```

## Training
```python

Trainging Teacher Net -python TSmain.py

Training Student Net  -python student.py

reflection refraction estimate -python estimation.py

```

## Run testing
```python

Test Teacher Net -python TS_teacher_test.py

Test Student Net -python TS_student_test.py

Calculating model performance  -python compute.py

```



The optimizer selected is Adam, and the initial learning rate is 1e-4. The program execution environment is Tensorflow 2.11.0, python 3.7.16, and the graphics cards used are NVIDIA GeForce GTX 1080 Ti and NVIDIA GeForce RTX 3090. Input video size: 256×256



## Citation
If you use this code for your research, please cite our paper :

```
@inproceedings{chan2025single,
  title={ Single-Image Reflection Removal via Knowledge
 Distillation Network with Deep Feature Extraction
 and Reflection-Refraction Estimation},
  author={Chan, U-In and Liu, Tsung-Jung},
  year={2024}
}
```
