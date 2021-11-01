# PytorchImgRecognition
## To Use This By Myself
datasets should be like
```
./datasets/DatasetRoot/ ┬ label1 ┬ image0.jpg
                        │        ├ image1.jpg
                        │        ├ image2.jpg
                        │        ...
                        ├ label2 ┬ img0.jpg
                        │        ├ img1.jpg
                        │        ├ img2.jpg
                        │        ...
                        ...
```  

make `temp` directory to store model  
using `is_fineTune = False` may consume a lot of memory.
using `is_randomTune = True` may consume a lot of memory.
