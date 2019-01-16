## RNNControl

Jupyter notebooks that run experiments investigating the effect of conditional parameters on audio synthesis.   
The code here is closely integrated with [audioDataloader](https://github.com/muhdhuz/AudioDataloader) and [paramManager](https://github.com/lonce/paramManager).
Clone those repositories and run the notebooks from this repo alongside them as follows: 

.  
+--model  
+--README.md  
+--paramManager  
+--audioDataloader  
+--explore_generation.ipynb  
+--main.ipynb  
+--data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|+--dataset    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|+--dataparams    
+--utils   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|+--myUtils.py    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|+--architectures.py  

**Files**  
* main.ipynb: for training and saving model (an RNN)
* explore_generation.ipynb: to do cool stuff i.e. audio synthesis with the trained model
* model: pre-trained models can be found here
* myUtils.py: contains some utility functions e.g for plotting  
* architectures.py: the model architectures should be loaded from here   

**Dependencies**  
* pytorch 0.4.0
* [audioDataloader](https://github.com/muhdhuz/AudioDataloader)
* [paramManager](https://github.com/lonce/paramManager)
  

**Authors**  
* Muhammad Huzaifah

- - -
NOTE: this project is incomplete, run at your own risk.  

**To do**  
 * Speed up generation
 * Explore conditional parameters






