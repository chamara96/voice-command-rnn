# Voice Command Interface for Robot Control
**(Machine Learning Part of the Final Year Undergraduate Project)**
  - Recurrent Neural Network
  - Create own datasets and Train
  - GUI

> Main Interface

![](https://github.com/chamara96/voice-command-rnn/blob/main/main_gui.png)

> During Training (Real-time ploting)

![](https://github.com/chamara96/voice-command-rnn/blob/main/during_training.png)
  
 ### Run
 ```python
$ python GUI.py
```

### Usage
  - Press and Hold "T" for testing using in-built Microphone (by Default model is loaded inside the model folder `my.ckpt`)
  - Press and Hold "R" for Record new voice for Make own datasets

### Python Libraries
  - pytorch cpu
  ```
  $ pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  ```
  
  - tkinter
  - librosa
  - soundfile
  - sounddevice
  - numpy
  - matplotlib
  - threading
