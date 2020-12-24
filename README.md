# Chatbot
A Chatbot made based on sequence to sequence model



### Set up the environment

* The whole script was written based on python 3.5

* First, create a virtual environment that can run python 3.5 on anaconda (or you can create it in your local machine), the name created here is called 'chatbot'

  ```
  conda create -n chatbot python=3.5 anaconda
  ```

* After the environment was set up, you need to activate the virtual environment

  ```
  source activate chatbot
  ```

* Once the virtual environment is activated, you are good to go

  

### Packages needed to be installed

* Package that needs to be installed here is 'tensorflow', specifically under the version of 1.0.0

  ```
  pip install tensorflow==1.0.0
  ```

### Train the model
* Run chatbot.py to train the model

  ```
  python chatbot.py
  ```

  * The training is time consuming, to run the model while your local machine is still available for other work, it is recommended to train this model using a virtual machine.
* To run the model on a virtual machine, using AWS EC2 or google colab are both great options.

### Setting up the chatbot

* After the training is done, `.ckpt` files will show up (Checkpoint is not uploaded here due to the large file size). 

* Run the file `execute_chat.py` to set up the chatbot and be ready to chat

  ```python
  python execute_chat.py
  ```

*  The path to the ckeckpoint will be asked from the console

  <img src="/Users/wengweicheng/googleLocal/Chatbot/data/Screen Shot 2020-12-24 at 6.00.14 PM.png" style="zoom:100%;" />



* After the path is entered, you will be able to chat with the chatbot, however, to increase the capability of this bot, more training will be required.



