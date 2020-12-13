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




