# IEOR4575 Project
Leon Li al4263

## Run the code
Use the notebook al4263_final_proj.ipynb to run the code

### Easy Mode:
option 1: directly train the policy network on the easy configuration. 

option 2: train the policy network on a configuration of 1 instance first, and then train on the easy configuration

Load model: 

PATH = cwd + '/Policy/easy_model'

Policy = torch.load(PATH)

### Hard Mode:
train on curriculum training set first, save model as curriculum_model

load the curriculum model to train on the hard configuration

Load model: 

PATH = cwd + '/Policy/hard_model'

Policy = torch.load(PATH)

### Test Mode:
Use either easy_model or hard_model to test

## Policy Network
The attention embedding is in the file attention_network.py

The policy network including backprop is in policy_network.py

## Helper Classes
In helper.py. Include discounter reward and evolution strategies

## Hyper-parameters:
ES: sigma = 5

lr: 3e-4

LSTM_embedding_size: 32

attention_embedding_size: 16
