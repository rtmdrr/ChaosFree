# Deep Learning on Computational Accelerators Project

This code implements a Word-level language modeling multi-layer RNN of the following configurations: RNN_TANH, RNN_RELU, LSTM, GRU, CFN, BNLSTM, BNCFN.
By default, the training script uses the PTB dataset, provided. 
The trained model can then be used by the generate script to generate new text.

```
# Train
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 100 --lr 0.001 --optim adam --model GRU
# Generate samples from the trained model.
python generate.py
```

The model will automatically use the cuDNN backend if run on CUDA with cuDNN installed.
The main.py script accepts the following arguments:
```
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, CFN, BNLSTM, BNCFN)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --optim OPTIM      learning rule, supports
                     adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta
  --epochs EPOCHS    upper epoch limit
  --batch_size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
  ```
  
  ## Models
  
  ### RNN_TANH
  ![alt text](https://datascience-enthusiast.com/figures/rnn_step_forward.png)

  ### RNN_RELU
  Same as RNN_TANH but with a ReLU activation function.
  ### LSTM
  ![alt text](https://i.stack.imgur.com/aTDpS.png)
  ### GRU
  ![alt text](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)
  ### CFN
  Implemented as described in the paper: [Chaos Free Network](https://openreview.net/pdf?id=S1dIzvclg)
  ### BNLSTM
  This is an implementation of an LSTM network with batch normalization. Implemented as described in the paper: [Batch Normalization LSTM](https://arxiv.org/abs/1603.09025)
  ### BNCFN
  Implemented according to the same principles as described in the paper of batch normalization RNN, only to the CFN architecture.
  
