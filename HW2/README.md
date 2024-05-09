依赖库：

`torch, gensim, numpy, argparse`

代码运行方式：

`python main.py [options]`

options:
  `--model MODEL_NAME`: `MLP, MLP2, CNN, RNN, RNN2, Transformer`, default: `RNN`
  `--epochs EPOCHS`: number of epochs, default: `50`
  `--batch_size BATCH_SIZE`: batch size, default: `512`
  `--lr LEARNING_RATE`: learning rate, default: `0.001`
  `--dropout DROPOUT`: dropout rate, default: `0.5`
