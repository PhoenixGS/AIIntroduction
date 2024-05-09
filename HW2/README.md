# README

* 数据准备

    将代码及数据集按如下形式放置：
    
    ```
    .
    ├── Dataset
    │   ├── test.txt
    │   ├── train.txt
    │   ├── validation.txt
    │   └── wiki_word2vec_50.bin
    ├── README.md
    ├── main.py
    └── models.py
    ```

* 依赖库：

    `torch, gensim, numpy, argparse`

* 代码运行方式：

    `python main.py [options]`
    
    options:
      `--model MODEL_NAME`: `MLP, MLP2, CNN, RNN, RNN2, Transformer`, default: `RNN`
      `--epochs EPOCHS`: number of epochs, default: `50`
      `--batch_size BATCH_SIZE`: batch size, default: `512`
      `--lr LEARNING_RATE`: learning rate, default: `0.001`
      `--dropout DROPOUT`: dropout rate, default: `0.5`
    
    


