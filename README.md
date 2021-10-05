# Adapting the Transformer architecture to model scientific real-time experimental data.

![In this research (A) Transformer encoder will be used to process user-defined variables, while the decoder will be used to generate scientific data. The main variations over the original Transformer (C) are the inclusion of an encoder reconstruction layer.](https://user-images.githubusercontent.com/1437098/135989873-f87dc36f-4dc9-4c88-a6da-1299c631a1e9.jpeg)

<img src="https://user-images.githubusercontent.com/1437098/135989873-f87dc36f-4dc9-4c88-a6da-1299c631a1e9.jpeg" width=50% height=50%>


This code implements a vanilla Transformer as seen in "Attention is all you need".
It has some modifications so that it can work with sequential scientific data. It is better to check the paper I wrote about this work. You can check it [here](https://www.juanma.io/scitrans.pdf). It has been accepted in a conference so I will soon update the proper link to the conference and also an arxiv version of it.

To understand the following code, you first need to understand the Transformer architecture. Do the following steps before trying to read this code:

1. Read the paper "Attention is all you need", or check tutorials about attention, self attention and transformers. Seqseq is also recommended.
2. Read the [TF tutorial about Transformers](https://www.tensorflow.org/tutorials/text/transformer). I also suggest you check the other tutorials related to attention.
3. Check [this implementation of the previous tutorial in Keras.](https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83)

My code is based on points 2/3 with the required variations needed for this research.

The route of action to get the stuff working is:

* First use prepare_data.py to generate sequences of data. It will be encapsulated in a keras generator.
* Then use train.py, which will use the data and the Transformer.py to create a model, and train it.
* Once the model has been trained, you can check the GA_RL folder to see different applicatons.
* You can also go to utils and use evaluate to generate sequences.

The code works now in my machine (and the different machines I use for training) using Python 3.8, Tensorflow and Keras 2.4. 
