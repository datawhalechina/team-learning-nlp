# task3：Faster 情感分析

上一章中我们已经介绍了基于RNN的升级版本的情感分析，在这一小节中，我们将学习一种不使用RNN的方法：我们将实现论文 [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)中的模型，该论文已经放在了教程中，感兴趣的小伙伴可以参考一下。这个简单的模型实现了与第二章情感分析相当的性能，但训练速度要快得多。

## 3.1 数据预处理

FastText分类模型与其他文本分类模型最大的不同之处在于其计算了输入句子的n-gram，并将n-gram作为一种附加特征来获取局部词序特征信息添加至标记化列表的末尾。n-gram的基本思想是，将文本里面的内容按照字节进行大小为n的滑动窗口操作，形成了长度是n的字节片段序列，其中每一个字节片段称为gram。具体而言，在这里我们使用bi-grams。

例如，在句子“how are you ?”中，bi-grams 是：“how are”、“are you”和“"you ?”。

“generate_bigrams”函数获取一个已经标注的句子，计算bigrams并将其附加到标记化列表的末尾。


```python
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x
```

例子：


```python
generate_bigrams(['This', 'film', 'is', 'terrible'])
```




    ['This', 'film', 'is', 'terrible', 'film is', 'This film', 'is terrible']



TorchText 'Field' 中有一个`preprocessing`参数。此处传递的函数将在对句子进行 tokenized （从字符串转换为标token列表）之后，但在对其进行数字化（从tokens列表转换为indexes列表）之前应用于句子。我们将在这里传递`generate_bigrams`函数。

由于我们没有使用RNN，所以不需要使用压缩填充序列，因此我们不需要设置“include_length=True”。


```python
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  preprocessing = generate_bigrams)

LABEL = data.LabelField(dtype = torch.float)
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/field.py:150: UserWarning: Field class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/field.py:150: UserWarning: LabelField class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    


与前面一样，加载IMDb数据集并创建拆分：


```python
import random

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/example.py:78: UserWarning: Example class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('Example class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.', UserWarning)
    


构建vocab并加载预训练好的词嵌入：


```python
MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)
```

创建迭代器：


```python
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/iterator.py:48: UserWarning: BucketIterator class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    

## 3.2 构建模型

FastText是一种典型的深度学习词向量的表示方法，通过将Embedding层将单词映射到稠密空间，然后将句子中所有单词在Embedding空间中进行平均，进而完成分类。所以这个模型参数量相较于上一章中的模型会减少很多。

具体地，它首先使用'Embedding'层（蓝色）计算每个词嵌入，然后计算所有词嵌入的平均值（粉红色），并通过'Linear'层（银色）将其输入。

![](assets/sentiment8.png)

我们使用二维池化函数“avg_pool2d”实现单词在Embedding空间中的平均化。我们可以将词嵌入看作为一个二维网格，其中词沿着一个轴，词嵌入的维度沿着另一个轴。下图是一个转换为5维词嵌入的示例句子，词沿纵轴，嵌入沿横轴。[4x5] tensor中的每个元素都由一个绿色块表示。

![](assets/sentiment9.png)

“avg_pool2d”使用大小为“embedded.shape[1]”（即句子长度）乘以1的过滤器。下图中以粉红色显示。

![](assets/sentiment10.png)


我们计算filter 覆盖的所有元素的平均值，然后filter 向右滑动，计算句子中每个单词下一列嵌入值的平均值。

![](assets/sentiment11.png)

每个filter位置提供一个值，即所有覆盖元素的平均值。filter 覆盖所有嵌入维度后，会得到一个[1x5] 的张量，然后通过线性层进行预测。


```python
import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)
```

与前面一样，创建一个'FastText'类的实例：


```python
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
```

查看模型中的参数数量，我们发现该参数与第一节中的标准RNN大致相同，只有前一个模型的一半。


```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```

    The model has 2,500,301 trainable parameters
    

预训练好的向量复制到嵌入层：


```python
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
```




    tensor([[-0.1117, -0.4966,  0.1631,  ...,  1.2647, -0.2753, -0.1325],
            [-0.8555, -0.7208,  1.3755,  ...,  0.0825, -1.1314,  0.3997],
            [-0.0382, -0.2449,  0.7281,  ..., -0.1459,  0.8278,  0.2706],
            ...,
            [-0.1606, -0.7357,  0.5809,  ...,  0.8704, -1.5637, -1.5724],
            [-1.3126, -1.6717,  0.4203,  ...,  0.2348, -0.9110,  1.0914],
            [-1.5268,  1.5639, -1.0541,  ...,  1.0045, -0.6813, -0.8846]])



将未知tokens和填充tokens的初始权重归零：


```python
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
```

## 3.3 训练模型

训练模型与上一节完全相同。

初始化优化器：


```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters())
```

定义标准并将模型和标准放置在GPU上：


```python
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)
```

精度函数的计算：


```python
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc
```

定义了一个函数来训练模型。

**注**：因为我们不会使用dropout，因此实际上我们不需要使用`model.train()`，但为了保持良好的代码习惯，在这还是保留此行代码。


```python
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

定义了一个函数来测试训练好的模型。

**注意**：同样，我们也保留`model.eval()`


```python
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

通过函数得到一个epoch需要多长时间：


```python
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

最后，训练我们的模型：


```python
N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/batch.py:23: UserWarning: Batch class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    

    Epoch: 01 | Epoch Time: 0m 7s
    	Train Loss: 0.688 | Train Acc: 61.31%
    	 Val. Loss: 0.637 |  Val. Acc: 72.46%
    Epoch: 02 | Epoch Time: 0m 6s
    	Train Loss: 0.651 | Train Acc: 75.04%
    	 Val. Loss: 0.507 |  Val. Acc: 76.92%
    Epoch: 03 | Epoch Time: 0m 6s
    	Train Loss: 0.578 | Train Acc: 79.91%
    	 Val. Loss: 0.424 |  Val. Acc: 80.97%
    Epoch: 04 | Epoch Time: 0m 6s
    	Train Loss: 0.501 | Train Acc: 83.97%
    	 Val. Loss: 0.377 |  Val. Acc: 84.34%
    Epoch: 05 | Epoch Time: 0m 6s
    	Train Loss: 0.435 | Train Acc: 86.96%
    	 Val. Loss: 0.363 |  Val. Acc: 86.18%
    

获得测试精度（比上一节中的模型训练时间少很多）：


```python
model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```

    Test Loss: 0.381 | Test Acc: 85.42%
    

## 3.3 模型验证


```python
import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
```

负面评论的例子：


```python
predict_sentiment(model, "This film is terrible")
```




    2.1313092350011553e-12



正面评论的例子：


```python
predict_sentiment(model, "This film is great")
```




    1.0



## 小结

在下一节中，我们将使用卷积神经网络（CNN）进行情感分析。

