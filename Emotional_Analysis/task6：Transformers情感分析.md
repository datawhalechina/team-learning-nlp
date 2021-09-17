# task6：使用Transformer进行情感分析

在本notebook中，我们将使用在 [Attention is all you need](https://arxiv.org/abs/1706.03762) 论文中首次引入的Transformer模型。 具体来说，我们将使用 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 论文中的 BERT模型。

Transformer 模型比本教程中涵盖的其他模型都要大得多。 因此，我们将使用 [transformers library](https://github.com/huggingface/transformers) 来获取预训练的Transformer并将它们用作我们的embedding层。 我们将固定（而不训练）transformer，只训练从transformer产生的表示中学习的模型的其余部分。 在这种情况下，我们将使用双向GRU继续提取从Bert embedding后的特征。最后在fc层上输出最终的结果。

## 6.1 数据准备

首先，像往常一样，我们导入库，然后设置随机种子


```python
import torch

import random
import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

Transformer 已经用特定的词汇进行了训练，这意味着我们需要使用完全相同的词汇进行训练，并以与 Transformer 最初训练时相同的方式标记我们的数据。

幸运的是，transformers 库为每个提供的transformer 模型都有分词器。 在这种情况下，我们使用忽略大小写的 BERT 模型（即每个单词都会小写）。 我们通过加载预训练的“bert-base-uncased”标记器来实现这一点。


```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```


    Downloading:   0%|          | 0.00/232k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/466k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/570 [00:00<?, ?B/s]


`tokenizer` 有一个 `vocab` 属性，它包含我们将使用的实际词汇。 我们可以通过检查其长度来检查其中有多少单词。


```python
len(tokenizer.vocab)
```




    30522



使用`tokenizer.tokenize`方法对字符串进行分词，并统一大小写。


```python
tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')

print(tokens)
```

    ['hello', 'world', 'how', 'are', 'you', '?']
    

我们可以使用我们的词汇表使用 `tokenizer.convert_tokens_to_ids` 来数字化标记。下面的tokens是我们之前上面进行了分词和统一大小写之后的list。


```python
indexes = tokenizer.convert_tokens_to_ids(tokens)

print(indexes)
```

    [7592, 2088, 2129, 2024, 2017, 1029]
    

Transformer还接受了特殊tokens的训练，以标记句子的开头和结尾， [详细信息](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel)。 就像我们标准化padding和未知的token一样，我们也可以从`tokenizer`中获取这些。

**注意**：`tokenizer` 确实有序列开始和序列结束属性（`bos_token` 和 `eos_token`），但我们没有对此进行设置，并且不适用于我们本次训练的transformer。


```python
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)
```

    [CLS] [SEP] [PAD] [UNK]
    

我们可以通过反转词汇表来获得特殊tokens的索引


```python
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)
```

    101 102 0 100
    

或者通过tokenizer的方法直接获取它们


```python
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)
```

    101 102 0 100
    

我们需要处理的另一件事是模型是在具有定义的最大长度的序列上训练的——它不知道如何处理比训练更长的序列。 我们可以通过检查我们想要使用的转换器版本的 `max_model_input_sizes` 来获得这些输入大小的最大长度。


```python
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

print(max_input_length)
```

    512
    

之前我们使用了 `spaCy` 标记器来标记我们的示例。 然而，我们现在需要定义一个函数，我们将把它传递给我们的 `TEXT` 字段，它将为我们处理所有的标记化。 它还会将令牌的数量减少到最大长度。 请注意，我们的最大长度比实际最大长度小 2。 这是因为我们需要向每个序列附加两个标记，一个在开头，一个在结尾。


```python
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens
```

现在我们开始定义我们的字段,transformer期望将batch维度放在第一维上，所以我们设置了 `batch_first = True`。 现在我们已经有了文本的词汇数据，由transformer提供，我们设置 `use_vocab = False` 来告诉 torchtext 已经不需要切分数据了。 我们将 `tokenize_and_cut` 函数作为标记器传递。 `preprocessing` 参数是一个函数，这是我们将token转换为其索引的地方。 最后，我们定义特殊的token——注意我们将它们定义为它们的索引值而不是它们的字符串值，即“100”而不是“[UNK]”这是因为序列已经被转换为索引。

我们像以前一样定义标签字段。


```python
from torchtext.legacy import data

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/field.py:150: UserWarning: Field class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/field.py:150: UserWarning: LabelField class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    

加载数据，拆分成训练集和验证集


```python
from torchtext.legacy import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/example.py:78: UserWarning: Example class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('Example class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.', UserWarning)
    


```python
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")
```

    Number of training examples: 17500
    Number of validation examples: 7500
    Number of testing examples: 25000
    

随便看一个例子，看下具体效果如何,输出其中一个句子的one-hot向量。


```python
print(vars(train_data.examples[6]))
```

    {'text': [1042, 4140, 1996, 2087, 2112, 1010, 2023, 3185, 5683, 2066, 1037, 1000, 2081, 1011, 2005, 1011, 2694, 1000, 3947, 1012, 1996, 3257, 2003, 10654, 1011, 28273, 1010, 1996, 3772, 1006, 2007, 1996, 6453, 1997, 5965, 1043, 11761, 2638, 1007, 2003, 2058, 13088, 10593, 2102, 1998, 7815, 2100, 1012, 15339, 14282, 1010, 3391, 1010, 18058, 2014, 3210, 2066, 2016, 1005, 1055, 3147, 3752, 2068, 2125, 1037, 16091, 4003, 1012, 2069, 2028, 2518, 3084, 2023, 2143, 4276, 3666, 1010, 1998, 2008, 2003, 2320, 10012, 3310, 2067, 2013, 1996, 1000, 7367, 11368, 5649, 1012, 1000, 2045, 2003, 2242, 14888, 2055, 3666, 1037, 2235, 2775, 4028, 2619, 1010, 1998, 2023, 3185, 2453, 2022, 2062, 2084, 2070, 2064, 5047, 2074, 2005, 2008, 3114, 1012, 2009, 2003, 7078, 5923, 1011, 27017, 1012, 2023, 2143, 2069, 2515, 2028, 2518, 2157, 1010, 2021, 2009, 21145, 2008, 2028, 2518, 2157, 2041, 1997, 1996, 2380, 1012, 4276, 3773, 2074, 2005, 1996, 2197, 2184, 2781, 2030, 2061, 1012], 'label': 'neg'}
    

我们可以使用 `convert_ids_to_tokens` 将这些索引转换回可读的tokens。


```python
tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])

print(tokens)
```

    ['f', '##ot', 'the', 'most', 'part', ',', 'this', 'movie', 'feels', 'like', 'a', '"', 'made', '-', 'for', '-', 'tv', '"', 'effort', '.', 'the', 'direction', 'is', 'ham', '-', 'fisted', ',', 'the', 'acting', '(', 'with', 'the', 'exception', 'of', 'fred', 'g', '##wyn', '##ne', ')', 'is', 'over', '##wr', '##ough', '##t', 'and', 'soap', '##y', '.', 'denise', 'crosby', ',', 'particularly', ',', 'delivers', 'her', 'lines', 'like', 'she', "'", 's', 'cold', 'reading', 'them', 'off', 'a', 'cue', 'card', '.', 'only', 'one', 'thing', 'makes', 'this', 'film', 'worth', 'watching', ',', 'and', 'that', 'is', 'once', 'gage', 'comes', 'back', 'from', 'the', '"', 'se', '##met', '##ary', '.', '"', 'there', 'is', 'something', 'disturbing', 'about', 'watching', 'a', 'small', 'child', 'murder', 'someone', ',', 'and', 'this', 'movie', 'might', 'be', 'more', 'than', 'some', 'can', 'handle', 'just', 'for', 'that', 'reason', '.', 'it', 'is', 'absolutely', 'bone', '-', 'chilling', '.', 'this', 'film', 'only', 'does', 'one', 'thing', 'right', ',', 'but', 'it', 'knocks', 'that', 'one', 'thing', 'right', 'out', 'of', 'the', 'park', '.', 'worth', 'seeing', 'just', 'for', 'the', 'last', '10', 'minutes', 'or', 'so', '.']
    

尽管我们已经处理了文本的词汇表，当然也需要为标签构建词汇表。


```python
LABEL.build_vocab(train_data)
```


```python
print(LABEL.vocab.stoi)
```

    defaultdict(None, {'neg': 0, 'pos': 1})
    

像之前一样，我们创建迭代器。根据以往经验，使用最大的batch size可以使transformer获得最好的效果，当然，你也可以尝试一下使用其他的batch size，如果你的显卡比较好的话。


```python
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/iterator.py:48: UserWarning: BucketIterator class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    

# 6.2 构建模型

接下来，我们导入预训练模型。


```python
from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-uncased')
```

接下来，我们将定义我们的实际模型。

我们将使用预训练的 Transformer 模型，而不是使用embedding层来获取文本的embedding。然后将这些embedding输入GRU以生成对输入句子情绪的预测。我们通过其 config 属性从transformer中获取嵌入维度大小（称为`hidden_size`）。其余的初始化是标准的。

在前向传递中，我们将transformer包装在一个`no_grad`中，以确保不会在模型的这部分计算梯度。transformer实际上返回整个序列的embedding以及 *pooled* 输出。 [Bert模型文档](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel) 指出，汇集的输出“通常不是输入语义内容的一个很好的总结，你通常更好对整个输入序列的隐藏状态序列进行平均或合并”，因此我们不会使用它。前向传递的其余部分是循环模型的标准实现，我们在最后的时间步长中获取隐藏状态，并将其传递给一个线性层以获得我们的预测。


```python
import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output
```

我们使用标准超参数创建模型的实例。


```python
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
```

我们可以检查模型有多少参数，我们的标准型号有不到5M的参数，但这个模型有112M 幸运的是，而且这些参数中有 110M 来自transformer，我们不必再训练它们。


```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```

    The model has 112,241,409 trainable parameters
    

为了固定参数（不需要训练它们），我们需要将它们的 `requires_grad` 属性设置为 `False`。 为此，我们只需遍历模型中的所有 `named_parameters`，如果它们是 `bert` 转换器模型的一部分，我们设置 `requires_grad = False`，如微调的话，需要将`requires_grad`设置为`True`


```python
for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False
```

我们现在可以看到我们的模型有不到3M的可训练参数，这使得它几乎可以与`FastText`模型相媲美。 然而，文本仍然必须通过transformer传播，这导致训练需要更长的时间。


```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```

    The model has 2,759,169 trainable parameters
    

我们可以仔细检查可训练参数的名称，确保它们有意义。 我们可以看到，它们都是 GRU（`rnn`）和线性层（`out`）的参数。


```python
for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)
```

    rnn.weight_ih_l0
    rnn.weight_hh_l0
    rnn.bias_ih_l0
    rnn.bias_hh_l0
    rnn.weight_ih_l0_reverse
    rnn.weight_hh_l0_reverse
    rnn.bias_ih_l0_reverse
    rnn.bias_hh_l0_reverse
    rnn.weight_ih_l1
    rnn.weight_hh_l1
    rnn.bias_ih_l1
    rnn.bias_hh_l1
    rnn.weight_ih_l1_reverse
    rnn.weight_hh_l1_reverse
    rnn.bias_ih_l1_reverse
    rnn.bias_hh_l1_reverse
    out.weight
    out.bias
    

## 6.3 训练模型

按照惯例，我们构建自己的模型评价标准(损失函数)，仍然是二分类


```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters())
```


```python
criterion = nn.BCEWithLogitsLoss()
```

将模型和评价标准（损失函数）放在 GPU 上，如果你有GPU的话


```python
model = model.to(device)
criterion = criterion.to(device)
```
接下来，我们将定义函数用于：计算准确度、定义train、evalute函数以及计算训练/评估时期每一个epoch所需的时间。

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


```python
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

最后，我们将训练我们的模型。 由于transformer的尺寸的原因，这比以前的任何型号都要长得多。 即使我们没有训练任何transformer的参数，我们仍然需要通过模型传递数据，这在标准 GPU 上需要花费大量时间。


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
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/batch.py:23: UserWarning: Batch class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    

    Epoch: 01 | Epoch Time: 7m 13s
    	Train Loss: 0.502 | Train Acc: 74.41%
    	 Val. Loss: 0.270 |  Val. Acc: 89.15%
    Epoch: 02 | Epoch Time: 7m 7s
    	Train Loss: 0.281 | Train Acc: 88.49%
    	 Val. Loss: 0.224 |  Val. Acc: 91.32%
    Epoch: 03 | Epoch Time: 7m 17s
    	Train Loss: 0.239 | Train Acc: 90.67%
    	 Val. Loss: 0.211 |  Val. Acc: 91.91%
    Epoch: 04 | Epoch Time: 7m 14s
    	Train Loss: 0.206 | Train Acc: 91.81%
    	 Val. Loss: 0.206 |  Val. Acc: 92.01%
    Epoch: 05 | Epoch Time: 7m 15s
    	Train Loss: 0.188 | Train Acc: 92.63%
    	 Val. Loss: 0.211 |  Val. Acc: 91.92%
    

我们将加载为我们提供最佳验证集上损失值的参数，并在测试集上应用这些参数 - 并在测试集上达到了最优的结果。


```python
model.load_state_dict(torch.load('tut6-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```

    Test Loss: 0.209 | Test Acc: 91.58%
    

## 6.4 模型验证

然后我们将使用该模型来测试一些序列的情绪。 我们对输入序列进行标记，将其修剪到最大长度，将特殊token添加到任一侧，将其转换为张量，使用unsqueeze函数增加一维，然后将其传递给我们的模型。


```python
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
```


```python
predict_sentiment(model, tokenizer, "This film is terrible")
```




    0.03391794115304947




```python
predict_sentiment(model, tokenizer, "This film is great")
```




    0.8869886994361877


