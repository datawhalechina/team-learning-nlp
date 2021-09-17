# task5：多类型情感分析

在之前的所有学习中，我们的数据集对于情感的分析只有两个分类：正面或负面。当我们只有两个类时，我们的输出可以是单个标量，范围在 0 和 1 之间，表示示例属于哪个类。当我们有 2 个以上的例子时，我们的输出必须是一个 $C$ 维向量，其中 $C$ 是类的数量。

在本次学习中，我们将对具有 6 个类的数据集执行分类。请注意，该数据集实际上并不是情感分析数据集，而是问题数据集，任务是对问题所属的类别进行分类。但是，本次学习中涵盖的所有内容都适用于任何包含属于 $C$ 类之一的输入序列的示例的数据集。

下面，我们设置字段并加载数据集，与之前不同的是：

第一，我们不需要在 `LABEL` 字段中设置 `dtype`。在处理多类问题时，PyTorch 期望标签被数字化为`LongTensor`。

第二，这次我们使用的是`TREC`数据集而不是`IMDB`数据集。 `fine_grained` 参数允许我们使用细粒度标签（其中有50个类）或不使用（在这种情况下它们将是6个类）。


```python
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',tokenizer_language = 'en_core_web_sm')

LABEL = data.LabelField()

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
```

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\util.py:740: UserWarning: [W094] Model 'en_core_web_sm' (2.2.0) specifies an under-constrained spaCy version requirement: >=2.2.0. This can lead to compatibility problems with older versions, or as new spaCy versions are released, because the model may say it's compatible when it's not. Consider changing the "spacy_version" in your meta.json to a version range, with a lower and upper pin. For example: >=3.1.2,<3.2.0
      warnings.warn(warn_msg)
    


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-1-78a725e4bc2e> in <module>
          9 torch.backends.cudnn.deterministic = True
         10 
    ---> 11 TEXT = data.Field(tokenize = 'spacy',tokenizer_language = 'en_core_web_sm')
         12 
         13 LABEL = data.LabelField()
    

    D:\ProgramData\Anaconda3\lib\site-packages\torchtext\legacy\data\field.py in __init__(self, sequential, use_vocab, init_token, eos_token, fix_length, dtype, preprocessing, postprocessing, lower, tokenize, tokenizer_language, include_lengths, batch_first, pad_token, unk_token, pad_first, truncate_first, stop_words, is_target)
        159         # in case the tokenizer isn't picklable (e.g. spacy)
        160         self.tokenizer_args = (tokenize, tokenizer_language)
    --> 161         self.tokenize = get_tokenizer(tokenize, tokenizer_language)
        162         self.include_lengths = include_lengths
        163         self.batch_first = batch_first
    

    D:\ProgramData\Anaconda3\lib\site-packages\torchtext\data\utils.py in get_tokenizer(tokenizer, language)
        113             import spacy
        114             try:
    --> 115                 spacy = spacy.load(language)
        116             except IOError:
        117                 # Model shortcuts no longer work in spaCy 3.0+, try using fullnames
    

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\__init__.py in load(name, vocab, disable, exclude, config)
         49     RETURNS (Language): The loaded nlp object.
         50     """
    ---> 51     return util.load_model(
         52         name, vocab=vocab, disable=disable, exclude=exclude, config=config
         53     )
    

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\util.py in load_model(name, vocab, disable, exclude, config)
        319             return get_lang_class(name.replace("blank:", ""))()
        320         if is_package(name):  # installed as package
    --> 321             return load_model_from_package(name, **kwargs)
        322         if Path(name).exists():  # path to model data directory
        323             return load_model_from_path(Path(name), **kwargs)
    

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\util.py in load_model_from_package(name, vocab, disable, exclude, config)
        352     """
        353     cls = importlib.import_module(name)
    --> 354     return cls.load(vocab=vocab, disable=disable, exclude=exclude, config=config)
        355 
        356 
    

    D:\ProgramData\Anaconda3\lib\site-packages\en_core_web_sm\__init__.py in load(**overrides)
         10 
         11 def load(**overrides):
    ---> 12     return load_model_from_init_py(__file__, **overrides)
    

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\util.py in load_model_from_init_py(init_file, vocab, disable, exclude, config)
        512     if not model_path.exists():
        513         raise IOError(Errors.E052.format(path=data_path))
    --> 514     return load_model_from_path(
        515         data_path,
        516         vocab=vocab,
    

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\util.py in load_model_from_path(model_path, meta, vocab, disable, exclude, config)
        386     config_path = model_path / "config.cfg"
        387     overrides = dict_to_dot(config)
    --> 388     config = load_config(config_path, overrides=overrides)
        389     nlp = load_model_from_config(config, vocab=vocab, disable=disable, exclude=exclude)
        390     return nlp.from_disk(model_path, exclude=exclude, overrides=overrides)
    

    D:\ProgramData\Anaconda3\lib\site-packages\spacy\util.py in load_config(path, overrides, interpolate)
        543     else:
        544         if not config_path or not config_path.exists() or not config_path.is_file():
    --> 545             raise IOError(Errors.E053.format(path=config_path, name="config.cfg"))
        546         return config.from_disk(
        547             config_path, overrides=overrides, interpolate=interpolate
    

    OSError: [E053] Could not read config.cfg from D:\ProgramData\Anaconda3\lib\site-packages\en_core_web_sm\en_core_web_sm-2.2.0\config.cfg


下面我们看一个训练集的示例


```python
vars(train_data[-1])
```




    {'text': ['What', 'is', 'a', 'Cartesian', 'Diver', '?'], 'label': 'DESC'}



接下来，我们将构建词汇表。 由于这个数据集很小（只有约 3800 个训练样本），它的词汇量也非常小（约 7500 个不同单词，即one-hot向量为7500维），这意味着我们不需要像以前一样在词汇表上设置“max_size”。


```python
MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)
```

接下来，我们可以检查标签。

6 个标签（对于非细粒度情况）对应于数据集中的 6 类问题：
- `HUM`：关于人类的问题
- `ENTY`：关于实体的问题的
- `DESC`：关于要求提供描述的问题
- `NUM`：关于答案为数字的问题
- `LOC`：关于答案是位置的问题
- `ABBR`：关于询问缩写的问题


```python
print(LABEL.vocab.stoi)
```

    defaultdict(None, {'HUM': 0, 'ENTY': 1, 'DESC': 2, 'NUM': 3, 'LOC': 4, 'ABBR': 5})
    

与往常一样，我们设置了迭代器。


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
    

我们将使用上一个notebook中的CNN模型，但是教程中涵盖的任何模型都适用于该数据集。 唯一的区别是现在 `output_dim` 是 $C$维而不是 $2$维。


```python
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)
```

我们定义我们的模型，确保将输出维度： `OUTPUT_DIM` 设置为 $C$。 我们可以通过使用 `LABEL` 词汇的大小轻松获得 $C$，就像我们使用 `TEXT` 词汇的长度来获取输入词汇的大小一样。

此数据集中的示例比 IMDb 数据集中的示例小很多，因此我们将使用较小的`filter`大小。


```python
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
```

检查参数的数量，我们可以看到较小的`filter`大小意味着我们的参数是 IMDb 数据集上 CNN 模型的三分之一。


```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```

    The model has 841,806 trainable parameters
    

之后，我们将加载我们的预训练embedding。


```python
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
```




    tensor([[-0.1117, -0.4966,  0.1631,  ...,  1.2647, -0.2753, -0.1325],
            [-0.8555, -0.7208,  1.3755,  ...,  0.0825, -1.1314,  0.3997],
            [ 0.1638,  0.6046,  1.0789,  ..., -0.3140,  0.1844,  0.3624],
            ...,
            [-0.3110, -0.3398,  1.0308,  ...,  0.5317,  0.2836, -0.0640],
            [ 0.0091,  0.2810,  0.7356,  ..., -0.7508,  0.8967, -0.7631],
            [ 0.5831, -0.2514,  0.4156,  ..., -0.2735, -0.8659, -1.4063]])



然后将用0来初始化未知的权重和padding参数。


```python
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
```

与之前notebook的另一个不同之处是我们的损失函数。  `BCEWithLogitsLoss` 一般用来做二分类，而 `CrossEntropyLoss`用来做多分类，`CrossEntropyLoss` 对我们的模型输出执行 *softmax* 函数，损失由该函数和标签之间的 *交叉熵 * 给出。

一般来说：
- 当我们的示例仅属于 $C$ 类之一时，使用 `CrossEntropyLoss`
- 当我们的示例仅属于 2 个类（0 和 1）时使用 `BCEWithLogitsLoss`，并且也用于我们的示例属于 0 和 $C$ 之间的类（也称为多标签分类）的情况。


```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
```

之前，我们有一个函数可以计算二进制标签情况下的准确度，我们说如果值超过 0.5，那么我们会假设它是正的。 在我们有超过 2 个类的情况下，我们的模型输出一个 $C$ 维向量，其中每个元素的值是示例属于该类的置信度。

例如，在我们的标签中，我们有：'HUM' = 0、'ENTY' = 1、'DESC' = 2、'NUM' = 3、'LOC' = 4 和 'ABBR' = 5。如果我们的输出 模型是这样的：**[5.1, 0.3, 0.1, 2.1, 0.2, 0.6]** 这意味着该模型确信该示例属于第 0 类：这是一个关于人类的问题，并且略微相信该示例属于该第3类：关于数字的问题。

我们通过执行 `argmax` 来获取批次中每个元素的预测最大值的索引，然后计算它与实际标签相等的次数来计算准确度。 然后我们对整个批次进行平均。


```python
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
```

训练循环与之前类似，`CrossEntropyLoss`期望输入数据为 **[batch size, n classes]** ，标签为 **[batch size]** 。

标签默认需要是一个 `LongTensor`类型的数据，因为我们没有像以前那样将 `dtype` 设置为 `FloatTensor`。


```python
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.label)
        
        acc = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

像之前一样对循环进行评估


```python
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

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

接下来，训练模型


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
        torch.save(model.state_dict(), 'tut5-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
```

    /home/ben/miniconda3/envs/pytorch17/lib/python3.8/site-packages/torchtext-0.9.0a0+c38fd42-py3.8-linux-x86_64.egg/torchtext/data/batch.py:23: UserWarning: Batch class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.
      warnings.warn('{} class will be retired soon and moved to torchtext.legacy. Please see the most recent release notes for further information.'.format(self.__class__.__name__), UserWarning)
    

    Epoch: 01 | Epoch Time: 0m 0s
    	Train Loss: 1.312 | Train Acc: 47.11%
    	 Val. Loss: 0.947 |  Val. Acc: 66.41%
    Epoch: 02 | Epoch Time: 0m 0s
    	Train Loss: 0.870 | Train Acc: 69.18%
    	 Val. Loss: 0.741 |  Val. Acc: 74.14%
    Epoch: 03 | Epoch Time: 0m 0s
    	Train Loss: 0.675 | Train Acc: 76.32%
    	 Val. Loss: 0.621 |  Val. Acc: 78.49%
    Epoch: 04 | Epoch Time: 0m 0s
    	Train Loss: 0.506 | Train Acc: 83.97%
    	 Val. Loss: 0.547 |  Val. Acc: 80.32%
    Epoch: 05 | Epoch Time: 0m 0s
    	Train Loss: 0.373 | Train Acc: 88.23%
    	 Val. Loss: 0.487 |  Val. Acc: 82.92%
    

最后，在测试集上运行我们的模型


```python
model.load_state_dict(torch.load('tut5-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```

    Test Loss: 0.415 | Test Acc: 86.07%
    

类似于我们创建一个函数来预测任何给定句子的情绪，我们现在可以创建一个函数来预测给定问题的类别。

这里唯一的区别是，我们没有使用 sigmoid 函数将输入压缩在 0 和 1 之间，而是使用 `argmax` 来获得最高的预测类索引。 然后我们使用这个索引和标签 vocab 来获得可读的标签string。


```python
import spacy
nlp = spacy.load('en_core_web_sm')

def predict_class(model, sentence, min_len = 4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()
```

现在，让我们在几个不同的问题上尝试一下……


```python
pred_class = predict_class(model, "Who is Keyser Söze?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')
```

    Predicted class is: 0 = HUM
    


```python
pred_class = predict_class(model, "How many minutes are in six hundred and eighteen hours?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')
```

    Predicted class is: 3 = NUM
    


```python
pred_class = predict_class(model, "What continent is Bulgaria in?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')
```

    Predicted class is: 4 = LOC
    


```python
pred_class = predict_class(model, "What does WYSIWYG stand for?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')
```

    Predicted class is: 5 = ABBR
    
