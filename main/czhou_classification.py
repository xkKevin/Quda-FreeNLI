import numpy as np
import tensorflow.contrib.keras as kr
import tensorflow as tf
import string

vocabPath = "./model/glove.6B.100d.txt"
savePath = "./model/c1_cnn_2"


def loadGloVe(filename):
    vocab = []
    embd = []
    print('Loading GloVe!')
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append([float(ei) for ei in row[1:]])
    file.close()
    print('Completed!')
    return vocab, embd


categories = ['Others', 'Retrieve Value', 'Filter', 'Compute Derived Value', 'Find Extremum', 'Sort',
              'Determine Range', 'Characterize Distribution', 'Find Anomalies', 'Cluster', 'Correlate']

# num_classes = len(categories)
num_classes = 10
seq_length = 35

vocab, embd = loadGloVe(vocabPath)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
word_to_id = dict(zip(vocab, range(vocab_size)))

# ======================================================CNN Model Start===============================================
# 输入内容及对应的标签
input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
# dropout的损失率
keep_prob = tf.placeholder(tf.float64, name='keep_prob')

# 词向量映射;实际上此处的词向量并不是用的预训练好的词向量，而是未经任何训练直接生成了一个矩阵，将此矩阵作为词向量矩阵使用，效果也还不错。
# 若使用训练好的词向量，或许训练此次文本分类的模型时会更快，更好。
# embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

num_filters = 256
kernel_size = 5
hidden_dim = 128
learning_rate = 1e-3
dropout_keep_prob = 0.5

num_epochs = 20
batch_size = 64
print_per_batch = 20  # 每多少轮输出一次结果
# save_per_batch = 5  # 每多少轮存入tensorboard


# CNN layer
conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size, name='conv')  # num_filters = 256 这是个啥
''' https://blog.csdn.net/khy19940520/article/details/89934335
tf.layers.conv1d：一维卷积一般用于处理文本数据，常用语自然语言处理中，输入一般是文本经过embedding的二维数据。
    inputs： 输入tensor， 维度(batch_size, seq_length, embedding_dim) 是一个三维的tensor；其中，
        batch_size指每次输入的文本数量；
        seq_length指每个文本的词语数或者单字数；
        embedding_dim指每个词语或者每个字的向量长度；
        例如每次训练输入2篇文本，每篇文本有100个词，每个词的向量长度为20，那input维度即为(2, 100, 20)。
    filters：过滤器（卷积核）的数目
    kernel_size：卷积核的大小，卷积核本身应该是二维的，这里只需要指定一维，因为第二个维度即长度与词向量的长度一致，卷积核只能从上往下走，不能从左往右走，即只能按照文本中词的顺序，也是列的顺序。
'''
# global max pooling layer
gmp = tf.reduce_max(conv, reduction_indices=[1],
                    name='gmp')  # https://blog.csdn.net/lllxxq141592654/article/details/85345864

# 全连接层，后面接dropout以及relu激活
fc = tf.layers.dense(gmp, hidden_dim, name='fc1')  # hidden_dim：128
''' https://blog.csdn.net/yangfengling1023/article/details/81774580
dense ：全连接层  inputs：输入该网络层的数据；units：输出的维度大小，改变inputs的最后一维
'''
fc = tf.nn.dropout(fc, keep_prob)
fc = tf.nn.relu(fc)

# 分类器
logits = tf.layers.dense(fc, num_classes, name='fc2')
y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别 tf.argmax：返回每一行或每一列的最大值 1为里面（每一行），0为外面（每一列）

# 损失函数，交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
loss = tf.reduce_mean(cross_entropy)
# 优化器
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 准确率
correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred_cls)
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# ======================================================CNN Model End============================================

# 创建session
session = tf.Session()
saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.restore(sess=session, save_path=savePath)


def preprocess_sentence(sent):
    new_sent = ''
    for i in range(len(sent)):
        if sent[i] in string.punctuation:
            if i > 0 and i < len(sent) - 1:
                if sent[i - 1] != ' ':
                    new_sent += ' ' + sent[i]
                elif sent[i + 1] != ' ':
                    new_sent += sent[i] + ' '
                else:
                    new_sent += sent[i]
            elif i == 0:
                if sent[i + 1] != ' ':
                    new_sent += sent[i] + ' '
                else:
                    new_sent += sent[i]
            else:
                if sent[i - 1] != ' ':
                    new_sent += ' ' + sent[i]
                else:
                    new_sent += sent[i]
        else:
            new_sent += sent[i]
    return new_sent.lower()


def predict(predict_sentences):
    """
    将文件转换为id表示,并且将每个单独的样本长度固定为pad_max_lengtn
    """

    data_id = []
    # 将文本内容转换为对应的id形式
    for i in range(len(predict_sentences)):
        data_id.append([word_to_id[x] for x in predict_sentences[i].lower().strip().split() if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)
    ''' https://blog.csdn.net/TH_NUM/article/details/80904900
    pad_sequences(sequences, maxlen=None, dtype=’int32’, padding=’pre’, truncating=’pre’, value=0.) 
        sequences：浮点数或整数构成的两层嵌套列表
        maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
        dtype：返回的numpy array的数据类型
        padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
        truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
        value：浮点数，此值将在填充时代替默认的填充值0
    '''
    feed_dict = {
        input_x: x_pad,
        keep_prob: 1.0
    }
    predict_result = session.run(y_pred_cls, feed_dict=feed_dict)
    predict_result = [i + 1 for i in predict_result]
    return predict_result


def predict11(predict_sentences, probability_threshold=0.3):  # 0.26189747
    """
    将文件转换为id表示,并且将每个单独的样本长度固定为pad_max_lengtn
    """
    data_id = []
    # 将文本内容转换为对应的id形式
    for psi in predict_sentences:

        data_id.append([word_to_id[x] for x in preprocess_sentence(psi).strip().split() if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)
    feed_dict = {
        input_x: x_pad,
        keep_prob: 1.0
    }
    predict_result = session.run(tf.nn.softmax(logits), feed_dict=feed_dict)
    result = []
    for i in predict_result:
        if max(i) > probability_threshold:
            result.append(i.argmax()+1)
        else:
            result.append(0)
    return result


predict_sentences = ["In the sixtieth ceremony , where were all of the winners from ?",  # 7
                     "On how many devices has the app \" CF SHPOP ! \" been installed ?",  # 1
                     "List center - backs by what their transfer _ fee was ."]  # 5

print(predict11(predict_sentences))
