import tensorflow.contrib.keras as kr
import tensorflow as tf
import numpy as np
import string

vocabPath = "./model/biLSTM/vocab.txt"
savePath = "./model/biLSTM/lstm_model"


def preprocess_sentence(sent):
    new_sent = ''
    for i in range(len(sent)):
        if sent[i] in string.punctuation:
            if i > 0 and i < len(sent) - 1:
                if sent[i] in ",." and sent[i-1].isdigit() and sent[i+1].isdigit():
                    new_sent += sent[i]
                    continue
                if sent[i] == "%" and sent[i-1].isdigit():
                    new_sent += sent[i]
                    continue
                if sent[i] == "$" and (sent[i-1].isdigit() or sent[i+1].isdigit()):
                    new_sent += sent[i]
                    continue
                if sent[i-1] != ' ':
                    new_sent += ' ' + sent[i]
                elif sent[i+1] != ' ':
                    new_sent += sent[i] + ' '
                else:
                    new_sent += sent[i]
            elif i == 0:
                if sent[i] == "$" and sent[i+1].isdigit():
                    new_sent += sent[i]
                    continue
                if sent[i+1] != ' ':
                    new_sent += sent[i] + ' '
                else:
                    new_sent += sent[i]
            else:
                if sent[i] == "%" and sent[i-1].isdigit():
                    new_sent += sent[i]
                    continue
                if sent[i] == "$" and sent[i-1].isdigit():
                    new_sent += sent[i]
                    continue
                if sent[i-1] != ' ':
                    new_sent += ' ' + sent[i]
                else:
                    new_sent += sent[i]
        else:
            new_sent += sent[i]
    return new_sent.strip().lower()


num_classes = 10
vocab_size = 400000
embedding_dim = 100

vocab = []
with open(vocabPath, "r", encoding="utf-8") as fp:
    for word in fp.readlines():
        vocab.extend(word.split())

word_to_id = dict(zip(vocab, range(vocab_size)))
embedding = np.zeros((vocab_size, embedding_dim))
seq_length = 41

hiddenSizes = [64]  # 定义LSTM的隐藏层（一层，128个神经元）  128
epsilon = 5
learning_rate = 1e-3


# 构建adversarailLSTM模型
class AdversarailLSTM(object):

    def __init__(self, wordEmbedding):
        # 定义输入
        self.inputX = tf.placeholder(tf.int32, [None, seq_length], name='inputX')
        self.inputY = tf.placeholder(tf.int32, [None, num_classes], name='inputY')

        self.dropoutKeepProb = tf.placeholder(tf.float64, name='keep_prob')

        # 词嵌入层
        with tf.name_scope("wordEmbedding"):
            wordEmbedding = tf.Variable(initial_value=wordEmbedding)
            self.embeddedWords = tf.nn.embedding_lookup(wordEmbedding, self.inputX)

        # 计算softmax交叉熵损失
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._Bi_LSTMAttention(self.embeddedWords)
                # self.y_pred_cls = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")
                self.y_pred_cls = tf.argmax(tf.nn.softmax(self.predictions),
                                            1)  # 预测类别 tf.argmax：返回每一行或每一列的最大值 1为里面（每一行），0为外面（每一列）
                # losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
                loss = tf.reduce_mean(losses)

        with tf.name_scope("perturloss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                print("perturbSize:{}".format(perturWordEmbedding))
                perturPredictions = self._Bi_LSTMAttention(perturWordEmbedding)
                # perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
                perturLosses = tf.nn.softmax_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
                perturLoss = tf.reduce_mean(perturLosses)

        self.loss = loss + perturLoss

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(self.loss)
        # 将梯度应用到变量下，生成训练器
        self.trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 准确率
        correct_pred = tf.equal(tf.argmax(self.inputY, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # self.loss = loss

    def _Bi_LSTMAttention(self, embeddedWords):
        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            fwHiddenLayers = []
            bwHiddenLayers = []
            for idx, hiddenSize in enumerate(hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向网络结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 定义反向网络结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                fwHiddenLayers.append(lstmFwCell)
                bwHiddenLayers.append(lstmBwCell)

            # 实现多层的LSTM结构， state_is_tuple=True，则状态会以元祖的形式组合(h, c)，否则列向拼接
            fwMultiLstm = tf.nn.rnn_cell.MultiRNNCell(cells=fwHiddenLayers, state_is_tuple=True)
            bwMultiLstm = tf.nn.rnn_cell.MultiRNNCell(cells=bwHiddenLayers, state_is_tuple=True)
            # 采用动态rnn，可以动态地输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元组(output_fw, output_bw), 其中两个元素的维度都是[batch_size, max_time, hidden_size], fw和bw的hiddensize一样
            # self.current_state是最终的状态，二元组(state_fw, state_bw), state_fw=[batch_size, s], s是一个元组(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(fwMultiLstm, bwMultiLstm,
                                                                          self.embeddedWords, dtype=tf.float64,
                                                                          scope="bi-lstm" + str(idx))

        # 在bi-lstm+attention论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到attention的输出
            output = self.attention(H)
            outputSize = hiddenSizes[-1]
            print("outputSize:{}".format(outputSize))

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW", dtype=tf.float64,
                shape=[outputSize, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, dtype=tf.float64, shape=[num_classes]), name="outputB")

            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")

            return predictions

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层lstm神经元的数量
        hiddenSize = hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1, dtype=tf.float64))

        # 对bi-lstm的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size], 计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1], 每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, seq_length])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, seq_length, 1]))

        # 将三维压缩成二维sequeezeR = [batch_size, hissen_size]
        sequeezeR = tf.squeeze(r)

        sentenceRepren = tf.tanh(sequeezeR)

        # 对attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output

    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        mean = tf.matmul(weights, wordEmbedding)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)

        var = tf.matmul(weights, powWordEmbedding)
        stddev = tf.sqrt(1e-6 + var)

        return (wordEmbedding - mean) / stddev

    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, epsilon)
        # print("perturbSize:{}".format(embedded+perturb))
        return embedded + perturb

    def _scaleL2(self, x, norm_length):
        # shape(x) = [batch, num_step, d]
        # divide x by max(abs(x)) for a numerically stable L2 norm
        # 2norm(x) = a * 2norm(x/a)
        # scale over the full sequence, dim(1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit


lstm = AdversarailLSTM(embedding)
session = tf.Session()
saver = tf.train.Saver()
saver.restore(sess=session, save_path=savePath)

def predict11(predict_sentences, probability_threshold=0.3):
    """
    将文件转换为id表示,并且将每个单独的样本长度固定为pad_max_lengtn
    """
    predict_sentences.append("")   # 不这样处理可能会报错误：tensorflow Invalid argument: In[0] is not a matrix
    data_id = []
    # 将文本内容转换为对应的id形式
    for psi in predict_sentences:

        data_id.append([word_to_id[x] for x in preprocess_sentence(psi).split() if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)
    feed_dict = {
        lstm.inputX: x_pad,
        lstm.dropoutKeepProb: 1.0
    }
    predict_result = session.run(tf.nn.softmax(lstm.predictions), feed_dict=feed_dict)
    # print(predict_result)
    result = []
    for i in predict_result[:-1]:
        if max(i) > probability_threshold:
            result.append(i.argmax()+1)
        else:
            result.append(0)
    return result


if __name__ == '__main__':
    predict_sentences = ["In the sixtieth ceremony , where were all of the winners from ?",  # 7
                         "On how many devices has the app \" CF SHPOP ! \" been installed ?",  # 1
                         "List center - backs by what their transfer _ fee was .",  # 5
                         "can you tell me what is arkansas 's population on the date july 1st of 2002 ?",  # 1
                         "show the way the number of likes were distributed .",  # 7
                         "is it true that people living on average depends on higher gdp of a country"  # 10
                         ]

    print(predict11(predict_sentences))
