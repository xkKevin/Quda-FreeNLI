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

hiddenSizes = [64]  # ??????LSTM????????????????????????128???????????????  128
epsilon = 5
learning_rate = 1e-3


# ??????adversarailLSTM??????
class AdversarailLSTM(object):

    def __init__(self, wordEmbedding):
        # ????????????
        self.inputX = tf.placeholder(tf.int32, [None, seq_length], name='inputX')
        self.inputY = tf.placeholder(tf.int32, [None, num_classes], name='inputY')

        self.dropoutKeepProb = tf.placeholder(tf.float64, name='keep_prob')

        # ????????????
        with tf.name_scope("wordEmbedding"):
            wordEmbedding = tf.Variable(initial_value=wordEmbedding)
            self.embeddedWords = tf.nn.embedding_lookup(wordEmbedding, self.inputX)

        # ??????softmax???????????????
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._Bi_LSTMAttention(self.embeddedWords)
                # self.y_pred_cls = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")
                self.y_pred_cls = tf.argmax(tf.nn.softmax(self.predictions),
                                            1)  # ???????????? tf.argmax?????????????????????????????????????????? 1???????????????????????????0????????????????????????
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
        # ?????????????????????????????????????????????
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # ????????????,?????????????????????
        gradsAndVars = optimizer.compute_gradients(self.loss)
        # ?????????????????????????????????????????????
        self.trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # ?????????
        correct_pred = tf.equal(tf.argmax(self.inputY, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # self.loss = loss

    def _Bi_LSTMAttention(self, embeddedWords):
        # ??????????????????LSTM???????????????
        with tf.name_scope("Bi-LSTM"):
            fwHiddenLayers = []
            bwHiddenLayers = []
            for idx, hiddenSize in enumerate(hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # ????????????????????????
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # ????????????????????????
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                fwHiddenLayers.append(lstmFwCell)
                bwHiddenLayers.append(lstmBwCell)

            # ???????????????LSTM????????? state_is_tuple=True???????????????????????????????????????(h, c)?????????????????????
            fwMultiLstm = tf.nn.rnn_cell.MultiRNNCell(cells=fwHiddenLayers, state_is_tuple=True)
            bwMultiLstm = tf.nn.rnn_cell.MultiRNNCell(cells=bwHiddenLayers, state_is_tuple=True)
            # ????????????rnn?????????????????????????????????????????????????????????????????????????????????
            # outputs???????????????(output_fw, output_bw), ?????????????????????????????????[batch_size, max_time, hidden_size], fw???bw???hiddensize??????
            # self.current_state??????????????????????????????(state_fw, state_bw), state_fw=[batch_size, s], s???????????????(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(fwMultiLstm, bwMultiLstm,
                                                                          self.embeddedWords, dtype=tf.float64,
                                                                          scope="bi-lstm" + str(idx))

        # ???bi-lstm+attention?????????????????????????????????????????????
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # ??????attention?????????
            output = self.attention(H)
            outputSize = hiddenSizes[-1]
            print("outputSize:{}".format(outputSize))

        # ?????????????????????
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
        ??????Attention?????????????????????????????????
        """
        # ??????????????????lstm??????????????????
        hiddenSize = hiddenSizes[-1]

        # ???????????????????????????????????????????????????
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1, dtype=tf.float64))

        # ???bi-lstm??????????????????????????????????????????
        M = tf.tanh(H)

        # ???W???M??????????????????W=[batch_size, time_step, hidden_size], ???????????????????????????[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1], ?????????????????????????????????????????????????????????
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # ???newM??????????????????[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, seq_length])

        # ???softmax??????????????????[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # ???????????????alpha?????????H????????????????????????????????????????????????
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, seq_length, 1]))

        # ????????????????????????sequeezeR = [batch_size, hissen_size]
        sequeezeR = tf.squeeze(r)

        sentenceRepren = tf.tanh(sequeezeR)

        # ???attention??????????????????dropout??????
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output

    def _normalize(self, wordEmbedding, weights):
        """
        ???word embedding ??????????????????????????????
        """
        mean = tf.matmul(weights, wordEmbedding)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)

        var = tf.matmul(weights, powWordEmbedding)
        stddev = tf.sqrt(1e-6 + var)

        return (wordEmbedding - mean) / stddev

    def _addPerturbation(self, embedded, loss):
        """
        ???????????????word embedding
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
    ??????????????????id??????,?????????????????????????????????????????????pad_max_lengtn
    """
    predict_sentences.append("")   # ????????????????????????????????????tensorflow Invalid argument: In[0] is not a matrix
    data_id = []
    # ?????????????????????????????????id??????
    for psi in predict_sentences:

        data_id.append([word_to_id[x] for x in preprocess_sentence(psi).split() if x in word_to_id])

    # ??????keras?????????pad_sequences????????????pad???????????????
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
