from simpletransformers.classification import MultiLabelClassificationModel
import string


num_labels = 10
model = MultiLabelClassificationModel('roberta', 'model/roberta/', num_labels=num_labels,  use_cuda=False)


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


def predict(predict_sentences):
    preprocess_s = []
    for psi in predict_sentences:
        preprocess_s.append(preprocess_sentence(psi))
    predictions, raw_outputs = model.predict(preprocess_s)  # predictions: the labels of each sentence; raw_outputs: the rate of each label
    result = []
    for ri in predictions:
        # labels = np.where(ri)[0]
        labels = [li + 1 for li in range(num_labels) if ri[li]]
        if len(labels):
            result.append(labels)
        else:
            result.append([0])
    return result


if __name__ == '__main__':
    predict_sentences = ["In the sixtieth ceremony , where were all of the winners from ?",  # 6,2
                         "On how many devices has the app \" CF SHPOP ! \" been installed ?",  # 1
                         "List center - backs by what their transfer _ fee was .",  # 5
                         "can you tell me what is arkansas 's population on the date july 1st of 2002 ?",  # 1
                         "show the way the number of likes were distributed .",  # 7
                         "is it true that people living on average depends on higher gdp of a country"  # 10
                         ]
    print(predict(predict_sentences))
