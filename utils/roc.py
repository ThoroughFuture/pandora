import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score

def roc(y_true, y_score, epoch, name, log_dir, sklearn=False, output_auc=False):

    pos_label = 1

    num_positive_examples = (y_true == pos_label).sum()
    num_negtive_examples = len(y_true) - num_positive_examples

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score)


    for i in np.flip(np.argsort(y_score)):

        if y_score[i] != score:
            fpr.append(fp / num_negtive_examples)
            tpr.append(tp / num_positive_examples)
            thresholds.append(score)
            score = y_score[i]

        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / num_negtive_examples)
    tpr.append(tp / num_positive_examples)
    thresholds.append(score)

    maxindex = (np.array(tpr) - np.array(fpr)).tolist().index(max(np.array(tpr) - np.array(fpr)))
    cutoff = thresholds[maxindex]  
    index = thresholds.index(cutoff)

    se = tpr[index]  
    sp = 1-fpr[index]  

    if sklearn:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(len(tpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * tpr[i + 1]


    y_pred = (y_score >= cutoff).astype(int)
    

    tp_final = np.sum((y_pred == 1) & (y_true == 1))
    fp_final = np.sum((y_pred == 1) & (y_true == 0))
    fn_final = np.sum((y_pred == 0) & (y_true == 1))
    tn_final = np.sum((y_pred == 0) & (y_true == 0))
    
  
    if tp_final + fp_final == 0:
        precision = 0
    else:
        precision = tp_final / (tp_final + fp_final)
    
    if tp_final + fn_final == 0:
        recall = 0
    else:
        recall = tp_final / (tp_final + fn_final)
    
  
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    correct = 0
    for i in range(y_score.shape[0]):
        if y_score[i] >= cutoff and y_true[i] == 1:
            correct += 1
        elif y_score[i] < cutoff and y_true[i] == 0:
            correct += 1

    fig, ax = plt.subplots()
    plt.plot([0, 1], '--')
    plt.plot(fpr[index], tpr[index], 'bo')
    ax.text(fpr[index], tpr[index] + 0.02, f'cut_off={cutoff:.3f}', fontdict={'fontsize': 10})
    plt.plot(fpr, tpr)
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")

    text = f'AUC:{auc:.3f}\nSE:{se:.3f}\nSP:{sp:.3f}\nF1:{f1:.3f}\nACC:{(correct / y_score.shape[0]*100):.3f}%\n'

    ax.text(0.6, 0.05, text, fontsize=12)

    if os.path.exists(f'{log_dir}/logging/roc_image_{name}'):
        aaaa = 1
    else:
        os.mkdir(f'{log_dir}/logging/roc_image_{name}')
    
    plt.savefig(f'{log_dir}/logging/roc_image_{name}/AUC_{epoch}.png')
    plt.savefig(f'{log_dir}/logging/roc_image_{name}/AUC_{epoch}.pdf', format='pdf')
    
    if output_auc:
        return auc
    return cutoff




