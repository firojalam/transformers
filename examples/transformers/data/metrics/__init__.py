# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import sys
import logging
import numpy as np
import re
import pandas as pd
logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    #from sklearn.metrics import matthews_corrcoef, f1_score
    from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support, classification_report, \
        confusion_matrix, roc_auc_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def format_conf_mat(y_true,y_pred):

        conf_mat = pd.crosstab(np.array(y_true), np.array(y_pred), rownames=['gold'], colnames=['pred'], margins=True)
        pred_columns = conf_mat.columns.tolist()
        gold_rows = conf_mat.index.tolist()
        conf_mat_str = ""
        header = "Pred\nGold"
        for h in pred_columns:
            header = header + "\t" + str(h)
        conf_mat_str = header + "\n"
        index = 0
        for r_index, row in conf_mat.iterrows():
            row_str = str(gold_rows[index])  # for class label (name)
            index += 1
            for col_item in row:
                row_str = row_str + "\t" + str(col_item)
            conf_mat_str = conf_mat_str + row_str + "\n"
        return conf_mat_str


    def format_classifaction_report(report):
        report_data = "class_label\tP\tR\tF1\tsupport\n"
        for k,row in report.items():
            if(k=="accuracy"):
                continue
            report_data = report_data+str(k)+"\t"+str(row['precision'])+"\t"+str(row['recall'])+"\t"+str(row['f1-score'])+"\t"+str(row['support'])+"\n"
        return report_data

    def acc_and_p_r_f_per_class(preds, labels, label_list):
        acc = simple_accuracy(preds, labels)
        prf = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='weighted')
        # prf_per_class = precision_recall_fscore_support(y_true=labels, y_pred=preds, average=None, labels=label_list)
        #print(label_list)
        logger.info(list(set(labels)))
        logger.info(len(labels))
        logger.info(list(set(preds)))
        logger.info(len(preds))

        prf_per_class = classification_report(y_true=labels, y_pred=preds, digits=4, labels=label_list,output_dict=True)
        prf_per_class = format_classifaction_report(prf_per_class)
        # to calculate per class accuracy
        cm = confusion_matrix(y_true=labels, y_pred=preds)
        #cm_str = np.array2string(cm, separator='\t')
        cm_str = format_conf_mat(labels, preds)
        per_class_acc=cm.diagonal() / cm.sum(axis=1)
        #per_class_acc = per_class_acc.tolist()
        per_class_acc_str =""
        for item in per_class_acc.tolist():
            per_class_acc_str=per_class_acc_str+str(item)+"\t"


        return {
            "acc": acc,
            "prec": prf[0],
            "rec": prf[1],
            "f1": prf[2],
            "perclass": prf_per_class,
            "confusion_matrix": cm_str,
            "perclassAcc": per_class_acc_str.strip(),
        }


    def glue_compute_metrics(task_name, preds, labels,label_list):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "multitask":
            #return {"acc-multitask: ": simple_accuracy(preds, labels)}
            return {"acc-multitask": acc_and_p_r_f_per_class(preds, labels, label_list)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
