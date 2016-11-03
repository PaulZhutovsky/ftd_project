from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import numpy as np
import inspect
from collections import OrderedDict


class Evaluater(object):

    def __init__(self, leave_one_out_case=False):
        self.loo = leave_one_out_case
        self.evaluations = self.set_evaluations()
        self.results = OrderedDict()
        self.evaluation_string = ''

    def evaluate(self, **kwargs):
        for eval_label, eval_fun in self.evaluations.iteritems():
            args_to_use = set(inspect.getargspec(eval_fun).args) & set(kwargs.keys())
            args_to_use = {key: kwargs[key] for key in args_to_use}
            self.results[eval_label] = eval_fun(**args_to_use)

    def evaluate_prediction(self, **kwargs):
        self.evaluate(**kwargs)
        return self.results.values()

    def evaluate_labels(self):
        return self.evaluations.keys()

    def print_evaluation(self):
        if not self.results:
            raise RuntimeError('evaluate has to be run first')

        if self.loo:
            self.evaluation_string = 'Accuracy: {accuracy:.2f}'.format(**self.results)
        else:
            self.evaluation_string = 'Accuracy: {accuracy:.2f}, AUC: {AUC:.2f}, F1-score: {F1:.2f}, Recall: ' \
                                     '{recall:.2f}, Precision: {precision:.2f}, Sensitivity: {sensitivity:.2f}, ' \
                                     'Specificity: {specificity:.2f}, ' \
                                     'PPV: {positive_predictive_value:.2f}'.format(**self.results)
        print self.evaluation_string

    def set_evaluations(self):
        if self.loo:
            evals = OrderedDict([('accuracy', accuracy_score),
                                 ('predictions_1st', self.__return_prediction_first),
                                 ('predictions_2nd', self.__return_prediction_second),
                                 ('true_1st', self.__return_true_first),
                                 ('ture_2nd', self.__return_true_second)])
        else:
            evals = OrderedDict([('accuracy', accuracy_score),
                                 ('balanced_accuracy', self.__balanced_accuracy),
                                 ('AUC', roc_auc_score),
                                 ('F1', f1_score),
                                 ('recall', recall_score),
                                 ('precision', precision_score),
                                 ('sensitivity', recall_score),           # recall is the same as sensitivity
                                 ('specificity', self.__specificity),
                                 ('positive_predictive_value', self.__ppv)])
        return evals

    @staticmethod
    def __balanced_accuracy(y_true, y_pred):
        return 0.5 * (Evaluater.__specificity(y_true, y_pred) + recall_score(y_true, y_pred))

    @staticmethod
    def __ppv(y_true, y_pred):
        # noinspection PyTypeChecker
        ppv = np.sum((y_true == 1) & (y_pred == 1)) / float(np.sum(y_pred == 1))
        if np.isnan(ppv):
            return 0
        return ppv

    @staticmethod
    def __specificity(y_true, y_pred):
        # noinspection PyTypeChecker
        return np.sum((y_true == 0) & (y_pred == 0)) / float(np.sum(y_true == 0))

    @staticmethod
    def __return_prediction_first(y_pred):
        return y_pred[0]

    @staticmethod
    def __return_prediction_second(y_pred):
        return y_pred[1]

    @staticmethod
    def __return_true_first(y_true):
        return y_true[1]

    @staticmethod
    def __return_true_second(y_true):
        return y_true[1]
