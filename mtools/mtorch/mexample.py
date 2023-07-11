import logging
import torch
from mtool.meval import get_multi_f1, get_multi_precision, get_multi_recall

from mtool.mtorch.mtrainer import TBase


class Trainer(TBase):
    def forward(self, data):
        images, labels = data
        images = images.float()
        labels = labels.long()
        if self.USE_CUDA and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        predict = self.model(images)
        loss = self.criterion(predict, labels)
        return loss

    def predict(self, data):
        images, labels = data
        images = images.float()
        labels = labels.float()
        if self.USE_CUDA and torch.cuda.is_available():
            images = images.cuda()

        with torch.no_grad():
            predict = self.model(images)

        if self.USE_CUDA and torch.cuda.is_available():
            predict = predict.cpu()

        predict = list(map(int, list(predict.numpy().argmax(axis=1))))
        labels = list(map(int, list(labels.numpy())))
        return predict, labels

    def metrics(self, y_pred, y_true):
        logging.info("\nVal num:{} Precision:{} Recall:{} F1:{}\npredict:{}\nlabel:{}".format(
            len(y_true),
            get_multi_precision(y_pred, y_true),
            get_multi_recall(y_pred, y_true),
            get_multi_f1(y_pred, y_true),
            y_pred,
            y_true
        ))
        self.monitor = get_multi_precision(y_pred, y_true)
