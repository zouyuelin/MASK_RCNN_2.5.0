import numpy as np

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        print(mask)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        # print(hist)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, lp, lt):

        self.hist += self._fast_hist(lp.flatten(), lt.flatten())

        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        ap = np.sum(np.diag(self.hist))/np.sum(self.hist)
        dice = 2 * np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0))
        dice = np.mean(dice)
        return iou, miou, ap, dice