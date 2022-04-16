from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
# y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
# fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)
#
# for i, value in enumerate(thersholds):
#     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# roc_auc = auc(fpr, tpr)
#
# plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
#
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

print("drug1: %d---%f---drug2: %d" %(1, 2, 3))