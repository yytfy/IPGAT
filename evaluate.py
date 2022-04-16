import time
import torch
import numpy as np
import datafunctions
import torch.utils.data as Data
from sklearn import metrics


def evaluate_auroc(prediction, targets):
    print('evaluating auroc----------')
    start = time.perf_counter()

    prediction = prediction.cpu()

    auroc = metrics.roc_auc_score(targets, prediction, multi_class='ovo')

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return auroc


def evaluate_aupr(prediction, targets):
    print('evaluating apur------------')
    start = time.perf_counter()

    prediction = prediction.cpu()

    precision, recall, thresholds = metrics.precision_recall_curve(targets, prediction)
    aupr = metrics.auc(recall, precision)

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return aupr


def evaluate_balanced_accuracy(prediction, targets):
    print('evaluate balanced accuracy------------')
    start = time.perf_counter()

    prediction = [0 if p < 0.5 else 1 for p in prediction]
    balanced_accuracy = metrics.balanced_accuracy_score(targets, prediction)

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return balanced_accuracy


def evaluate_top_k_accuracy(prediction, targets, k=2):
    print('evaluate top %d accuracy-----------')
    start = time.perf_counter()

    prediction = [0 if p < 0.5 else 1 for p in prediction]
    top_k_accuracy = metrics.top_k_accuracy_score(targets, prediction)

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return top_k_accuracy


def evaluate_accuracy(prediction, targets):
    print('evaluateing accuracy-------------')
    start = time.perf_counter()

    prediction = [0 if p < 0.5 else 1 for p in prediction]
    accuracy = metrics.accuracy_score(targets, prediction)

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return accuracy


def evaluate_precision(prediction, targets, m, l=250):
    print('evaluating precision-------')
    start = time.perf_counter()

    prediction = [0 if p < 0.5 else 1 for p in prediction]
    precision = metrics.precision_score(targets, prediction)
    # prediction, indices = torch.sort(prediction, 0, descending=True)
    #
    # sum = 0
    # precision = []
    #
    # for i in range(l):
    #     t = prediction[i][0]
    #     if m[nodes_a[indices[i][0]], nodes_b[indices[i][0]]] == 1:
    #         sum = sum + 1
    #     if i % 50 == 49:
    #         precision.append(sum/(50*(int(i/50)+1)))

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return precision


def evaluate_average_precision(prediction, targets):
    print('evaluate average precision------------')
    start = time.perf_counter()

    prediction = [0 if p < 0.5 else 1 for p in prediction]
    average_precision = metrics.average_precision_score(targets, prediction)

    end = time.perf_counter()
    print('evaluate time %f' % (end - start))
    return average_precision


if __name__ == '__main__':
    modelType = 'IPGAT'
    trainVersion = '5.0.0'
    validationVersion = '5.1.0'
    testVersion = '5.1.0'
    model_path = '/data/DDI_AMF/results/models/IPGAT/holdout/8.pkl'
    BATCH_SIZE = 5000

    print('preparing data.')
    m_train, m_test, drugs = datafunctions.create_train_test_split_version(trainVersion, testVersion)
    # m_train, m_test, drugs = datafunctions.create_train_test_single_version(trainVersion)

    model = torch.load(model_path).cuda()



    print('creating test data.')
    test_list = datafunctions.get_test_list(m_train)
    test_nodes_a, test_nodes_b, test_targets = datafunctions.list2inputs(test_list, m_test)

    print('test total: %d' % len(test_targets))

    with torch.no_grad():
        # if gpu memory is enough
        if modelType == 'FAN':
            adj = torch.tensor(m_test).cuda()
            prediction = model(test_nodes_a, test_nodes_b, adj)
        elif modelType == 'IPGAT':
            adj = torch.tensor(m_test).cuda()
            prediction = model(test_nodes_a, test_nodes_b, adj)
        elif modelType == 'AMF':
            prediction = model(test_nodes_a, test_nodes_b)

        #if gpu memory is not enough
        # test_dataset = Data.TensorDataset(test_nodes_a, test_nodes_b)
        # test_loader = Data.DataLoader(
        #     dataset=test_dataset,
        #     batch_size=BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=8,
        # )
        # if modelType == 'FAN':
        #     adj = torch.tensor(m_test).cuda()
        #     prediction = torch.cat([model(batch_a, batch_b, adj) for step, (batch_a, batch_b) in enumerate(test_loader)], dim=0)
        # elif modelType == 'IPGAT':
        #     adj = torch.tensor(m_test).cuda()
        #     prediction = torch.cat([model(batch_a, batch_b, adj) for step, (batch_a, batch_b) in enumerate(test_loader)], dim=0)
        # elif modelType == 'AMF':
        #     prediction = torch.cat([model(batch_a, batch_b) for step, (batch_a, batch_b) in enumerate(test_loader)], dim=0)

    auroc = evaluate_auroc(prediction, test_targets)
    print('evaluate auroc: %f' % auroc)

    aupr = evaluate_aupr(prediction, test_targets)
    print('evaluate aupr: %f' % aupr)

    # precision = evaluate_precision(prediction, test_targets, m_test)
    # print('evaluate precision: %f ' % precision)
    #
    # average_precision = evaluate_average_precision(prediction, test_targets)
    # print('evaluate average precision: %f' % average_precision)
    #
    # accuracy = evaluate_accuracy(prediction, test_targets)
    # print('evaluate accuracy: %f' % accuracy)
    #
    # balanced_accuracy = evaluate_balanced_accuracy(prediction, test_targets)
    # print('evaluate balanced accuracy: %f' % balanced_accuracy)

    # top_k_accuracy = evaluate_top_k_accuracy(prediction, test_targets)
    # print('evaluate top k accuracy: %f' % top_k_accuracy)




