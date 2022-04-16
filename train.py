import datafunctions
import evaluate
import torch
import models
import torch.nn.functional as F
import torch.utils.data as Data
import os
import numpy as np
import pickle


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    modelSavePath = "results/models"
    modelType = 'IPGAT'
    trainVersion = '5.0.0'
    validationVersion = '5.1.0'
    testVersion = '5.1.0'
    experiment_type = 'retrospective'
    epoch = 9
    BATCH_SIZE = 2000
    neg_per_pos = 1.0   # ratio of neg and pos data in training set


    print('preparing data.')
    if experiment_type == 'retrospective':
        m_train, m_test, drugs = datafunctions.create_train_test_split_version(trainVersion, testVersion)
    else:
        m_train, m_test, test_tuples, drugs = datafunctions.create_train_test_single_version(trainVersion, testVersion)

    if modelType == 'IPGAT':
        model = models.IPGAT(len(drugs)).cuda()
        adj = torch.Tensor(m_test).cuda()
    elif modelType == 'AMF':
        model = models.AMF(len(drugs)).cuda()
    elif modelType == 'Con_LSTM':
        model = models.Con_LSTM(len(drugs)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    modelSavePath = modelSavePath + '/' + modelType + '/' + experiment_type
    if not os.path.exists(modelSavePath):
        os.mkdir(modelSavePath)

    train_tuples = [i for (i, v) in np.ndenumerate(m_train) if i[0] > i[1]]
    train_pos_list, train_neg_list = datafunctions.matrix2pos_neg_list(m_train, train_tuples)

    for i in range(epoch):
        nodes_a, nodes_b, targets = datafunctions.get_train_sample(train_pos_list, train_neg_list, m_train, neg_to_pos_ratio=neg_per_pos)
        dataset = Data.TensorDataset(nodes_a, nodes_b, targets)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=3,
        )
        for step, (batch_a, batch_b, batch_targets) in enumerate(loader):
            batch_targets = batch_targets.cuda()
            optimizer.zero_grad()
            if modelType == 'IPGAT':
                prediction = model(batch_a, batch_b, adj)
            elif modelType == 'AMF':
                prediction = model(batch_a, batch_b)
            loss = F.binary_cross_entropy(prediction, batch_targets)
            loss.backward()
            optimizer.step()

            print(('Loss %.02f at epoch %d step %d' % (loss, i, step)))

        path = modelSavePath + '/' + str(i) + '.pkl'
        torch.save(model, path)

    print('creating test data.')
    test_list = datafunctions.get_test_list(m_train)
    print(len(test_list))
    test_nodes_a, test_nodes_b, test_targets = datafunctions.list2inputs(test_list, m_test)
    # test_nodes_a, test_nodes_b, test_targets = datafunctions.list2inputs(test_tuples, m_test)
    print('test total: %d' % len(test_targets))

    with torch.no_grad():
        # if gpu memory is enough
        if modelType == 'IPGAT':
            adj = torch.tensor(m_test).cuda()
            prediction = model(test_nodes_a, test_nodes_b, adj)
        elif modelType == 'AMF':
            prediction = model(test_nodes_a, test_nodes_b)
        elif modelType == 'Con_LSTM':
            prediction = model(test_nodes_a, test_nodes_b)

        # if gpu memory is not enough
        # test_dataset = Data.TensorDataset(test_nodes_a, test_nodes_b)
        # test_loader = Data.DataLoader(
        #     dataset=test_dataset,
        #     batch_size=BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=2,
        # )
        # if modelType == 'FAN':
        #     adj = torch.tensor(m_test).cuda()
        #     prediction = torch.cat([model(batch_a, batch_b, adj) for step, (batch_a, batch_b) in enumerate(test_loader)], dim=0)
        # elif modelType == 'IPGAT':
        #     adj = torch.tensor(m_test).cuda()
        #     prediction = torch.cat([model(batch_a, batch_b, adj) for step, (batch_a, batch_b) in enumerate(test_loader)], dim=0)
        # elif modelType == 'AMF':
        #     prediction = torch.cat([model(batch_a, batch_b) for step, (batch_a, batch_b) in enumerate(test_loader)], dim=0)
    file_prediction = open('IPGAT_Prediction_retrospective.pkl', 'wb')
    file_target = open('IPGAT_Target_retrospective.pkl', 'wb')
    pickle.dump(prediction, file_prediction)
    pickle.dump(test_targets, file_target)
    file_prediction.close()
    file_target.close()

    auroc = evaluate.evaluate_auroc(prediction, test_targets)
    print('evaluate auroc: %f' % auroc)

    aupr = evaluate.evaluate_aupr(prediction, test_targets)
    print('evaluate aupr: %f' % aupr)


    print(prediction.size())
    # plot
    s, indices = torch.sort(prediction, dim=0, descending=True)

    for i in range(100):
        print(s[i])

    num = 0
    i = 0
    while(num < 10):
        if(test_targets[indices[i]] == 1):
            print("drug1: %d---%f---drug2: %d" %(test_nodes_a[indices[i]], s[i], test_nodes_b[indices[i]]))
            num += 1
        i += 1








    # precision = evaluate.evaluate_precision(prediction, test_targets, m_test)
    # print('evaluate precision: %f ' % precision)
    #
    # average_precision = evaluate.evaluate_average_precision(prediction, test_targets)
    # print('evaluate average precision: %f' % average_precision)
    #
    # accuracy = evaluate.evaluate_accuracy(prediction, test_targets)
    # print('evaluate accuracy: %f' % accuracy)
    #
    # balanced_accuracy = evaluate.evaluate_balanced_accuracy(prediction, test_targets)
    # print('evaluate balanced accuracy: %f' % balanced_accuracy)


