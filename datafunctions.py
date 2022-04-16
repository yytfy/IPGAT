import os
from XMLReader import drug_xml_reader
from DrugPreprocessor import drugs_preprocessor
from dataset_dates import d2d_versions_metadata
from utils import unpickle_object, pickle_object
import random
import numpy as np
import torch

relase_base_path = os.path.join('data', 'DrugBank_Versions')
version_base_file = os.path.join(relase_base_path, '%s', 'drugbank_all_full_database.xml.zip')
pickle_path = os.path.join('data', 'pickles','')
preproc_pickle_path = os.path.join('data', 'pickles', 'preproc')
pickle_extension = '.pickle'


def create_train_test_single_version(old_version, new_version, train_ratio=0.7, validation_ratio=0, ):
    """
        read old version and new version.
        """
    print('reading old verion file, version:' + old_version)
    drug_reader_old = read_version(old_version)
    drug_preprocessor_old = preprocess_version(drug_reader_old, old_version)
    print('num interactions in old version:', sum([len(drug_preprocessor_old.valid_drug_to_interactions[x]) for x in
                                                   drug_preprocessor_old.valid_drug_to_interactions]) / 2)
    print('num drugs old', len(drug_preprocessor_old.valid_drug_to_interactions))
    confirm_interactions(drug_preprocessor_old.valid_drugs_array, drug_preprocessor_old.valid_drug_to_interactions)

    print('reading new verion file, version:' + new_version)
    drug_reader_new = read_version(new_version)
    drug_preprocessor_new = preprocess_version(drug_reader_new, new_version)
    print('num interactions in old version:', sum([len(drug_preprocessor_new.valid_drug_to_interactions[x]) for x in
                                                   drug_preprocessor_new.valid_drug_to_interactions]) / 2)
    print('num drugs old', len(drug_preprocessor_new.valid_drug_to_interactions))
    confirm_interactions(drug_preprocessor_new.valid_drugs_array, drug_preprocessor_new.valid_drug_to_interactions)

    print('preprocessing two versions')
    drugs, interactions_old, interactions_new = drug_preprocessor_old.get_intersection(drug_preprocessor_new)
    print('common drugs len: ', len(drugs))

    print('creating train matrix')
    m_test = drugs_preprocessor.create_d2d_sparse_matrix(drugs, interactions_old)

    all_tuples = [(x, y) for x in range(len(drugs)) for y in range(len(drugs)) if x < y]

    train_tuples, validation_tuples, test_tuples = [], [], []

    for t in all_tuples:
        r = random.uniform(0, 1)
        if r < train_ratio:
            train_tuples.append(t)
        elif r < train_ratio + validation_ratio:
            validation_tuples.append(t)
        else:
            test_tuples.append(t)

    m_train = m_test.copy()
    for t in test_tuples:
        m_train[t[0], t[1]] = 0
        m_train[t[1], t[0]] = 0

    return m_train, m_test, test_tuples, drugs


def create_train_test_split_version(old_version, new_version):
    """
    read old version and new version.
    """
    print('reading old verion file, version:' + old_version)
    drug_reader_old = read_version(old_version)

    print("DB09282: " + drug_reader_old.drug_id_to_name["DB09282"])
    print("DB08868: " + drug_reader_old.drug_id_to_name["DB08868"])
    print("DB06761: " + drug_reader_old.drug_id_to_name["DB06761"])
    print("DB04897: " + drug_reader_old.drug_id_to_name["DB04897"])
    print("DB01248: " + drug_reader_old.drug_id_to_name["DB01248"])
    print("DB01158: " + drug_reader_old.drug_id_to_name["DB01158"])
    print("DB01119: " + drug_reader_old.drug_id_to_name["DB01119"])
    print("DB01073: " + drug_reader_old.drug_id_to_name["DB01073"])
    print("DB00888: " + drug_reader_old.drug_id_to_name["DB00888"])
    print("DB00851: " + drug_reader_old.drug_id_to_name["DB00851"])
    print("DB00606: " + drug_reader_old.drug_id_to_name["DB00606"])
    print("DB00305: " + drug_reader_old.drug_id_to_name["DB00305"])
    print("DB00112: " + drug_reader_old.drug_id_to_name["DB00112"])



    drug_preprocessor_old = preprocess_version(drug_reader_old, old_version)
    print('num interactions in old version:', sum([len(drug_preprocessor_old.valid_drug_to_interactions[x]) for x in drug_preprocessor_old.valid_drug_to_interactions]) / 2)
    print('num drugs old', len(drug_preprocessor_old.valid_drug_to_interactions))
    confirm_interactions(drug_preprocessor_old.valid_drugs_array, drug_preprocessor_old.valid_drug_to_interactions)

    print('reading new verion file, version:' + new_version)
    drug_reader_new = read_version(new_version)
    drug_preprocessor_new = preprocess_version(drug_reader_new, new_version)
    print('num interactions in old version:', sum([len(drug_preprocessor_new.valid_drug_to_interactions[x]) for x in drug_preprocessor_new.valid_drug_to_interactions]) / 2)
    print('num drugs old', len(drug_preprocessor_new.valid_drug_to_interactions))
    confirm_interactions(drug_preprocessor_new.valid_drugs_array, drug_preprocessor_new.valid_drug_to_interactions)

    print('preprocessing two versions')
    drugs, interactions_old, interactions_new = drug_preprocessor_old.get_intersection(drug_preprocessor_new)
    print('common drugs len: ', len(drugs))

    print('creating train matrix')
    m_train = drugs_preprocessor.create_d2d_sparse_matrix(drugs, interactions_old)
    print('creating test matrix')
    m_test = drugs_preprocessor.create_d2d_sparse_matrix(drugs, interactions_new)



    return m_train, m_test, drugs


def confirm_interactions(drugs, interactions):
    for d in drugs:
        assert d in interactions
        assert len(interactions[d]) > 0 and d not in interactions[d]


def read_version(version):
    """
    read XML file.
    """
    print('reading version:' + version)
    version = normalize_version(version)
    pickle_path = get_pickle_path(version)
    try:
        drug_reader = unpickle_object(pickle_path)
    except:
        print('failed to unpickle')
        version_path = get_version_path(version)
        drug_reader = drug_xml_reader(version_path)
        drug_reader.read_data_from_file()
        pickle_object(pickle_path, drug_reader)
    return drug_reader


def preprocess_version(drug_reader, version):
    """
    preprocess
    at least one interaction with a drug from the db and the interaction is symmetric.
    """
    print('preprocessing version...' + version)
    version = normalize_version(version)
    preprocessor_pickle_path = get_preproc_pickle_for_version(version)
    try:
        preprocessor = unpickle_object(preprocessor_pickle_path)
    except:
        print('failed to unpickle')
        print('num all drugs in reader:', len(drug_reader.all_drugs))
        preprocessor = drugs_preprocessor(drug_reader.drug_to_interactions, drug_reader.all_drugs)
        preprocessor.calc_valid_drugs_print_summary()
        preprocessor.create_valid_drug_interactions()
        pickle_object(preprocessor_pickle_path, preprocessor)
    return preprocessor


def matrix2pos_neg_list(m, tuples):
    pos_list = [t for t in tuples if m[t] == 1]
    neg_list = [t for t in tuples if m[t] == 0]
    print('pos_list total:', len(pos_list))
    print('neg_list total:', len(neg_list))
    return pos_list, neg_list


def get_train_sample(train_pos_list, train_neg_list, m, neg_to_pos_ratio=1.0):
    if neg_to_pos_ratio is None:
        train_list = train_neg_list + train_pos_list
    else:
        train_list = list(train_pos_list)
        if len(train_pos_list) * neg_to_pos_ratio < len(train_neg_list):
            train_list += random.sample(train_neg_list, int(len(train_pos_list) * neg_to_pos_ratio))
        else:
            print('not sampling due to increased number of positive samples')
            train_list += train_neg_list
    a = np.array([t[0] for t in train_list])
    b = np.array([t[1] for t in train_list])
    targets = np.array([[m[t[0], t[1]]] for t in train_list])
    a = torch.tensor(a, dtype=torch.long)
    b = torch.tensor(b, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.float)
    return a, b, targets


def list2inputs(list_sample, m):
    a = np.array([t[0] for t in list_sample])
    b = np.array([t[1] for t in list_sample])
    targets = np.array([[m[t[0], t[1]]] for t in list_sample])
    a = torch.LongTensor(a)
    b = torch.LongTensor(b)
    targets = torch.FloatTensor(targets)
    return a, b, targets


def get_test_list(m_train):
    """
    get total list except train intersections list
    """
    m = np.mat(np.ones(shape=m_train.shape))
    m = m - m_train
    test_list = [i for (i, v) in np.ndenumerate(m) if v == 1 and i[0] > i[1]]
    return test_list


def normalize_version(version):
    if version == '':
        version = d2d_versions_metadata[0]['VERSION']
        print('using latest version')
    elif version == '-1':
        version = d2d_versions_metadata[-1]['VERSION']
        print('using oldest version')
    print("Release number:", version)
    release_metada = get_version_metadata(version)
    print('relese metadata:', release_metada)
    return version


def get_version_metadata(version):
    for x in d2d_versions_metadata:
        if x['VERSION'] == version:
            return x
    assert False,'cant find version %s in metadata array' % version


def get_pickle_path(version):
    return pickle_path + version + pickle_extension


def get_version_path(version):
    return version_base_file % version


def get_preproc_pickle_for_version(version):
    return preproc_pickle_path + version + pickle_extension





