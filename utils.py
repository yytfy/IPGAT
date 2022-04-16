import pickle
import itertools



def pickle_object(path,object):
    print('pickling:',path)
    filehandler = open(path, "wb")
    pickle.dump(object, filehandler)
    filehandler.close()
    print('done pickling:', path)

def unpickle_object(path):
    print('trying to unpick:', path)
    file = open(path, 'rb')
    object_file = pickle.load(file)
    file.close()
    print('done unpickling:', path)
    return object_file

def array_to_dict(array):
    return dict([(v, i) for i, v in enumerate(array)])

def flatten_list(l):
    return list(itertools.chain.from_iterable(l))