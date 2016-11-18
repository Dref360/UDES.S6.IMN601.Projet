import pickle


def save_obj(obj, name):
    pickle.dump(obj, open("obj/{}.pkl".format(name), "wb"))


def load_obj(name):
    return pickle.load(open("obj/{}.pkl".format(name), "rb"))
