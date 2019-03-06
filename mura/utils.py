import torch
from torch.autograd import Variable


def numpy_float_to_variable_tensor_float(x):
    '''convert numpy float to Variable tensor float'''
    return Variable(torch.FloatTensor([x]), requires_grad=False)


def get_count(df, cat):
    '''
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    '''
    return len(df[df['Path'].str.contains(cat)]) #['Count'].sum()


if __name__ == 'main':
    pass