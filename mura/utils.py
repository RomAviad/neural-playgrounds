import torch
from torch.autograd import Variable

STUDY_TYPES = [
    "XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"
]

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


def body_part_to_one_hot(body_part):
    global STUDY_TYPES
    if body_part not in STUDY_TYPES:
        raise Exception("Invalid body part")
    result = torch.zeros(7)
    result[STUDY_TYPES.index(body_part)] = 1
    return result

if __name__ == '__main__':
    from pprint import pprint
    from sklearn.metrics.classification import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import numpy as np
    import matplotlib.pyplot as plt
    total_corrects = 0
    total_images = 0
    results = {}
    ground_truth = []
    flattened_preds = []
    for st in STUDY_TYPES:
        res = torch.load("/Users/aviadrom/Dev/neural_networks/mura_results/{}_results.pt".format(st))
        valid = res["valid"]
        total_images += len(valid)
        valid_norm = [torch.Tensor(v).argmax().item() for v in valid]
        gt = [l.argmax().item() for l in res["valid_labels"]]
        assert (len(set(gt)) == 1)
        st_norm = st.replace("XR_", "")
        print(f"{st_norm} Classifications")
        d = {STUDY_TYPES[i].replace("XR_", ""): valid_norm.count(i) for i in set(valid_norm)}
        pprint(d)
        print(f"{st_norm} Accuracy: {d[st_norm] / sum(d.values())}")
        total_corrects += d[st_norm]
        results[st_norm] = d
        for _ in range(len(valid_norm)):
            ground_truth.append(st_norm)
        for study_type, count in d.items():
            for _ in range(count):
                flattened_preds.append(study_type)
        # wrong = [v for i, v in enumerate(valid) if STUDY_TYPES.index(st) != valid_norm[i]]
    #     pprint(wrong)
    print()
    print(f"TOTAL IMAGES: {total_images}")
    print(f"TOTAL CORRECT CLASSIFICATIONS: {total_corrects} ({total_corrects / total_images})")

    cm = confusion_matrix(ground_truth, flattened_preds, [st.replace("XR_", "") for st in STUDY_TYPES])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = [st.replace("XR_", "") for st in STUDY_TYPES]
    # classes = classes[unique_labels(ground_truth, flattened_preds)]

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="Body Part Classifier Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = ".3f"#'d'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    pprint(cm)