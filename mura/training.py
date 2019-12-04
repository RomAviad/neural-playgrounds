import copy
import time
import torch
from torch.autograd import Variable

from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils import model_zoo
from mura.paper import MuraDenseNet
import numpy as np
from visdom import Visdom

IS_CUDA = False


def get_pretrained_with_imagenet():
    state_dict = model_zoo.load_url("https://download.pytorch.org/models/densenet169-b2777c0a.pth")
    # Delete last layer weights as we use a different setup on that
    del state_dict["classifier.bias"]
    del state_dict["classifier.weight"]
    model = MuraDenseNet()
    model.load_state_dict(state_dict, strict=False)
    return model


class MuraBCELoss(Module):
    def __init__(self, W1, W0, batch_size):
        super().__init__()
        self.W0 = W0
        self.W1 = W1

    def forward(self, inputs, targets, phase, study_types):
        Wt1 = torch.Tensor([W1[study_type][phase] for study_type in study_types])
        Wt0 = torch.Tensor([W0[study_type][phase] for study_type in study_types])
        inputs_t = inputs.view_as(targets)
        loss = torch.neg(Wt1 * targets * inputs_t.log() + Wt0 * (1 - targets) * (1 - inputs_t).log())
        return loss


def train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs, label_key="label"):
    vis = Visdom()
    line_windows = {"train_loss": None, "valid_loss": None, "train_acc": None, "valid_acc": None}
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x: [] for x in data_cat} # for storing costs per epoch
    accs = {x: [] for x in data_cat} # for storing accuracies per epoch
    print("Train batches:", len(dataloaders["train"]))
    print("Valid batches:", len(dataloaders["valid"]), "\n")
    for epoch in range(num_epochs):
        # confusion_matrix = {x: meter.ConfusionMeter(2, normalized=True)
        #                     for x in data_cat}
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end="\r")
                inputs = data["images"]
                labels = data[label_key].type(torch.FloatTensor)
                body_parts = data["type"]
                # wrap them in Variable
                inputs = Variable(inputs) #.cuda())
                labels = Variable(labels) #.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                # outputs = torch.mean(outputs)
                # loss = criterion(outputs, labels, phase, body_parts).mean()
                loss = criterion(outputs, labels).mean()
                running_loss += loss.data.item()#[0]
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.7).type(torch.FloatTensor) #cuda.FloatTensor)
                correct_preds = torch.Tensor([all(preds[i] == labels[i]) for i in range(len(preds))]) #BODY_PARTS
                running_corrects += torch.sum(correct_preds)#preds == labels.view(-1, 1))

                if i % 100 == 99:
                    print("Running Corrects: {} / {}; Running Loss {}".format(running_corrects, (i + 1) * len(labels),
                                                                              running_loss))  # 8 being batch size

                # confusion_matrix[phase].add(preds, labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            loss_window = "{}_loss".format(phase)
            acc_window = "{}_acc".format(phase)
            if line_windows[loss_window] is None:
                line_windows[loss_window] = vis.line(X=np.array([epoch, epoch]),
                                                     Y=np.array([epoch_loss.item(), epoch_loss.item()]), env="main",
                                                     name=loss_window)
            else:
                vis.line(X=np.array([epoch]), Y=np.array([epoch_loss.item()]), env="main",
                         win=line_windows[loss_window], update="append", name=loss_window)
            if line_windows[acc_window] is None:
                line_windows[acc_window] = vis.line(X=np.array([epoch, epoch]), Y=np.array([epoch_acc, epoch_acc]),
                                                    env="main", name=acc_window)
            else:
                vis.line(X=np.array([epoch]), Y=np.array([epoch_loss]), env="main",
                         win=line_windows[acc_window], update="append", name=acc_window)

            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print("{} Loss: {:.10f} Acc: {:.10f}".format(
                phase, epoch_loss, epoch_acc))
            # print("Confusion Meter:\n", confusion_matrix[phase].value())
            # deep copy the model
            if phase == "valid":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "checkpoint_{}epochs.pt".format(epoch+1))
        time_elapsed = time.time() - since
        print("Time elapsed: {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best valid Acc: {:4f}".format(best_acc))
    # plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    from mura.dataloaders import get_dataloaders, get_study_level_data, get_image_level_data
    from mura.utils import numpy_float_to_variable_tensor_float, get_count, STUDY_TYPES
    from mura import body_part_net

    model = body_part_net.load_pretrained_with_imagenet()
    # model = get_pretrained_with_imagenet()
    # if os.environ.get("CUDA", "0") != "0":
    # model.cuda()

    STUDY_TYPES = ["XR_ELBOW"]
    data_cat = ["train", "valid"]
    data = {study_type: {"train": None, "valid": None} for study_type in STUDY_TYPES}
    W1 = {}
    W0 = {}
    sizes = {x: 0 for x in data_cat}
    for study_type, study_data in data.items():
        category_data = get_image_level_data(study_type)  #get_study_level_data(body_part) #
        for dc in data_cat:
            if study_data[dc] is None:
                study_data[dc] = category_data[dc]
            else:
                study_data[dc] = study_data[dc].append(category_data[dc])
        tai = {x: get_count(study_data[x], "positive") for x in data_cat}
        tni = {x: get_count(study_data[x], "negative") for x in data_cat}
        Wt1 = {x: numpy_float_to_variable_tensor_float(tni[x] / (tni[x] + tai[x])) for x in data_cat}
        Wt0 = {x: numpy_float_to_variable_tensor_float(tai[x] / (tni[x] + tai[x])) for x in data_cat}
        W1[study_type] = Wt1
        W0[study_type] = Wt0
        for dc in data_cat:
            sizes[dc] += len(study_data[dc])
    batch_size = 8
    # flatten the data
    data_flat = {"train": None, "valid": None}
    for study_type, study_data in data.items():
        for dc in data_cat:
            if data_flat[dc] is None:
                data_flat[dc] = study_data[dc]
            else:
                data_flat[dc] = data_flat[dc].append(study_data[dc])
    dataloaders = get_dataloaders(data_flat, batch_size=batch_size)

    # criterion = MuraBCELoss(W1, W0, batch_size)
    criterion = torch.nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1, verbose=True)
    num_epochs = 200
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes=sizes, num_epochs=num_epochs,
                        label_key="type_oh")
    torch.save(model.state_dict(), "body_part_train_{}epochs.pt".format(num_epochs))
    # torch.save(model.state_dict(), "mura_train_{}epochs.pt".format(num_epochs))
