import matplotlib.pyplot as plt
import torch


def run_train(net, train_loader, criterion, optimizer, device):
    net.train()
    train_loss, correct, total = 0, 0, 0

    for _, samples in enumerate(train_loader):                          # len(train_loader)={2500}
        inputs, targets = samples['feat'], samples['label']
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                                           # reset all the parameter gradients

        # 1-1) forward pass: pass the input batch & call forward() method
        outputs = net(inputs)                                           # (batch_size,3,32,32) -> (batch_size,10)

        # 1-2) calculate loss
        loss = criterion(outputs, targets)

        # 1-3) back-propagation
        loss.backward()                                                 # the gradients have been calculated
        optimizer.step()                                                # execute gradients descent step

        # 1-4) update training loss
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        train_loss += loss.item()
        total += targets.size(0)
    return train_loss/len(train_loader), correct/total


def run_validate(net, test_loader, criterion, device):
    net.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, samples in enumerate(test_loader):
            inputs, targets = samples['feat'], samples['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)
    return test_loss/len(test_loader), correct/total


def run_test(net, test_loader, device):
    net.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for samples in test_loader:
            inputs, targets = samples['feat'], samples['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct, total


def summarize_learning(train_loss, val_loss, train_acc, val_acc, fn_fig, title):
    # loss
    plt.subplot(211)
    plt.plot(train_loss, color='blue', label='train')
    plt.plot(val_loss, color='orange', label='val')
    plt.title(title+': Loss')
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.set_xticklabels([])

    # accuracy
    plt.subplot(212)
    plt.plot(train_acc, color='blue', label='train')
    plt.plot(val_acc, color='orange', label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.set_xlabel('epoch')

    # save the plot to file
    plt.savefig(fn_fig)
    plt.close()
