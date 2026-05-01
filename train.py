def Train(DataLoader, Model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CNN_Emoji_model.parameters(), lr=0.001, momentum=0.9)
    
    CNN_Emoji_model.to(device)
    for epoch in range(30):  # loop over the dataset multiple times
    
        Avg_loss = 0
        Accuracy = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs_image, labels_class, text_list = data
            inputs_image, labels_class = inputs_image.to(device), labels_class.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = CNN_Emoji_model(inputs_image)
            loss = criterion(outputs, labels_class)
            loss.backward()
            optimizer.step()
            Avg_loss += loss
            accuracy = (outputs.argmax(dim=1)==labels_class).to(torch.float).detach().cpu().mean() * 100
            Accuracy += accuracy
        print(f'[Epoch: {epoch + 1}] loss: {Avg_loss/i:.3f}, \tAccuracy: {Accuracy/i:.3f}')
    
    print('Finished Training')

Train(trainloader, CNN_Emoji_model, device)
