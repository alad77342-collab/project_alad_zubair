class CNN_Emoji(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 24, 12)
        self.conv3 = nn.Conv2d(24, 48, 48)
        self.fc1 = nn.Linear(48*12*12, 1816)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

CNN_Emoji_model = CNN_Emoji()  
