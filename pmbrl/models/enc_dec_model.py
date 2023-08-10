import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (10,)
    """
    def __init__(self,device):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 64*4
        self.fc1 = nn.Linear(64*4, 10)
        self.to(device)

    def forward(self, vec, img):
        hidden = F.relu(self.cv1(img))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        embedded_obs = F.relu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        embedded_obs = self.fc1(embedded_obs)
        embedded_obs = torch.cat([vec, embedded_obs], dim=1)
        return embedded_obs

class Decoder(nn.Module):
    """
    p(o_t | s_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """
    def __init__(self, vec_dim, embedded_img_dim, device):
        super(Decoder, self).__init__()
        self.vec_dim = vec_dim
        self.embedded_img_dim = embedded_img_dim
        self.fc = nn.Linear(embedded_img_dim, 256)
        self.dc1 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2)
        self.to(device)

    def forward(self, embedded_obs):
        # TODO 独立的img emb可以切换成混合的
        vec = embedded_obs[:, :self.vec_dim]
        embedded_img = embedded_obs[:, self.vec_dim:]
        hidden = self.fc(embedded_img, dim=1)
        hidden = hidden.view(hidden.size(0), 256, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return vec, obs