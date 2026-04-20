import torch
import torch.nn as nn
import torch.fft

class FrequencySelfAttention(nn.Module):
    def __init__(self, in_channels, k=16):
        super().__init__()
        self.k = k 
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        

        x_freq = torch.fft.rfft2(x, norm='ortho')

        k_h = min(self.k, x_freq.shape[2])
        k_w = min(self.k, x_freq.shape[3])
        low_freq = x_freq[:, :, :k_h, :k_w]
        
        low_freq_real = torch.view_as_real(low_freq).mean(dim=-1)
        
        proj_query = self.query_conv(low_freq_real).view(B, -1, k_h * k_w).permute(0, 2, 1)
        proj_key = self.key_conv(low_freq_real).view(B, -1, k_h * k_w)
        proj_value = self.value_conv(low_freq_real).view(B, -1, k_h * k_w)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        out = out.view(B, C, k_h, k_w)
        out_complex = torch.complex(out, torch.zeros_like(out))
        
        padded_freq = torch.zeros_like(x_freq)
        padded_freq[:, :, :k_h, :k_w] = out_complex
        
        out_spatial = torch.fft.irfft2(padded_freq, s=(H, W), norm='ortho')
        
        return x + self.gamma * out_spatial