import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy

class EncoderSeparatorDecoder(nn.Module):
    """
    A standard Speech Separation architecture.
    """

    def __init__(self, encoder, decoder, separator):
        super(EncoderSeparatorDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.separator = separator
    
    def forward(self, mix_audio, window):
        """
        input: 
            mix_audio: Batch, nmic, T
        output:
            separated_audio: Batch, nspk, T
        """
        return self.decode(self.separate(self.encode(mix_audio, window=window)), window=window)

    def encode(self, mix_audio, window):
        return self.encoder(mix_audio, window)
    
    def separate(self, enc_output):
        return self.separator(enc_output)
    
    def decode(self, sep_output, window):
        return self.decoder(sep_output, window)


# Encoder
class Encoder(nn.Module):
    "STFT -> conv2d -> gLN"

    def __init__(self, n_fft=256, embed_dim=32, nmic=6):
        """
        The sampling rate is 8 kHz. The STFT window size is 32 ms and hop size 8 ms.
        n_fft = 32*8 = 256, hop_length = 8*8 = 64
        """
        super(Encoder, self).__init__()
        # self.stft = torch.stft
        self.n_fft = n_fft
        # self.window = torch.hann_window(n_fft)

        self.conv2d = nn.Conv2d(in_channels=2*nmic, out_channels=embed_dim, kernel_size=3, padding=1)
        self.gLN = nn.GroupNorm(1, embed_dim, eps=1e-8)
    
    def forward(self, mix_audio, window):
        """
        mix_audio: batch, nmic, T
        """
        batch, nmic, T = mix_audio.shape
        output = mix_audio.view(-1, T) # batch*nmic, T
        output = torch.stft(output, self.n_fft,
                            window=window,
                            return_complex=False) # batch*nmic, n_fft/2 + 1, T', 2
        # 2 ways to do D dimentional embedding
        # output = output.permute(0, 3, 2, 1).contiguous() # batch*nmic, 2, T', n_fft/2 + 1
        output = output.permute(0, 3, 2, 1).contiguous().view(batch, 2*nmic, -1, self.n_fft//2 + 1) # batch, 2*nmic, T', n_fft/2 + 1
        output = self.conv2d(output) # batch, embed_dim, T', n_fft/2+1
        output = self.gLN(output)

        return output



# Decoder
class Decoder(nn.Module):
    "Deconv2d -> linear -> iSTFT"

    def __init__(self, n_fft=256, embed_dim=32, nspk=2):
        super(Decoder, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=2*nspk, kernel_size=3, padding=1)
        # 论文里没仔细说，后续可以多加几层看看有没有效果
        self.linear = nn.Linear(in_features=2*nspk, out_features=2*nspk)

        self.n_fft = n_fft
        # self.window = torch.hann_window(n_fft)
        # self.istft = torch.istft

    def forward(self, x, window):
        "x: batch, D, T, F"

        output = self.deconv2d(x) # batch, 2*nspk, T, F

        batch, _, T, F = output.shape

        output = output.permute(0,2,3,1).contiguous().view(batch*T*F, -1)
        output = self.linear(output)

        output = output.view(batch, T, F, -1).transpose(1,2).contiguous() # batch, F, T, 2*nspk
        output_split = torch.split(output, 2, dim=3) # nspk*(batch, F, T, 2)

        separation_res = []
        for utterance in output_split:
            audio = torch.istft(utterance, self.n_fft,
                                window=window,
                                return_complex=False) # batch, T'
            separation_res.append(audio)
        
        res = torch.stack(separation_res, 1) # batch, nspk, T'
        return res



# Separator
def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Separator(nn.Module):
    "Separator is a stack of N layers"

    def __init__(self, layer, N=6):
        super(Separator, self).__init__()
        self.layers = clones(layer, N)
    
    def forward(self, x):
        "Pass the input through each layer in turn"
        for layer in self.layers:
            x = layer(x)
        return x

# TF-GridNet block
class SeparatorLayer(nn.Module):
    "Separator is made up of spectral module, temporal module and self-attention module(defined below)"

    def __init__(self, RNN_module_s, RNN_module_t, self_attn):
        super(SeparatorLayer, self).__init__()
        self.spectral = RNN_module_s
        self.temporal = RNN_module_t
        self.self_attn = self_attn

    def forward(self, x):
        """
        x: batch, D, T, F 
        """
        output = self.spectral(x) # batch, D, T, F

        output = output.transpose(2,3).contiguous()
        output = self.temporal(output) # batch, D, F, T

        output = output.transpose(2,3).contiguous()
        output = self.self_attn(output) # batch, D, T, F

        return output
    
class RNNModule(nn.Module):
    "Unfold -> LN -> BLSTM -> Deconv1D -> residual"

    def __init__(self, hidden_size=256, kernel_size=8, stride=1, embed_dim=32, dropout=0, bidirectional=True):
        super(RNNModule, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        self.layernorm = nn.LayerNorm(embed_dim*kernel_size)
        self.rnn = nn.LSTM(embed_dim*kernel_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional) # N,L,F -> N,L,2H
        self.deconv1d = nn.ConvTranspose1d(2*hidden_size, embed_dim, kernel_size, stride=stride) 
    

    def forward(self, x):
        "x: batch, D, dim1, dim2"
        batch, embed_dim, dim1, dim2 = x.shape

        output = x.unfold(-1, self.kernel_size, self.stride) # batch, D, dim1, dim2/stride, kernel_size

        output = output.permute(0,2,3,4,1).contiguous().view(batch, dim1, -1, self.kernel_size*embed_dim)
        output = self.layernorm(output) # batch, dim1, dim2/stride, D*kernel_size

        output = output.view(batch*dim1, -1, self.kernel_size*embed_dim)
        output, _ = self.rnn(output) # batch*dim1, dim2/stride, 2*hidden_size

        output = output.contiguous().transpose(1,2).contiguous() # batch*dim1, 2*hidden_size, dim2/stride
        output = self.deconv1d(output) # batch*dim1, D, dim2
        # output = output.transpose(1,2).contiguous() # batch*dim1, dim2, D

        output = output.view(batch, dim1, embed_dim, dim2).permute(0,2,1,3).contiguous()
        output = output + x

        return output

# Pre-processing method to generate qkv in Attention module
class Generator(nn.Module):
    "1X1Conv2d -> PReLU -> cfLN"

    def __init__(self, input_dim=32, output_dim=4, F=129):
        super(Generator, self).__init__()
        self.conv2d = nn.Conv2d(input_dim, output_dim, 1)
        self.prelu = nn.PReLU()
        self.cfLN = nn.LayerNorm([output_dim, F])
    
    def forward(self, x):
        "x: batch, embed_dim, T, F"
        output = self.conv2d(x) # batch, output_dim, T, F
        output = self.prelu(output) # batch, output_dim, T, F

        output = output.transpose(1,2).contiguous() # batch, T, output_dim, F
        output = self.cfLN(output)
        output = output.transpose(1,2).contiguous()

        return output


# Dot-product Attention
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    Q: batch, T, FxE
    K: batch, T, FxE
    V: batch, T, FxD/L
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# Multi-head Attention (full-band self-attention module)
class MultiHeadAttention(nn.Module):
    def __init__(self, h=4, d_model=32, d_q=4, F=129, dropout=0.1):
        "Take in model size and number of heads"
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0
        self.d_q = d_q
        self.d_k = d_q
        self.d_v = d_model // h

        self.q_generators = clones(Generator(d_model, self.d_q, F), h)
        self.k_generators = clones(Generator(d_model, self.d_k, F), h)
        self.v_generators = clones(Generator(d_model, self.d_v, F), h)
        self.out_generator = Generator(d_model, d_model, F)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        "x: batch, D, T, F"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch, d_model, T, F = x.shape

        # 1) do all the linear projections in batch from d_model => h x d_k
        attn_output = []
        for q_generate, k_generate, v_generate in zip(self.q_generators, self.k_generators, self.v_generators):
            q = q_generate(x) # batch, E, T, F
            q = q.permute(0,2,3,1).contiguous().view(batch, T, F*self.d_q) # batch, T, FxE
            k = k_generate(x)
            k = k.permute(0,2,3,1).contiguous().view(batch, T, F*self.d_k)
            v = v_generate(x)# batch, D/L, T, F
            v = v.permute(0,2,3,1).contiguous().view(batch, T, F*self.d_v) # batch, T, FxD/L

            single_output, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout) # batch, T, FxD/L

            attn_output.append(single_output)
        
        output = torch.cat(attn_output, 2) # batch, T, FxD
        output = output.view(batch, T, F, d_model).permute(0,3,1,2).contiguous() # batch, D, T, F
        output = self.out_generator(output)
        output = output + x

        del x, q, k, v
        return output



# Full Model
def make_TF_GridNet(nmic=6, nspk=2, n_fft=256, D=32, B=6, I=8, J=1, H=256, E=4, L=4):
    "Helper: Construct a model from hyperparameters"
    F = n_fft // 2 + 1

    c = copy.deepcopy
    RNN_module = RNNModule(hidden_size=H, kernel_size=I, stride=J, embed_dim=D)
    self_attn = MultiHeadAttention(h=L, d_model=D, d_q=E, F=F)
    model = EncoderSeparatorDecoder(
        encoder=Encoder(n_fft=n_fft, embed_dim=D, nmic=nmic), 
        decoder=Decoder(n_fft=n_fft, embed_dim=D, nspk=nspk),
        separator=Separator(SeparatorLayer(c(RNN_module), c(RNN_module), c(self_attn)), N=B)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model



if __name__ == "__main__":
    # test Encoder
    # encoder = Encoder()
    # x = torch.randn(3,6,64000)
    # y = encoder(x)
    # print(y.shape)

    # test Decoder
    # decoder = Decoder()
    # x = torch.randn(3, 32, 1001, 129)
    # y = decoder(x)
    # print(y.shape)

    # test RNNModule
    # rnn_module = RNNModule(hidden_size=256)
    # x = torch.rand(3, 32, 1001, 129)
    # y = rnn_module(x)
    # print(y.shape)

    # test MultiheadAttention

    # 1.test Generator
    # generator = Generator(F=127)
    # x = torch.randn(3, 32, 1001, 129)
    # y = generator(x)
    # print(y.shape)

    # 2.test attention module
    # attn = MultiHeadAttention(F=127)
    # x = torch.randn(3, 32, 1001, 129)
    # y = attn(x)
    # print(y.shape)

    # test full model
    test_model = make_TF_GridNet()
    test_model.eval()
    mix_audio = torch.randn(3,6,64000)
    separated_audio = test_model(mix_audio)
    print(separated_audio.shape)





