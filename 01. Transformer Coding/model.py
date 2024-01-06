import torch
import torch.nn as nn
import math

# 01. input embedding 부분 만들기
# 일단 512 dimension을 기본으로 취하고 있음
class InputEmbedding(nn.Module):
     
     def __init__(self, d_model: int, vocab_size: int): # dimension과 vocab사이즈 int로 설정
          super().__init__() # 파이토치의 nn.Module 사용하려면 super 해줘야 함.
          self.d_model = d_model
          self.vocab_size = vocab_size
          # 임베딩 코드 부분.
          self.embedding = nn.Embedding(vocab_size, d_model)

     def forward(self, x):
          # transformer 논문의 3.4Embedding and Softmax 부분 보면 dimension에 루트 씌운 값을 곱해(multiply)주는 것 알 수 있음.
          return self.embedding(x) * math.sqrt(self.d_model)
     
# 02. Positional Encoding 부분 만들기
     
class PositionalEncoding(nn.Module):

     def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
          # d_model : dimension, seq_len : maximum of the sentencem, dropout : 모델 학습시 overfit 방지
          super().__init__()
          self.d_model = d_model
          self.seq_len = seq_len
          self.dropout = nn.Dropout(dropout) # 파이토치에 내장된 dropout 사용

          # 포지션 임베딩을 위한 메트릭스 만들기 (seq_len, d_model) > positional encoding(pe)
          # RNN 계열의 순차 모델과 달리 transformer는 병렬처리되기 때문에 단어의 위치를 알려주는 포지셔널 인코딩이 필요함
          # 결국 matrix 형태로 임베딩matrix + 포지셔널matrix 이기 때문에  matrix 먼저 만들어주어야함.
          pe = torch.zeros(seq_len, d_model)
          # create a vector of shape(seq_len)
          position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)인 vector 생성
          # 논문에 나온 positional encoding 수식 부분 적용(기본 수식대로 해놓은뒤에, sin, cosine 적용해서 각각 홀수, 짝수 d에 적용해줄거임)
          div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) )
          # Apply the sin to 짝수 position / cosine to 홀수
          # 만들어둔 pe matrix에 포지셔널 인코딩 값을 집어넣는 과정
          pe[:, 0::2] = torch.sin(position * div_term)
          pe[:, 1::2] - torch.cos(position * div_term)

          # 집어넣을 문장이 batch size 만큼 있을 것이므로, unsqueeze해서 문장 개수 넣을 수 있게 차원 맞춰줌.
          pe = pe.unsqueeze(0) # (1, seq_l en, d_model)

          # buffer of the module > what is buffer?
          # 모듈 내에서 'pe'에 접근 가능하도록 pe를 설정해주는 작업.(이때 pe는 학습되지 않음.)
          # 포지셔널 인코딩은 절대값이기 때문에 학습되면 안됨. 즉, parameter에서 빼주는 것. 
          self.register_buffer('pe', pe)


     def forward(self, x):
          # 문장에 대한 positional encoding 실행 구간(requires_grad를 False로 해서 학습 안되도록.)
          x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
          return self.dropout(x)



