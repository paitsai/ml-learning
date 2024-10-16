#实现了注意力机制的RNN网络来实现一个简单的日期翻译功能

import torch as th
import torch.nn as nn

EMBEDDING_LENGTH=128 # 需要embedding的元素个数
OUTPUT_LENGTH=10


class AttentionRNN(nn.Module):
    def __init__(self,embedding_dim=32,encoder_dim=32,decoder_dim=32,dropout_rate=0.3):
        
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.dropout_rate = dropout_rate
        
        
        self.dropout=nn.Dropout(dropout_rate)
        self.embedding=nn.Embedding(EMBEDDING_LENGTH,embedding_dim)
        
        self.att_linear=nn.Linear(encoder_dim*2+decoder_dim,1)
        self.softmax=nn.Softmax(dim=-1)
        
        
        self.encoder=nn.LSTM(input_size=embedding_dim,
                             hidden_size=decoder_dim,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True
                             )
        
        # 因为编码器是双向的rnn所以解码器的输入这里×2
        self.decoder=nn.LSTM(input_size=EMBEDDING_LENGTH+2*encoder_dim,
                             hidden_size=decoder_dim,
                             num_layers=1,
                             batch_first=True
                             )
        
        self.output_linear=nn.Linear(encoder_dim,EMBEDDING_LENGTH)
        
        
    def  forward(self,x,n_output=OUTPUT_LENGTH):
        
        # x [batch_size,sequence_id,embedding_length]
        
        batch_size,squence_size=x.shape[0:2]
        
        x=self.dropout(self.embedding(x))
        
        a,_=self.encoder(x) # a [batch_size,sqeunce_length,hidden_size] 
        
        prev_state=th.zeros((batch_size,1,self.decoder_dim)) # 保存上一步的解码器状态
        prev_y=th.zeros((batch_size,1,EMBEDDING_LENGTH))
        curr_state=None
        y=th.tensor((batch_size,n_output,EMBEDDING_LENGTH))
        
        # 上下文注意力由 上一步的解码器的隐状态和编码器的输入cat而成
        
        # 解码器输出y由解码器隐藏状态得到
        y=th.empty((batch_size,OUTPUT_LENGTH,EMBEDDING_LENGTH))
        
        for i in range(n_output):
            prev_s_repeat=prev_state.repeat(1,squence_size,1)
            
            prev_y_repeat=prev_y.repeat(1,squence_size,1)
            
            attention_input=th.cat((prev_s_repeat,a),dim=2).reshape(batch_size*squence_size,-1)
            alpha=self.softmax(self.att_linear(attention_input).reshape(batch_size,squence_size)) #[batch_size,squence_length,1]
            alpha=alpha.reshape(batch_size,squence_size,1)
            c=th.sum(a*alpha,1) # [batch_size,1,encode_hidden_dim]
            c=c.unsqueeze(1) # [batch_size,encode_hidden_dimension

            decoder_input=th.cat((prev_y,c),dim=2)
            
            
            if curr_state is None:
                prev_state,curr_state=self.decoder(decoder_input)
            else:
                prev_state,curr_state=self.decoder(decoder_input,curr_state)
                
            prev_y=self.output_linear(prev_state)
            y[:,i]=prev_y.squeeze(1)
            
        return y
    
    
    # 在编码的过程中也不难发现：RNN存在并行化低的弊端
    # 测试代码
def test_attention_rnn():
    # 创建模型
    model = AttentionRNN()

    # 创建输入数据，假设 batch_size=2，sequence_length=4
    batch_size = 2
    sequence_length = 4
    input_data = th.randint(0, EMBEDDING_LENGTH, (batch_size, sequence_length))

    # 模型前向传播
    output = model(input_data)

    # 打印输出
    print("输出形状:", output.shape)  # 应该是 [batch_size, OUTPUT_LENGTH, EMBEDDING_LENGTH]
    print("输出内容:", output)

test_attention_rnn()
    
    
    
    
    
    
    
            
            
            
            
        
        
        
        
        
        
        
        
        
        
                