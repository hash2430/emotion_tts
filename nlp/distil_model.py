from transformers import DistilBertModel, DistilBertPreTrainedModel, DistilBertConfig
import torch
class EmotionDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = DistilBertModel.from_pretrained('monologg/distilkobert')
        self.seq_len = config.max_length
        self.linear = torch.nn.Linear(config.hidden_size, 4)

        # Erasing below line gives 7 to 9 %p performance gain
        # self.init_weights()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_layer = output
        # I wish this returned attention and other hidden layers for further analysis,
        # but this pretrained model does not provide that.
        output = output[0]
        pooled = output[:,0,:] # batch, length, hidden_dim
        linear_output = self.linear(pooled)
        return linear_output, last_hidden_layer

