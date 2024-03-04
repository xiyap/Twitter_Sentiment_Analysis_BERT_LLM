from transformers import AutoModel
import torch.nn as nn

model_name = 'distilbert/distilbert-base-uncased'
model = AutoModel.from_pretrained(model_name)

class TweetClassifer(nn.Module):
    def __init__(self, model, num_classes = 2):
        super(TweetClassifer, self).__init__()

        self.base_model = model
        self.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, num_classes),
            nn.Sigmoid()
            )

    def forward(self, forward_seq, forward_mask):
        cls_hs = self.base_model(forward_seq, attention_mask = forward_mask)
        cls_output = cls_hs.last_hidden_state[:, 0, :]
        output = self.classifier(cls_output)
        return output