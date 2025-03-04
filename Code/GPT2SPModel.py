import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2PreTrainedModel, GPT2Config
import numpy as np
import torch.nn as nn

class GPT2SPModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.dense1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.dense2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        # self.dense3 = nn.Linear(config.n_embd, 1, bias=False)
        # self.score = nn.Linear(50, 1, bias=False)

        # self.transformer = GPT2Model.from_pretrained("openai-community/gpt2")
        # self.dense1 = nn.Linear(768, 128, bias=False)
        # self.dense2 = nn.Linear(128, 32, bias=False)
        # self.score = nn.Linear(32, 1, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(self, emb):
        transformer_outputs = self.transformer(emb)
        hidden_states = transformer_outputs[0]

        # print(hidden_states.shape)

        # MLP Layer
        hidden_states = self.dense1(hidden_states)
        # print(hidden_states.shape)
        hidden_states = self.dense2(hidden_states)
        # print(hidden_states.shape)
        logits = self.score(hidden_states)

        # hidden_states = np.squeeze(hidden_states)
        # hidden_states = self.dense3(hidden_states)
        # # print(hidden_states.shape)
        # hidden_states = np.squeeze(hidden_states)
        # # print(hidden_states.shape)
        # logits = self.score(hidden_states)
        # # print(logits.shape)
        # logits = np.squeeze(logits)
        #
        # return logits

        if emb is not None:
            batch_size, sequence_length = emb.shape[:2]
        else:
            batch_size, sequence_length = emb.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if emb is not None:
                sequence_lengths = torch.ne(emb, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[range(batch_size), sequence_lengths]

        pooled_logits = torch.squeeze(pooled_logits)

        return pooled_logits

# def GPTencode(tokenizer, sentence):
#     SEQUENCE_LEN = 300
#     encoded = tokenizer.encode(sentence)
#     if len(encoded) > SEQUENCE_LEN:
#         encoded = encoded[:SEQUENCE_LEN]
#     elif len(encoded) < SEQUENCE_LEN:
#         padding = SEQUENCE_LEN - len(encoded)
#         for _ in range(padding):
#             encoded.append(3)
#     return encoded

# LM = GPT2Tokenizer.from_pretrained('gpt2')
# LM.pad_token = '[PAD]'
#
# random_sentences = ["Let's see if this works.", "Let's see if this works again.", "Third attempt", "One last for rounding off the batch size"]
#
# embs = []
# emb = GPTencode(LM, random_sentences[0])
# embs.append(emb)
# emb = GPTencode(LM, random_sentences[1])
# embs.append(emb)
# embs = torch.Tensor(embs).to(torch.int)
#
# config = GPT2Config(num_labels=1, pad_token_id=50256)
#
# model = GPT2SPModel(config)
# output = model.forward(embs)
# print(output.shape)






