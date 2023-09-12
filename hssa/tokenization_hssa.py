from transformers.models.mpnet.tokenization_mpnet_fast import MPNetTokenizerFast
from transformers.tokenization_utils import AddedToken
import torch


class HSSATokenizer(MPNetTokenizerFast):
    def __init__(
        self,
        sys_token='<system>',
        user_token='<user>',
        **kwargs
    ):
        # sending them as keyword argument `additional_special_tokens` will add these tokens to `additional_special_tokens`
        additional_special_tokens = [
            AddedToken(sys_token, lstrip=False, rstrip=False),
            AddedToken(user_token, lstrip=False, rstrip=False),
        ]

        super().__init__(additional_special_tokens=additional_special_tokens, **kwargs)
    
        self.sys_token, self.user_token = self.additional_special_tokens
        self.sys_token_id, self.user_token_id = self.additional_special_tokens_ids

    # overrides Base's __call__
    def __call__(self, text, **kwargs):
        """
        Input text is a list of dialogues, where each dialogue is a list of objects with following schema:
        {
            "type": "object",
            "properties":
            {
                "utterance": {"type": "string"},
                "speaker": {"type": "number"}
            }
        }
        User and system speakers correspond to speaker==0 and speaker==1 respectively.
        """
        tokenizer = super().__call__
        uts = []
        uts_mask = []
        max_dia_len = max(len(dia) for dia in text)

        for dia in text:
            for item in dia:
                role_token = self.sys_token if item['speaker'] else self.user_token
                uts.append(role_token + item['utterance'])
            for _ in range(max_dia_len - len(dia)):
                uts.append('')
            uts_mask.append([1] * len(dia) + [0] * (max_dia_len - len(dia)))
        
        tok_res = tokenizer(uts, padding='longest', return_tensors='pt')
        tok_res['segment_size'] = max_dia_len
        tok_res['utterance_mask'] = torch.tensor(uts_mask)
        return tok_res
