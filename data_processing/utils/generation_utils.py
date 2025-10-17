'''
Created on Fri Nov 24 16:19:32 2023

@author: Kendrick
'''
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

@torch.no_grad()
def decoder_greedy_search(tokenizer:PreTrainedTokenizer,
                          decoder:PreTrainedModel,
                          device=torch.device,
                          encoder_hidden_states:torch.tensor=None,
                          encoder_attention:torch.tensor=None,
                          repeat_penality:float=1.,
                          n_gram_repeatation:int=1) -> torch.tensor:
    assert repeat_penality <= 1
    assert (n_gram_repeatation >= 1 and n_gram_repeatation % 1 == 0)
    
    decoder_inputs = tokenizer(tokenizer.bos_token, add_special_tokens=False, return_tensors="pt").to(device)
    
    while decoder_inputs.input_ids[:, -1] != tokenizer.eos_token_id:
        input_length = decoder_inputs.input_ids.size(-1)
        attention_mask = torch.ones((decoder_inputs.input_ids.size())).to(device)
        outputs = decoder(input_ids=decoder_inputs.input_ids,
                          attention_mask=attention_mask,
                          encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention)
        output_last_prob = outputs.logits[:,-1]
        if output_last_prob.argmax(dim=-1) in decoder_inputs.input_ids[0, input_length-n_gram_repeatation : input_length]:
            output_last_prob[:, output_last_prob.argmax(dim=-1)] *= repeat_penality
            
        output_last_token = output_last_prob.argmax(dim=-1).unsqueeze(0)
        decoder_inputs.input_ids = torch.cat((decoder_inputs.input_ids, output_last_token), dim=1)
        
    return decoder_inputs.input_ids