import matplotlib.pyplot as plt
import torch
import os

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def sanity_check(self, sentence, block_size):
        wordids = self.tokenizer.encode(sentence)

        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        print("Input sentence: ", sentence)
        print("Input tensor shape: ", input_tensor.shape)

        if self.model.is_causal:
            model_name = 'decoder'
            _, attn_maps, _ = self.model(input_tensor)
        else:
            model_name = 'encoder'
            _, attn_maps = self.model(input_tensor)
        print("Number of attention maps: ", len(attn_maps))

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")

        os.makedirs('./attn_maps', exist_ok=True)
        for j, attn_map in enumerate(attn_maps):
            attn_map = attn_map.squeeze(0).detach().cpu().numpy()
            total_prob_over_rows = torch.sum(torch.tensor(attn_map), dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probabilitiy over rows:", total_prob_over_rows.numpy())
            
            fig, ax = plt.subplots()
            cax = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)
            plt.title(f"Attention Map {j+1}")
            
            plt.savefig(f"./attn_maps/{model_name}-attention_map_{j+1}.png")
