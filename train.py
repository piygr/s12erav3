import torch
import tiktoken
from model import GPT, GPTConfig

# model = GPT.from_pretrained('gpt2')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


model = GPT(GPTConfig())
model.to(device)
prevloss = 100
epochs = 150
B = 256
T = 32
train_loader = DataLoaderLite(B=B, T=T)
steps_per_epoch = len(train_loader.tokens) // (B * T)
num_steps = epochs * steps_per_epoch
print(num_steps)

# NEW CODE
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(num_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if i % steps_per_epoch == 0:
        print(f'step{i}, loss: {loss.item()}')
        if prevloss > loss.item():
            torch.save(model.state_dict(), "model.pth")
            prevloss = loss.item()

print(loss.item())
torch.save(model.state_dict(), "model.pth")



import sys;

sys.exit(0)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
tokenizer = tiktoken.get_encoding('gpt2')
text_sample = "Tell me what you know"
input_ids = [tokenizer.encode(text_sample)]

output_tensor = model.generate(input_ids, 50, EOS_TOKEN_ID=50256)
generated_text = tokenizer.decode(output_tensor[0].tolist())
print(generated_text)

'''
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0]  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

'''