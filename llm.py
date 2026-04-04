import torch
import torch.nn as nn
import transformers
import wandb
from datasets import load_dataset
from transformers import TextStreamer

import math
import random
import time

from dbsf import factorize


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers and "lm_head" not in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def test_perplexity_and_response(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = 'The quick brown fox jumps over the lazy dog.'
    inputs = tokenizer(text, return_tensors='pt', padding=True).to('cuda')

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss.float()).item()

    print(f'Loss: {loss.item():.4f}')
    print(f'Perplexity: {perplexity:.4f}\n')

    print('Model response: ', end='')
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model.generate(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=20,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7
    )


### ZERO-SHOT pruning ###

class FactorizedLinear(nn.Sequential):
    '''Helper to wrap L and R into a single unit'''
    def __init__(self, layer_R, layer_L):
        super().__init__(layer_R, layer_L)

@torch.no_grad()
def prune_model(model):
    layers_to_replace = []
    for name, module in model.named_modules():
        if name == 'lm_head':
            continue
        if isinstance(module, nn.Linear):
            layers_to_replace.append((name, module))
    print(len(layers_to_replace))

    for name, layer in layers_to_replace:
        dtype = layer.weight.dtype
        W = layer.weight.data
        print(W.shape)

        XX = torch.eye(W.shape[1], device=W.device)
        prod, A, B = factorize(W, XX, mask_type='blocks', bsp=0.25, sp=0.5)
        mid_dim = A.shape[1]
        layer_R = nn.Linear(layer.in_features, mid_dim, bias=False, dtype=dtype)
        layer_L = nn.Linear(mid_dim, layer.out_features, bias=layer.bias is not None, dtype=dtype)

        frobenius = torch.norm(prod - W, p='fro')
        print(f'Frobenius norm: {frobenius.item()}')

        A_cpu = A.cpu()
        B_cpu = B.cpu()

        nz_count_A = torch.count_nonzero(A_cpu).item()
        nz_count_B = torch.count_nonzero(B_cpu).item()

        print(f'A.size() = {A_cpu.size()}')
        print(f'B.size() = {B_cpu.size()}')

        n_a = A_cpu.size(dim=0)
        m_a = A_cpu.size(dim=1)
        n_b = B_cpu.size(dim=0)
        m_b = B_cpu.size(dim=1)

        print(f'A has {nz_count_A} non-zero entries ({round(nz_count_A/(n_a*m_a)*100, 1)}%)')
        print(f'B has {nz_count_B} non-zero entries ({round(nz_count_B/(n_b*m_b)*100, 1)}%)')

        layer_R.weight.copy_(B.to(dtype))
        layer_L.weight.copy_(A.to(dtype))

        if '.' in name:
            parent_name, attr_name = name.rsplit('.', 1)
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name
            
        setattr(parent, attr_name, FactorizedLinear(layer_R, layer_L))
        print(f'Replaced {name} with FactorizedLinear (Bottleneck: {mid_dim})')
        torch.cuda.empty_cache()

##########################


### Dataset-based pruning ###
NSAMPLES = 128
NO_FINAL = True
SPARSITY = 0.5

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


DEBUG = True

class DoubleSparse:
    def __init__(self, layer, nofinal=True):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.nofinal = nofinal

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity):
        W = self.layer.weight.data.clone()
        W = W.float()
        tick = time.time()

        W2, _, _ = factorize(W, self.H, sparsity, nofinal=self.nofinal, fixmask=None)

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

        self.layer.weight.data = W2.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


# Calibration on C4
@torch.no_grad()
def llama_sequential(model):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (NSAMPLES, model.seqlen, model.config.hidden_size), dtype=dtype, device="cuda"
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}
            for name in subset:
                gpts[name] = DoubleSparse(subset[name], nofinal=NO_FINAL, fixmask=None)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(NSAMPLES):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
            del outs

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = SPARSITY
                gpts[name].fasterprune(sparsity)
                gpts[name].free()

        outs = torch.zeros_like(inps)
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


# Evaluation on Wikitext2
@torch.no_grad()
def llama_eval(model, testenc, dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device="cuda"
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i]

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
        
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache
