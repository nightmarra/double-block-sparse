import torch
import torch.nn as nn
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import get_balanced_memory
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TextStreamer

import gc
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
        print(f"\nCurrently pruning: {name}")
        dtype = layer.weight.dtype
        W = layer.weight.data

        XX = torch.eye(W.shape[1], device=W.device, dtype=dtype)
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
        torch.cuda.empty_cache()

    filepath = "./../pruned/"
    model.save_pretrained(filepath)

##########################


### Dataset-based pruning ###
NSAMPLES = 128
NO_FINAL = True
SPARSITY = 0.5

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

        W2, _, _ = factorize(W=W, 
                             XX=self.H, 
                             mask_type='blocks', 
                             bsp=sparsity/2, 
                             sp=sparsity, 
                             run_finalize=not self.nofinal)

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


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    elif isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    return obj


# Calibration on C4
@torch.no_grad()
def llama_sequential(model, dataloader):
    for _, module in model.named_modules():
        remove_hook_from_module(module)
    model.cpu()
    torch.cuda.empty_cache()

    print("Starting calibration...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_batch = next(iter(dataloader))[0]
    seqlen = first_batch.shape[1]

    inps = torch.zeros(
        (NSAMPLES, seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {"i": 0, "kwargs": None} # kwargs = rotary embeddings, attention mask...

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            cache["kwargs"] = _move_to_device(kwargs, 'cpu') 
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].cpu()) 
        except ValueError:
            pass

    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    caught_kwargs = cache["kwargs"]

    # everything should be on CPU at this point
    # now we begin the process (for each layer):
    # 1. move layer + inputs to GPU
    # 2. process
    # 3. return to CPU
    # loop start
    for i in range(len(layers)):
        # STEP 1 (layer)
        layer = layers[i].to(device)
        current_layer_kwargs = _move_to_device(caught_kwargs, device)

        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}
            for name in subset:
                gpts[name] = DoubleSparse(subset[name], nofinal=NO_FINAL)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(NSAMPLES):
                # STEP 1 (inputs)
                cur_inp = inps[j:j+1].to(device)
                # STEP 2
                layer_out = layer(cur_inp, **current_layer_kwargs)[0]
                # STEP 3 (inputs) to save space on the GPU
                inps[j] = layer_out.squeeze(0).cpu()
                del cur_inp, layer_out

            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = SPARSITY
                gpts[name].fasterprune(sparsity)
                gpts[name].free()

        # STEP 3 (layer)
        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    print("Sequential pruning finished. Model is on CPU.")
    # loop end

    # now the model is on CPU, we return it back to GPUs
    max_memory = get_balanced_memory(
        model,
        dtype=dtype,
        low_zero=False,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        dtype=dtype,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    model = dispatch_model(model, device_map=device_map)
    print(f"Model dispatched on GPUs with device map: {device_map}")

    return model


# Inner loop for evaluation extracted
# Keeps the model on CPU
@torch.no_grad()
def _run_layer_loop(model, testenc_ids, nsamples, seqlen):
    for _, module in model.named_modules():
        remove_hook_from_module(module)
    model.cpu()
    torch.cuda.empty_cache()

    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            cache["kwargs"] = _move_to_device(kwargs, "cpu")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc_ids[:, (i * seqlen):((i+1) * seqlen)]
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    caught_kwargs = cache["kwargs"]

    for i in range(len(layers)):
        print(f"Currently at layer: {i}")
        layer = layers[i].to(device)
        current_kwargs = _move_to_device(caught_kwargs, device)

        for j in range(nsamples):
            cur_inp = inps[j:j+1].to(device)
            layer_out = layer(cur_inp, **current_kwargs)[0]
            inps[j] = layer_out.squeeze(0).cpu()
            del cur_inp, layer_out

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.model.norm.cpu()
    model.lm_head.cpu()
    return inps, model.model.norm, model.lm_head


# Evaluation on Wikitext2
@torch.no_grad()
def llama_eval(model, testenc):
    print("Evaluating PPL...")
    testenc_ids = testenc.input_ids
    seqlen = model.seqlen
    nsamples = testenc_ids.numel() // seqlen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, norm, lm_head = _run_layer_loop(model, testenc_ids, nsamples, seqlen)
    norm = norm.to(device)
    lm_head = lm_head.to(device)

    nlls = []
    for i in range(nsamples):
        hidden_states = norm(inps[i].unsqueeze(0).to(device))
        logits = lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testenc_ids[:, (i * seqlen):((i+1) * seqlen)][:, 1:].to(device)
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(
                -1, 
                shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)
        del hidden_states, logits, shift_logits, shift_labels
        torch.cuda.empty_cache()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"Perplexity: {ppl.item():.3f}")

    model.config.use_cache = use_cache
    return float(ppl.item())


# PPL + KL calculation pipeline
# Runs eval on both models, keeps them on CPU
# Then performs the process (for each SAMPLE):
    # 1. move model to GPU
    # 2. process
    # 3. return to CPU
@torch.no_grad()
def ppl_kl_pipeline(filepath_dense, filepath_pruned, testenc, seqlen=2048):
    testenc_ids = testenc.input_ids
    nsamples = testenc_ids.numel() // seqlen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # first run evaluation on dense model
    print("=== Layer loop: dense model ===")
    model_p = AutoModelForCausalLM.from_pretrained(filepath_dense)
    model_p.seqlen = seqlen
    model_p.config.use_cache = False
    inps_p, norm_p, lm_head_p = _run_layer_loop(model_p, testenc_ids, nsamples, seqlen)
    del model_p
    gc.collect()
    torch.cuda.empty_cache()

    # then run evaluation on the pruned model
    print("=== Layer loop: pruned model ===")
    model_q = AutoModelForCausalLM.from_pretrained(filepath_pruned)
    model_q.seqlen = seqlen
    model_q.config.use_cache = False
    inps_q, norm_q, lm_head_q = _run_layer_loop(model_q, testenc_ids, nsamples, seqlen)
    del model_q
    gc.collect()
    torch.cuda.empty_cache()

    # now PPL and KL
    print("=== Computing PPL and KL ===")
    norm_p = norm_p.to(device)
    norm_q = norm_q.to(device)
    lm_head_p = lm_head_p.to(device)
    lm_head_q = lm_head_q.to(device)

    nlls_p = []
    nlls_q = []
    kl_accum = []

    # compute both values
    # loop start
    for i in range(nsamples):
        h_p = norm_p(inps_p[i].unsqueeze(0).to(device))
        h_q = norm_q(inps_q[i].unsqueeze(0).to(device))
        logits_p = lm_head_p(h_p) # shape: (1, seqlen, vocab_size)
        logits_q = lm_head_q(h_q) # shape: (1, seqlen, vocab_size)

        # target shape: (seqlen-1, vocab)
        shift_p = logits_p[:, :-1, :].contiguous().squeeze(0)
        shift_q = logits_q[:, :-1, :].contiguous().squeeze(0)
        labels = testenc_ids[:, (i * seqlen):((i+1) * seqlen)][:, 1:].to(device)

        # PPL
        for shift, nlls in [(shift_p, nlls_p), (shift_q, nlls_q)]:
            loss = nn.CrossEntropyLoss()(
                shift.view(-1, shift.size(-1)),
                labels.view(-1),
            )
            nlls.append(loss.float() * seqlen)

        # KL via softmax
        log_p = torch.log_softmax(shift_p, dim=-1)
        log_q = torch.log_softmax(shift_q, dim=-1)
        kl_tokens = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        kl_accum.append(kl_tokens.float().cpu())

        del h_p, h_q, logits_p, logits_q
        del shift_p, shift_q, log_p, log_q, kl_tokens, labels
        torch.cuda.empty_cache()
    # loop end

    ppl_p = torch.exp(torch.stack(nlls_p).sum() / (nsamples * seqlen))
    ppl_q = torch.exp(torch.stack(nlls_q).sum() / (nsamples * seqlen))
    per_token_kl = torch.cat(kl_accum) # shape: (nsamples * (seqlen - 1))

    kl_results = {
        "mean_kl":   per_token_kl.mean().item(),
        "median_kl": per_token_kl.median().item(),
        "max_kl":    per_token_kl.max().item(),
        "per_token": per_token_kl,
    }

    print(f"PPL dense:  {ppl_p.item():.3f}")
    print(f"PPL pruned: {ppl_q.item():.3f}")
    print(f"KL(dense || pruned) mean: {kl_results['mean_kl']:.4f}")
    print(f"KL median: {kl_results['median_kl']:.4f}")
    print(f"KL max: {kl_results['max_kl']:.4f}")

    return float(ppl_p.item()), float(ppl_q.item()), kl_results
