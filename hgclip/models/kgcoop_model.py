import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from collections import OrderedDict
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # prompts.shape = [n_cls, ctx_length, ctx_dim]
        # tokenized_prompts.shape = [n_cls, ctx_length]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if args.ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init) # [1, context_length]
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) # [1, context_length, ctx_dim]
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # [n_ctx, ctx_dim]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if args.csc is True:
                print("Initializing class-specific contexts")
                n_ctx = args.n_ctx
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                n_ctx = args.n_ctx
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized [n_ctx, ctx_dim]
        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
                
        temp = args.ctx_init
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.to(args.device)

        with torch.no_grad():
            clip_model.to(args.device)
            text_features = clip_model.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512)),
            ("relu", nn.ReLU(inplace=True))
            #("linear2", nn.Linear(128, 512))
        ]))

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(args.device)  # [n_cls, context_length]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, context_length, ctx_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts  # (n_cls, n_ctx, dim)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class kgcoop_vl(nn.Module):
    def __init__(self, args, clip_model, classnames):
        super().__init__()
        self.args = args
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.ori_embedding = self.prompt_learner.text_features
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.meta_net = self.prompt_learner.meta_net
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

    def forward(self, image, class_image_features=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_old = self.ori_embedding
        image_features = self.image_encoder(image.type(self.dtype)).to(text_features.dtype)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        cos = nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = cos(text_features,text_features_old)
        score = 1.0-torch.mean(score)

        if self.args.dataset == 'cifar-100':
            logits_coarse = logits[:,:20]
            logits_fine = logits[:,20:]

            return logits_coarse, logits_fine, score

        elif self.args.dataset == 'ethec':
            logits_family = logits[:,:6]
            logits_subfamily = logits[:,6:27]
            logits_genus = logits[:,27:162]
            logits_specific_epithet = logits[:,162:]

            return logits_family, logits_subfamily, logits_genus, logits_specific_epithet, score
        
        elif self.args.dataset == 'car':
            logits_coarse = logits[:,:9]
            logits_fine = logits[:,9:]

            return logits_coarse, logits_fine, score
        
        elif self.args.dataset == 'air':
            logits_maker = logits[:,:30]
            logits_family = logits[:,30:100]
            logits_model = logits[:,100:]

            return logits_model, logits_family, logits_maker, score

        elif self.args.dataset == 'caltech-101':
            logits_l1 = logits[:,:11]
            logits_l2 = logits[:,11:62]
            logits_l3 = logits[:,62:]

            return logits_l1, logits_l2, logits_l3, score

        elif self.args.dataset == 'food-101':
            logits_l1 = logits[:,:18]
            logits_l2 = logits[:,18:]

            return logits_l1, logits_l2, score

        elif self.args.dataset == 'fruits-360':
            logits_l1 = logits[:,:4]
            logits_l2 = logits[:,4:73]
            logits_l3 = logits[:,73:]

            return logits_l1, logits_l2, logits_l3, score

        elif self.args.dataset == 'pets':
            logits_l1 = logits[:,:2]
            logits_l2 = logits[:,2:]

            return logits_l1, logits_l2, score
            
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
