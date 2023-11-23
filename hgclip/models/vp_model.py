import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import dgl
import tqdm
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

class VLPromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert args.n_ctx >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = clip_model.visual.input_resolution

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init) # [1, n_tkn]
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) # [1, n_tkn, ctx_dim]
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :] # [n_ctx, ctx_dim]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, n_tkn]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # [n_cls, n_tkn, ctx_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class VP_CLIP(nn.Module):
    def __init__(self, args, g, clip_model, classnames):
        super().__init__()
        self.g = g
        self.args = args
        self.prompt_learner = VLPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = self.image_encoder(image.type(self.dtype)).to(text_features.dtype)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.args.dataset == 'cifar-100':
            logits_coarse = logits[:,:20]
            logits_fine = logits[:,20:]

            return logits_coarse, logits_fine

        elif self.args.dataset == 'ethec':
            logits_family = logits[:,:6]
            logits_subfamily = logits[:,6:27]
            logits_genus = logits[:,27:162]
            logits_specific_epithet = logits[:,162:]

            return logits_family, logits_subfamily, logits_genus, logits_specific_epithet
        
        elif self.args.dataset == 'car':
            logits_coarse = logits[:,:9]
            logits_fine = logits[:,9:]

            return logits_coarse, logits_fine
        
        elif self.args.dataset == 'air':
            logits_maker = logits[:,:30]
            logits_family = logits[:,30:100]
            logits_model = logits[:,100:]

            return logits_model, logits_family, logits_maker

        elif self.args.dataset == 'caltech-101':
            logits_l1 = logits[:,:11]
            logits_l2 = logits[:,11:62]
            logits_l3 = logits[:,62:]

            return logits_l1, logits_l2, logits_l3

        elif self.args.dataset == 'food-101':
            logits_l1 = logits[:,:18]
            logits_l2 = logits[:,18:]

            return logits_l1, logits_l2

        elif self.args.dataset == 'fruits-360':
            logits_l1 = logits[:,:4]
            logits_l2 = logits[:,4:73]
            logits_l3 = logits[:,73:]

            return logits_l1, logits_l2, logits_l3
        
        elif self.args.dataset == 'pets':
            logits_l1 = logits[:,:2]
            logits_l2 = logits[:,2:]

            return logits_l1, logits_l2
            
        elif self.args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
            
            return logits

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
