import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from collections import OrderedDict

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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # [n_cls, ctx_dim]
        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        vis_dim = clip_model.visual.output_dim

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

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, context_length]
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
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CoCoOp_CLIP(nn.Module):
    def __init__(self, args, clip_model, classnames):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts # [n_cls, ctx_length]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.args = args

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if self.args.dataset == 'cifar-100':
            logits_coarse = logits[:,:20]
            logits_fine = logits[:,20:]

            return logits_coarse, logits_fine
        
        elif self.args.dataset == 'ethec':
            text_features_family, text_features_subfamily, text_features_genus, text_features_specific_epithet = text_features[:6,], text_features[6:27,], text_features[27:162,], text_features[162:,]
            text_features_family = text_features_family / text_features_family.norm(dim=-1, keepdim=True)
            text_features_subfamily = text_features_subfamily / text_features_subfamily.norm(dim=-1, keepdim=True)
            text_features_genus = text_features_genus / text_features_genus.norm(dim=-1, keepdim=True)
            text_features_specific_epithet = text_features_specific_epithet / text_features_specific_epithet.norm(dim=-1, keepdim=True)

            logits_family = logit_scale * image_features @ text_features_family.t()
            logits_subfamily = logit_scale * image_features @ text_features_subfamily.t()
            logits_genus = logit_scale * image_features @ text_features_genus.t()
            logits_specific_epithet = logit_scale * image_features @ text_features_specific_epithet.t()

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
