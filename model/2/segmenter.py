import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.clip import build_model
from .layers import Neck, Decoder, Projector, CrossModalTransformer
from .fusion import Fusion
from .dinov2.models.vision_transformer import vit_base,vit_large





class DETRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Text Encoder

        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.txt_backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size, cfg.txtual_adapter_layer,cfg.txt_adapter_dim).float()
        self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
    
       # Fix Backbone
        for param_name, param in self.txt_backbone.named_parameters():
            if 'adapter' not in param_name : 
                param.requires_grad = False       
   

        state_dict = torch.load(cfg.dino_pretrain) 
        if cfg.dino_name=='dino-base':
            self.dinov2 = vit_base(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        else:
            self.dinov2=vit_large(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        self.dinov2.load_state_dict(state_dict, strict=False)

        for param_name, param in self.dinov2.named_parameters():
            if 'adapter' not in param_name:
                param.requires_grad = False
        
        # Multi-Modal Decoder
        self.neck = Neck(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)

        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        # 视觉对语言影响的 Transformer 解码器
        self.vis2txt_decoder = CrossModalTransformer(d_model=cfg.vis_dim, nhead=cfg.num_head, num_layers=4)



        
        


    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024

        vis, word, state= self.fusion(img, word, self.txt_backbone, self.dinov2)
        #for i, v in enumerate(vis):
            #print(f"视觉特征 (DINOv2 输出) 形状: {v.shape}")  # 视觉特征
        #print(f"文本特征 (CLIP 输出) 形状: {word.shape}")  # 文本特征
        #print(f"全局状态特征 形状: {state.shape}")  # 全局特征
        


        
        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        #print(f"fq 形状: {fq.shape}")  # 输出 fq 的形状
        
        b, c, h, w = fq.size()
        
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)
        
        fq_flattened = fq.view(b, c, -1).permute(0, 2, 1)  # 变为 (b, hw, c)
        state_expanded = state.unsqueeze(1)  # (b, 1, c)
        A = self.vis2txt_decoder(state_expanded, fq_flattened)  # 交互得到新的全局文本特征 A
        A = A.squeeze(1)  # (b, c)
        



        # b, 1, 104, 104
        pred = self.proj(fq, A)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask) 
            return pred.detach(), mask, loss
        else:
            return pred.detach()
