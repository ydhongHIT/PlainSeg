# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask2former_eva2_pascal.py',
    '../_base_/datasets/pascal_context_59_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_adamw_fine_decay.py'
]
crop_size = (480,480)
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth'
model = dict(
    pretrained='pretrain/eva02_L_pt_m38m_p14to16.pt',
    backbone=dict(
        type='EVA2',
        img_size=480, 
        patch_size=16, 
        in_chans=3,
        embed_dim=1024, 
        depth=24,
        num_heads=16, 
        mlp_ratio=4*2/3,      # GLU default
        out_indices=[7, 11, 15, 23],        
        qkv_bias=True, 
        drop_path_rate=0.3,         
        init_values=None, 
        use_checkpoint=True, 
        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        subln=True,
        xattn=False,
        naiveswiglu=True,
        pretrained=None),
    neck=dict(type='Feature2PyramidSlim', embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        input_transform='multiple_select',
        in_channels=[256, 512, 1024, 1024],
        feat_channels=256,
        out_channels=256,
        num_queries=100,
        in_index=[0, 1, 2, 3],
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        with_cp=True,  # set with_cp=True to save memory
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    with_cp=True,  # set with_cp=True to save memory
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None)
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 320))
)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(520, 520), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4096, 520),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LearningRateDecayOptimizerConstructor_custom',
    paramwise_cfg=dict(num_layers=24, decay_rate=0.90, decay_type='layer_wise_vit', custom_keys={'head': dict(lr_mult=10.)}))
#optim_wrapper = dict(
#    type='OptimWrapper',
#    optimizer=optimizer,
#    grad_clip=dict(max_norm=0.01, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data = dict(
    train=dict(
        dataset = dict(
        pipeline=train_pipeline)),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
