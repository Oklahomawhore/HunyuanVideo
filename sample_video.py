import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import torch

from div_prompt import PromptEmbeddingExtractor

def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    if args.div_prompt:
        if os.path.exists(args.prompt):
            with open(args.prompt, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Prompt list: {prompts}")
        prompt = "<|reserved_special_token_103|>".join(prompts)
        token_id = 128108

        token_ids = hunyuan_video_sampler.text_encoder.text2tokens(prompt, data_type="video")
        crop_start = hunyuan_video_sampler.text_encoder.prompt_template_video.get("crop_start", -1)
        token_ids = token_ids["input_ids"][:, crop_start:]  # skip template embeddings
        sep_mask = (token_ids == token_id)  # shape: (batch_size, seq_len)

        
        embeds1, embeds2 = hunyuan_video_sampler.predict(
            prompt=prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=hunyuan_video_sampler.default_negative_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            encode_prompt=True,
        )

        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = embeds2

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = embeds1
        assert len(prompt_embeds[0]) == len(sep_mask[0]), f"Embeddings and prompts length mismatch: {len(prompt_embeds)} vs {len(sep_mask)}"
        

        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
                "crop_start", 0
            )
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = 512

        # 完成分段嵌入提取
        embeds_list = []
        masks_list = []
        # 处理每个批次的嵌入向量
        for batch_idx in range(sep_mask.shape[0]):
            # 找出所有分隔符的位置
            separator_positions = torch.where(sep_mask[batch_idx])[0]

            # 如果没有找到分隔符，将整个嵌入向量作为一个单独的段
            if len(separator_positions) == 0:
                embeds_list.append(prompt_embeds[batch_idx:batch_idx+1])
                continue
            
            # 添加起始位置 0
            positions = torch.cat([torch.tensor([0], device=separator_positions.device), separator_positions])

            # 添加结束位置（序列长度）
            positions = torch.cat([positions, torch.tensor([min(prompt_embeds.shape[1], prompt_mask.sum().item())], device=separator_positions.device)])

            # 根据分隔符位置切分嵌入向量
            batch_embeds = []
            batch_masks = []
            for i in range(len(positions) - 1):
                start_pos = positions[i].item()
                # 如果是分隔符位置，则跳过这个标记本身
                if i > 0:
                    start_pos = start_pos + 1
                end_pos = positions[i + 1].item()

                # 确保切片有内容
                if end_pos > start_pos:
                    segment = prompt_embeds[batch_idx:batch_idx+1, start_pos:end_pos]
                    # append padding token embeddings
                    n_dim = segment.shape[-1]
                    padding = torch.zeros((1, max_length - (end_pos - start_pos), n_dim), device=segment.device)
                    segment = torch.cat([segment, padding], dim=1)
                    mask = torch.ones_like(prompt_mask[batch_idx:batch_idx+1, start_pos:end_pos], device=segment.device, dtype=int)
                    mask_padding = torch.zeros((1, max_length - (end_pos - start_pos)), device=segment.device, dtype=int)
                    mask = torch.cat([mask, mask_padding], dim=1)
                    print(f"saving segment: {segment.shape}")
                    batch_embeds.append(segment)
                    batch_masks.append(mask)

            embeds_list.append(batch_embeds)
            masks_list.append(batch_masks)
        logger.info(f"成功将嵌入向量分割为 {len(embeds_list)} 批次，共 {sum(len(batch) for batch in embeds_list)} 个分段")

        for batch_idx, (batch_embeds, batch_masks) in enumerate(zip(embeds_list, masks_list)):
            for i, (embeds, mask) in enumerate((zip(batch_embeds, batch_masks))):
                with open(os.path.join(args.save_path, f"batch{batch_idx}_scene{i}_embeddings.pt"), "wb") as f:
                    torch.save({"prompt_embeds" : embeds, "attention_mask": mask}, f)
            with open(os.path.join(args.save_path, f"batch{batch_idx}_global_embeddings.pt"), "wb") as f:
                torch.save({"prompt_embeds_2" : prompt_embeds_2, "prompt_mask_2": prompt_mask_2}, f)

    else:
        
        dirs = args.prompt_embed.split("|||")
        embeds_list = []
        for dir in dirs:
            with open(dir, "rb") as f:
                embed = torch.load(f)
                embeds_list.append(embed)
        
        logger.info(f"Prompt embedding shape: {embeds_list[0]['prompt_embeds'].shape}")
        logger.info(f"Global embedding shape: {embeds_list[1]['prompt_embeds_2'].shape}")
        outputs = hunyuan_video_sampler.predict(
            prompt=None if args.prompt_embed is not None else args.prompt,
            prompt_embeds=embeds_list[0]["prompt_embeds"],
            prompt_embeds_2=embeds_list[1]["prompt_embeds_2"],
            attention_mask=embeds_list[0]["attention_mask"],
            attention_mask_2=embeds_list[1]["prompt_mask_2"],
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
