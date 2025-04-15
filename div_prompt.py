import os
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle

from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.text_encoder import TextEncoder
from hyvideo.vae import AutoencoderKLCausal3D
from hyvideo.inference import HunyuanVideoSampler


def parse_args():
    parser = argparse.ArgumentParser(description="提取HunyuanVideo的提示词嵌入向量")
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default=None, 
        help="包含提示词的文本文件，每行一个提示词"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None, 
        help="直接输入提示词，多个提示词用特殊标记分隔"
    )
    parser.add_argument(
        "--separator", 
        type=str, 
        default="|||", 
        help="提示词分隔符"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./prompt_embeddings", 
        help="输出嵌入向量的目录"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="运行设备"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="计算精度"
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        required=True,
        help="文本编码器路径"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="pt",
        choices=["pt", "pkl", "npy"],
        help="保存格式: pt (PyTorch), pkl (Pickle), npy (Numpy)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="处理批次大小"
    )
    return parser.parse_args()


class PromptEmbeddingExtractor:
    def __init__(self, hunyuanSampler: HunyuanVideoSampler):
        self.sampler = hunyuanSampler
        self.text_encoder = hunyuanSampler.pipeline.text_encoder
    
        self.pipeline = hunyuanSampler.pipeline
    
    @torch.no_grad()
    def extract_embeddings(self, prompts):
        """从提示词列表中提取嵌入向量"""
        
        sep_token_id = 128108
        prompt = "<|reserved_special_token_103|>".join(prompts)
        prompt_embeds, _, _, _ = self.pipeline.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_videos_per_prompt=1,
                    do_classifier_free_guidance=False,
                    data_type="video"
                )
        input_ids = self.pipeline.text_encoder.text2tokens(prompt, data_type="video")
        sep_mask = (input_ids == sep_token_id)  # shape: (batch_size, seq_len)
        sep_pos = sep_mask.int().argmax(dim=1)  # shape: (batch_size,)


        
        return prompt_embeds
    
    def save_embeddings(self, embeddings, output_dir, format="pt", prefix="prompt_embedding"):
        """保存嵌入向量到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, embedding in enumerate(embeddings):
            output_path = os.path.join(output_dir, f"{prefix}_{i}")
            
            if format == "pt":
                torch.save(embedding.cpu(), f"{output_path}.pt")
            elif format == "pkl":
                with open(f"{output_path}.pkl", "wb") as f:
                    pickle.dump(embedding.cpu(), f)
            elif format == "npy":
                np.save(f"{output_path}.npy", embedding.cpu().numpy())
            
        print(f"已将 {len(embeddings)} 个嵌入向量保存到 {output_dir}")
        
        # 同时保存合并的嵌入向量以便批量处理
        combined_path = os.path.join(output_dir, f"{prefix}_all")
        combined_embeddings = torch.cat([emb.cpu() for emb in embeddings], dim=0)
        
        if format == "pt":
            torch.save(combined_embeddings, f"{combined_path}.pt")
        elif format == "pkl":
            with open(f"{combined_path}.pkl", "wb") as f:
                pickle.dump(combined_embeddings, f)
        elif format == "npy":
            np.save(f"{combined_path}.npy", combined_embeddings.numpy())
        
        print(f"已将合并嵌入向量保存到 {combined_path}.{format}")


def main():
    args = parse_args()
    
    # 获取提示词列表
    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    elif args.prompt:
        prompts = args.prompt.split(args.separator)
    else:
        raise ValueError("必须提供 --prompt 或 --prompt_file 参数")
    
    prompts = [p.strip() for p in prompts]
    print(f"共加载 {len(prompts)} 个提示词")
    
    # 创建提取器
    extractor = PromptEmbeddingExtractor(
        args.text_encoder_path, 
        device=args.device, 
        precision=args.precision
    )
    
    # 提取嵌入向量
    embeddings = extractor.extract_embeddings(prompts, batch_size=args.batch_size)
    
    # 保存嵌入向量
    extractor.save_embeddings(embeddings, args.output_dir, format=args.save_format)
    
    print("提示词嵌入提取完成")


if __name__ == "__main__":
    main()