#!/bin/bash
# filepath: /root/HunyuanVideo/batch_inference.sh

# 设置基本参数
EMBEDDINGS_DIR="/root/HunyuanVideo/results"  # 嵌入文件目录
OUTPUT_DIR="./batch_results"                 # 输出视频目录
MODEL_BASE="/root/HunyuanVideo/ckpts"       # 模型目录
PATTERN="*.pt"                               # 文件匹配模式

# 视频生成参数
VIDEO_HEIGHT=720
VIDEO_WIDTH=1280
VIDEO_LENGTH=129
INFER_STEPS=50
CFG_SCALE=3.0
NEGATIVE_PROMPT="低质量, 糟糕的艺术, 丑陋, 变形"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 获取开始时间，用于记录总耗时
START_TIME=$(date +%s)

# 计数器
TOTAL_FILES=$(find $EMBEDDINGS_DIR -name "$PATTERN" | wc -l)
PROCESSED=0

# 显示开始信息
echo "开始处理 $EMBEDDINGS_DIR 目录下的 $TOTAL_FILES 个嵌入文件..."
echo "输出将保存到 $OUTPUT_DIR"
echo "----------------------------------------"

# 循环处理每个嵌入文件
find $EMBEDDINGS_DIR -name "$PATTERN" | sort | while read -r file; do
    # 获取文件名（不含路径和扩展名）
    filename=$(basename "$file" .pt)
    
    # 更新并显示进度
    PROCESSED=$((PROCESSED + 1))
    echo "[$PROCESSED/$TOTAL_FILES] 处理文件: $filename"
    
    # 为每个文件调用sample_video.py
    python3 sample_video.py \
        --model-base $MODEL_BASE \
        --video-size $VIDEO_HEIGHT $VIDEO_WIDTH \
        --video-length $VIDEO_LENGTH \
        --infer-steps $INFER_STEPS \
        --cfg-scale $CFG_SCALE \
        --prompt-embed "$file|||/root/HunyuanVideo/batch0_global_embeddings.pt" \
        --flow-reverse \
        --save-path "$OUTPUT_DIR" \
        --text-len 512 \
        --use-cpu-offload
    
    echo "----------------------------------------"
done

# 计算并显示总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "所有视频生成完成！"
echo "总处理文件数: $TOTAL_FILES"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "结果保存在 $OUTPUT_DIR"