#!/bin/bash
HISTO="features_phikon"
CAPTION="features_mpnet"
COMMON="--histo_feature_dir $HISTO --caption_feature_dir $CAPTION --epochs 20"

# Use alternate tmux socket since default tmux server is broken
TMUX="tmux"

for SEED in 0 1 2; do
    LOGDIR="logs_seed${SEED}"
    OUTDIR="results_seed${SEED}"
    mkdir -p "$LOGDIR"
    
    SEED_COMMON="--seed $SEED --output_dir $OUTDIR $COMMON"

    # ── Baselines ──

    $TMUX new-session -d -s image_only_s$SEED \
        "python main.py --model image_only $SEED_COMMON 2>&1 | tee $LOGDIR/image_only.log; read"

    $TMUX new-session -d -s text_only_s$SEED \
        "python main.py --model text_only $SEED_COMMON 2>&1 | tee $LOGDIR/text_only.log; read"

    $TMUX new-session -d -s concat_fusion_s$SEED \
        "python main.py --model concat_fusion $SEED_COMMON 2>&1 | tee $LOGDIR/concat_fusion.log; read"

    $TMUX new-session -d -s late_fusion_s$SEED \
        "python main.py --model late_fusion $SEED_COMMON 2>&1 | tee $LOGDIR/late_fusion.log; read"

    # ── Full method ──

    $TMUX new-session -d -s full_method_s$SEED \
        "python main.py --model full_method $SEED_COMMON 2>&1 | tee $LOGDIR/full_method.log; read"

    # ── Ablations ──

    $TMUX new-session -d -s no_attn_pool_s$SEED \
        "python main.py --model no_attn_pool $SEED_COMMON 2>&1 | tee $LOGDIR/no_attn_pool.log; read"

    $TMUX new-session -d -s no_lmf_s$SEED \
        "python main.py --model no_lmf $SEED_COMMON 2>&1 | tee $LOGDIR/no_lmf.log; read"

    $TMUX new-session -d -s no_mil_s$SEED \
        "python main.py --model no_mil $SEED_COMMON 2>&1 | tee $LOGDIR/no_mil.log; read"

    # ── Rank sweep ──

    $TMUX new-session -d -s rank_4_s$SEED \
        "python main.py --model full_method --rank 4 $SEED_COMMON 2>&1 | tee $LOGDIR/rank_4.log; read"

    $TMUX new-session -d -s rank_16_s$SEED \
        "python main.py --model full_method --rank 16 $SEED_COMMON 2>&1 | tee $LOGDIR/rank_16.log; read"

    $TMUX new-session -d -s rank_32_s$SEED \
        "python main.py --model full_method --rank 32 $SEED_COMMON 2>&1 | tee $LOGDIR/rank_32.log; read"

    $TMUX new-session -d -s rank_128_s$SEED \
        "python main.py --model full_method --rank 128 $SEED_COMMON 2>&1 | tee $LOGDIR/rank_128.log; read"
done

echo "Launched $($TMUX ls | wc -l) tmux sessions."
echo "Logs → logs_seed[012]/"
echo "Use '$TMUX ls' to list sessions."
echo "Use '$TMUX attach -t <name>' to inspect a session."