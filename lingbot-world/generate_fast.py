import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import merge_video_audio, save_video, str2bool
from wan.utils.selector_defaults import (
    SELECTOR_CKPT_NAME,
    resolve_selector_ckpt,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
        "image":
            "examples/04/image.jpg",
    },
}


def _resolve_input_path(path, should_exist=True):
    if path is None or os.path.isabs(path):
        return path

    search_roots = [
        os.getcwd(),
        SCRIPT_DIR,
        WORKSPACE_DIR,
    ]
    for root in search_roots:
        candidate = os.path.join(root, path)
        if not should_exist or os.path.exists(candidate):
            return candidate
    return path


def _apply_ws_v2_selector(args):
    """Future-use selector + world_state.v2 features (shared by v2/v3)."""
    args.enable_motion_adaptive_kv_eviction = True
    args.selector = "learned"
    if not getattr(args, "selector_ckpt", None):
        args.selector_ckpt = resolve_selector_ckpt(
            None, schema="world_state_future")


def _apply_ws_v3_consolidation(args):
    """v3 package: SWTP + consolidation full.

    Default knobs match default_loop sweep winner ``ws_v3_a05_g64``:
    rank_alpha=0.5, gist_tokens=64, l2_bottom_ratio=0.5.
    """
    args.enable_swtp = True
    args.consolidation = "full"
    if int(getattr(args, "archive_diversity_pool", 0) or 0) <= 0:
        args.archive_diversity_pool = 4
    if getattr(args, "consol_rank_alpha", None) is None:
        args.consol_rank_alpha = 0.5
    if getattr(args, "consol_gist_tokens", None) is None:
        args.consol_gist_tokens = 64
    if getattr(args, "consol_l2_bottom_ratio", None) is None:
        args.consol_l2_bottom_ratio = 0.5


def _apply_memory_policy(args):
    """Map --memory_policy onto low-level flags.

    Policies:
      window                 — sliding-window baseline (no MoCE)
      heuristic_cr           — motion-score archive ranking
      learned_cr             — 5-D ChunkSelector (selector_all4.pt)
      world_state_cr         — default = v3 (v2 selector + consolidation full)
      world_state_cr_v3      — alias of world_state_cr
      world_state_cr_consol  — alias of world_state_cr
      world_state_cr_future  — alias of world_state_cr (back-compat)
      world_state_cr_v2      — frozen future-use selector only (former default)
      world_state_cr_v1      — frozen attention-mass selector (selector_ws_v1.pt)
    """
    policy = getattr(args, "memory_policy", None)
    if policy is None or policy == "legacy":
        return
    # Aliases of the default World-State CR (v3).
    if policy in (
        "world_state_cr_future",
        "world_state_cr_v3",
        "world_state_cr_consol",
    ):
        policy = "world_state_cr"
        args.memory_policy = policy
    if policy == "window":
        args.enable_motion_adaptive_kv_eviction = False
        args.selector = "heuristic"
        args.selector_ckpt = None
    elif policy == "heuristic_cr":
        args.enable_motion_adaptive_kv_eviction = True
        args.selector = "heuristic"
        args.selector_ckpt = None
    elif policy == "learned_cr":
        args.enable_motion_adaptive_kv_eviction = True
        args.selector = "learned"
        if not getattr(args, "selector_ckpt", None):
            args.selector_ckpt = resolve_selector_ckpt(
                None, schema="learned")
    elif policy == "world_state_cr":
        # Default v3: future-use v2 selector + Memory Consolidation + SWTP.
        _apply_ws_v2_selector(args)
        _apply_ws_v3_consolidation(args)
    elif policy == "world_state_cr_v2":
        # Frozen ablation: former default (selector only, no consol/SWTP).
        _apply_ws_v2_selector(args)
        args.consolidation = "off"
        args.enable_swtp = False
    elif policy == "world_state_cr_v1":
        # Frozen ablation: attention-mass oracle + world_state.v1 features.
        args.enable_motion_adaptive_kv_eviction = True
        args.selector = "learned"
        if not getattr(args, "selector_ckpt", None):
            args.selector_ckpt = resolve_selector_ckpt(
                None, schema="world_state")
        args.consolidation = "off"
        args.enable_swtp = False
    else:
        raise AssertionError(f"Unknown memory_policy: {policy}")


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    args.ckpt_dir = _resolve_input_path(args.ckpt_dir, should_exist=True)
    args.image = _resolve_input_path(args.image, should_exist=True)
    args.action_path = _resolve_input_path(args.action_path, should_exist=True)
    args.save_dir = _resolve_input_path(args.save_dir, should_exist=False)

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    # Resolve high-level memory policy before other MoCE/SWTP checks.
    _apply_memory_policy(args)

    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    if args.enable_motion_adaptive_kv_eviction:
        assert 0.0 <= args.ma_kv_keep_ratio <= 1.0, \
            "`ma_kv_keep_ratio` should be between 0 and 1."
        assert args.ma_kv_recent_window >= 0, \
            "`ma_kv_recent_window` should be non-negative."
        assert args.ma_kv_min_keep_chunks >= 0, \
            "`ma_kv_min_keep_chunks` should be non-negative."
        assert args.ma_kv_latent_rescue_thr >= 0.0, \
            "`ma_kv_latent_rescue_thr` should be non-negative."
    if args.enable_swtp:
        assert 0.0 < args.swtp_keep_ratio < 1.0, \
            "`swtp_keep_ratio` must be in (0, 1)."
        assert args.swtp_num_summary >= 0, \
            "`swtp_num_summary` must be non-negative."
        assert args.swtp_min_saliency_gini >= 0.0, \
            "`swtp_min_saliency_gini` must be non-negative."
        assert 0.0 < getattr(args, "swtp_energy_cover", 0.9) <= 1.0, \
            "`swtp_energy_cover` must be in (0, 1]."
    if getattr(args, "consolidation", "off") != "off":
        assert args.enable_motion_adaptive_kv_eviction, \
            "`--consolidation` requires `--enable_motion_adaptive_kv_eviction`."
        if args.consolidation == "full":
            assert args.enable_swtp, \
                "`--consolidation full` requires `--enable_swtp` for L1/L2 compression."
    if args.enable_motion_adaptive_kv_eviction and args.selector == "learned":
        policy = getattr(args, "memory_policy", None)
        if policy == "learned_cr":
            schema = "learned"
        elif policy == "world_state_cr_v1":
            schema = "world_state"
        else:
            # Default World-State CR (+ alias / legacy learned flags).
            schema = "world_state_future"
        args.selector_ckpt = resolve_selector_ckpt(args.selector_ckpt, schema=schema)
        args.selector_ckpt = _resolve_input_path(args.selector_ckpt, should_exist=True)
        assert os.path.isfile(args.selector_ckpt), \
            f"Chunk selector checkpoint not found: {args.selector_ckpt}"
    if args.selector == "learned" and not args.enable_motion_adaptive_kv_eviction:
        logging.info(
            "selector=learned ignored without --enable_motion_adaptive_kv_eviction; "
            "using heuristic archive ranking.")
        args.selector = "heuristic"
        args.selector_ckpt = None
    if args.collect_oracle:
        assert args.enable_motion_adaptive_kv_eviction, \
            "`--collect_oracle` requires `--enable_motion_adaptive_kv_eviction`."
        assert args.oracle_out is not None, \
            "`--collect_oracle` requires `--oracle_out`."
    # Parse comma-separated schedule indices into a list of ints.
    if isinstance(args.timesteps_index, str):
        args.timesteps_index = [int(x.strip()) for x in args.timesteps_index.split(",") if x.strip()]
    assert len(args.timesteps_index) >= 1, "`timesteps_index` must contain at least one index."
    assert all(0 <= i < 1000 for i in args.timesteps_index), \
        "`timesteps_index` values must be in [0, 1000)."


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help='The local size of kv cache during inference')
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help='The sink size of kv cache during inference')
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--memory_policy",
        type=str,
        default="legacy",
        choices=[
            "legacy", "window",
            "heuristic_cr", "learned_cr", "world_state_cr",
            "world_state_cr_v3",      # alias of world_state_cr
            "world_state_cr_consol",  # alias of world_state_cr
            "world_state_cr_future",  # alias of world_state_cr
            "world_state_cr_v2",      # frozen future-use selector only
            "world_state_cr_v1",      # frozen attention-mass ablation
        ],
        help="Chunk retention policy (preferred over raw flags). "
             "window=sliding baseline; heuristic_cr=motion archive; "
             "learned_cr=5-D ChunkSelector; "
             "world_state_cr=default v3 (future-use selector + consolidation "
             "full + SWTP); "
             "world_state_cr_v3/consol/future=aliases of world_state_cr; "
             "world_state_cr_v2=frozen selector-only (former default); "
             "world_state_cr_v1=frozen attention-mass selector "
             "(selector_ws_v1.pt); "
             "legacy=honour individual --enable_* / --selector flags only.")
    parser.add_argument(
        "--enable_motion_adaptive_kv_eviction",
        action="store_true",
        default=False,
        help="Enable motion-adaptive chunk-level KV eviction during fast inference.")
    parser.add_argument(
        "--ma_kv_recent_window",
        type=int,
        default=1,
        help="Number of recent chunks to always keep when Chunk Retention is enabled. "
             "Default 1 (tier ratio sink:recent:archive_min = 1:1:2 from RealCam-Vid ratio sweep).")
    parser.add_argument(
        "--ma_kv_keep_ratio",
        type=float,
        default=0.5,
        help="Keep ratio for archive chunks when motion-adaptive KV eviction is enabled.")
    parser.add_argument(
        "--ma_kv_min_keep_chunks",
        type=int,
        default=2,
        help="Minimum number of archive chunks to keep when Chunk Retention is enabled "
             "(tier ratio 1:1:2 with sink=1, recent=1).")
    parser.add_argument(
        "--ma_kv_latent_rescue",
        action="store_true",
        default=False,
        help="Use latent residual as a rescue signal for motion-adaptive KV eviction.")
    parser.add_argument(
        "--ma_kv_latent_rescue_thr",
        type=float,
        default=0.08,
        help="Threshold of latent residual rescue signal for motion-adaptive KV eviction.")
    parser.add_argument(
        "--enable_cond_hoist", action="store_true", default=False,
        help="Timestep-Invariant Condition Hoisting: exact elimination of "
             "chunk-invariant camera/I2V condition compute inside the "
             "denoising loop. Orthogonal to KV retention. No approx.")
    parser.add_argument(
        "--cond_hoist_global_cam",
        type=lambda x: str(x).lower() not in ("0", "false", "no"),
        default=True,
        help="Cache global camera embedding per chunk (default True).")
    parser.add_argument(
        "--cond_hoist_block_cam",
        type=lambda x: str(x).lower() not in ("0", "false", "no"),
        default=True,
        help="Cache per-block cam_scale/cam_shift (4 Linears) (default True).")
    parser.add_argument(
        "--cond_hoist_conv_split",
        type=lambda x: str(x).lower() not in ("0", "false", "no"),
        default=True,
        help="I2V patch Conv3d static/dynamic split (default True).")
    parser.add_argument(
        "--cond_hoist_profile", action="store_true", default=False,
        help="CUDA-event profile of hoistable vs residual kernels.")
    parser.add_argument(
        "--cond_hoist_verify", action="store_true", default=False,
        help="Record max_abs diffs when comparing hoist paths (debug).")
    parser.add_argument(
        "--enable_swtp",
        action="store_true",
        default=False,
        help="Enable Saliency-Weighted Token Pruning (SWTP). Token-level pruning within each "
             "archive chunk based on latent-residual saliency. Used with World-State CR for archive "
             "(lazy SWTP at archive promotion). Standalone SWTP (without MoCE) is also supported.")
    parser.add_argument(
        "--swtp_keep_ratio",
        type=float,
        default=0.5,
        help="Fraction of high-saliency tokens to keep per chunk (default 0.5).")
    parser.add_argument(
        "--swtp_num_summary",
        type=int,
        default=64,
        help="Number of mean-pooled summary tokens replacing the dropped tokens (default 64).")
    parser.add_argument(
        "--swtp_min_saliency_gini",
        type=float,
        default=0.20,
        help="If chunk saliency Gini is below this, use uniform lattice pooling "
             "instead of saliency top-k (default 0.20). Never leaves the chunk uncompressed.")
    parser.add_argument(
        "--swtp_energy_cover",
        type=float,
        default=0.9,
        help="Keep the smallest top-K (capped by swtp_keep_ratio) whose saliency "
             "mass covers this fraction (default 0.9).")
    parser.add_argument(
        "--consolidation",
        type=str,
        default="off",
        choices=["off", "ema", "full"],
        help="Memory Consolidation for World-State CR: off | ema (EMA+hysteresis ranking) | "
             "full (EMA + L1/L2/L3 tiered demotion). Not WorldKV bank retrieval.")
    parser.add_argument(
        "--consol_beta", type=float, default=0.7,
        help="EMA coefficient for consolidation utility (default 0.7).")
    parser.add_argument(
        "--consol_patience", type=int, default=2,
        help="Consecutive low-utility steps before L2 demotion (default 2).")
    parser.add_argument(
        "--consol_stabilize_thr", type=float, default=0.6,
        help="EMA threshold for marking a chunk stabilized (default 0.6).")
    parser.add_argument(
        "--consol_gist_tokens", type=int, default=64,
        help="Summary-token count for L2 gist demotion (default 64; default_loop sweep winner).")
    parser.add_argument(
        "--consol_gist_budget", type=int, default=512,
        help="Max total L2 gist tokens across archive (default 512).")
    parser.add_argument(
        "--consol_rank_alpha", type=float, default=0.5,
        help="Mix instantaneous selector score into consolidation ranking: "
             "rank = α·s + (1-α)·u_ema. 0=pure EMA, 1=pure instantaneous "
             "(default 0.5; default_loop sweep winner).")
    parser.add_argument(
        "--consol_l2_bottom_ratio", type=float, default=0.5,
        help="Fraction of kept archive chunks demoted to L2 gist (bottom by rank). "
             "0 disables rank-based L2; 0.5 demotes the weaker half (default).")
    parser.add_argument(
        "--archive_diversity_pool",
        type=int,
        default=0,
        help="Trajectory-diversity archive selection: size of motion-sorted candidate pool "
             "from which Farthest-Point Sampling picks archive chunks by camera forward direction. "
             "0 (default) disables (pure motion-only, current MoCE behavior). "
             "Set > archive budget (e.g., 4) to enable; larger = stronger diversity, weaker motion gatekeeper. "
             "Requires --enable_motion_adaptive_kv_eviction and --action_path with poses.npy.")
    parser.add_argument(
        "--selector",
        type=str,
        default="learned",
        choices=["heuristic", "learned"],
        help="Archive-ranking signal when MoCE is enabled. 'learned' = "
             "ChunkSelector MLP. 'heuristic' = motion score (Heuristic CR). "
             "Prefer --memory_policy {heuristic_cr,learned_cr,world_state_cr}. "
             "Requires --enable_motion_adaptive_kv_eviction.")
    parser.add_argument(
        "--selector_ckpt",
        type=str,
        default=None,
        help="Path to a trained ChunkSelector checkpoint (from train_selector.py). "
             f"Defaults to assets/selectors/{SELECTOR_CKPT_NAME} when --selector learned.")
    parser.add_argument(
        "--collect_oracle",
        action="store_true",
        default=False,
        help="Teacher pass: log per-chunk attention-mass labels for selector "
             "training. Run at a high --max_attention_size so few chunks are evicted (all "
             "candidates stay scorable). Writes to --oracle_out.")
    parser.add_argument(
        "--oracle_out",
        type=str,
        default=None,
        help="Where to save the oracle .pt (records + chunk metadata) when --collect_oracle.")
    parser.add_argument(
        "--oracle_probe_every",
        type=int,
        default=8,
        help="Compute the attention-mass oracle on every Nth DiT layer (default 8 -> layers 0,8,...).")
    parser.add_argument(
        "--oracle_label_type",
        type=str,
        default="attention_mass",
        choices=["attention_mass", "future_use_v1"],
        help="Oracle dump label. attention_mass=World-State CR default; "
             "future_use_v1=Future Coverage Oracle (P0; post-aggregates after mass collection).")
    parser.add_argument(
        "--oracle_future_horizon",
        type=int,
        default=8,
        help="Future Coverage Oracle horizon H in chunks (default 8).")
    parser.add_argument(
        "--oracle_future_gamma",
        type=float,
        default=0.9,
        help="Future Coverage Oracle discount gamma (default 0.9).")
    parser.add_argument(
        "--oracle_future_alpha",
        type=float,
        default=0.5,
        help="Future Coverage Oracle mix alpha: attn vs pose-reuse (default 0.5).")
    parser.add_argument(
        "--timesteps_index",
        type=str,
        default="0,179,358,679",
        help="Comma-separated indices into the 1000-step flow schedule used by Fast sampling "
             "(default 0,179,358,679 = 4-step distilled). Use e.g. 0,358 for 2-step ablation.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            input_prompt = args.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    logging.info("Creating WanI2VFast pipeline.")
    wan_i2v = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        enable_motion_adaptive_kv_eviction=args.enable_motion_adaptive_kv_eviction,
        ma_kv_recent_window=args.ma_kv_recent_window,
        ma_kv_keep_ratio=args.ma_kv_keep_ratio,
        ma_kv_min_keep_chunks=args.ma_kv_min_keep_chunks,
        ma_kv_latent_rescue=args.ma_kv_latent_rescue,
        ma_kv_latent_rescue_thr=args.ma_kv_latent_rescue_thr,
        enable_swtp=args.enable_swtp,
        swtp_keep_ratio=args.swtp_keep_ratio,
        swtp_num_summary=args.swtp_num_summary,
        swtp_min_saliency_gini=args.swtp_min_saliency_gini,
        swtp_energy_cover=getattr(args, "swtp_energy_cover", 0.9),
        archive_diversity_pool=args.archive_diversity_pool,
        selector=args.selector,
        selector_ckpt=args.selector_ckpt,
        collect_oracle=args.collect_oracle,
        oracle_probe_every=args.oracle_probe_every,
        oracle_out=args.oracle_out,
        oracle_label_type=getattr(args, "oracle_label_type", "attention_mass"),
        oracle_future_horizon=getattr(args, "oracle_future_horizon", 8),
        oracle_future_gamma=getattr(args, "oracle_future_gamma", 0.9),
        oracle_future_alpha=getattr(args, "oracle_future_alpha", 0.5),
        consolidation=getattr(args, "consolidation", "off"),
        consol_beta=getattr(args, "consol_beta", 0.7),
        consol_patience=getattr(args, "consol_patience", 2),
        consol_stabilize_thr=getattr(args, "consol_stabilize_thr", 0.6),
        consol_gist_tokens=getattr(args, "consol_gist_tokens", 64),
        consol_gist_budget=getattr(args, "consol_gist_budget", 512),
        consol_rank_alpha=getattr(args, "consol_rank_alpha", 0.5),
        consol_l2_bottom_ratio=getattr(args, "consol_l2_bottom_ratio", 0.5),
        enable_cond_hoist=bool(getattr(args, "enable_cond_hoist", False)),
        cond_hoist_global_cam=bool(getattr(args, "cond_hoist_global_cam", True)),
        cond_hoist_block_cam=bool(getattr(args, "cond_hoist_block_cam", True)),
        cond_hoist_conv_split=bool(getattr(args, "cond_hoist_conv_split", True)),
        cond_hoist_profile=bool(getattr(args, "cond_hoist_profile", False)),
        cond_hoist_verify=bool(getattr(args, "cond_hoist_verify", False)),
    )
    logging.info("Generating video ...")
    video = wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=3,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        timesteps_index=args.timesteps_index,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
