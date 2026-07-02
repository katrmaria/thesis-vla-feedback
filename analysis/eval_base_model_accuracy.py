"""
eval_base_model_accuracy.py

Measures the base OpenVLA model's teacher-forced action token accuracy and loss
on the LIBERO spatial training data — NO feedback module, NO two-pass, just the
vanilla model doing a single forward pass.

This gives us the baseline to compare against our feedback variants.

Usage:
    python eval_base_model_accuracy.py \
        --data-path /path/to/libero_spatial.npz \
        --model-name openvla/openvla-7b-finetuned-libero-spatial \
        --half-precision
"""

import os
import sys
import argparse
import logging
import importlib.util
from dataclasses import dataclass
from typing import Callable, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# ---- OpenVLA imports (direct file import to avoid draccus dependency) ----
OPENVLA_REPO = os.environ.get("OPENVLA_REPO", "openvla")

def _import_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_at_mod = _import_file("action_tokenizer", os.path.join(OPENVLA_REPO, "prismatic/vla/action_tokenizer.py"))
ActionTokenizer = _at_mod.ActionTokenizer

_bp_mod = _import_file("base_prompter", os.path.join(OPENVLA_REPO, "prismatic/models/backbones/llm/prompting/base_prompter.py"))
PurePromptBuilder = _bp_mod.PurePromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ---- Action normalizer (mirrors RLDS BOUNDS_Q99 normalization used during training) ----
def make_action_normalizer(norm_stats: dict):
    """
    Returns a function that normalizes a raw action the same way the RLDS pipeline does:
        normalized = clip(2 * (raw - q01) / (q99 - q01) - 1, -1, 1)
    norm_stats must have keys 'q01', 'q99', and optionally 'mask'.
    """
    q01 = np.array(norm_stats["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["q99"], dtype=np.float32)
    mask = np.array(norm_stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)

    def normalize(action: np.ndarray) -> np.ndarray:
        action = action.astype(np.float32).copy()
        # Fix LIBERO gripper: raw {-1=open, +1=close} -> {+1=open, 0=close}
        action[-1] = 1.0 - np.clip(action[-1], 0.0, 1.0)
        normalized = np.where(mask, 2.0 * (action - q01) / np.where(q99 - q01 > 1e-8, q99 - q01, 1e-8) - 1.0, action)
        return np.clip(normalized, -1.0, 1.0)

    return normalize


# ---- Batch Transform (same as FeedbackBatchTransform but only produces one pass) ----
@dataclass
class SinglePassBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: object
    image_transform: Callable
    prompt_builder_fn: Type
    action_normalizer: Callable  # BOUNDS_Q99 normalization using model's norm_stats
    predict_stop_token: bool = True

    def __call__(self, step):
        img = Image.fromarray(step["image"]).convert("RGB")
        action = self.action_normalizer(step["action"])  # normalize before tokenizing
        lang = step["task"].lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        labels[:-(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


# ---- Dataset (same as LIBERORLDSDataset but simpler) ----
class LIBERODataset(IterableDataset):
    def __init__(self, all_episodes, batch_transform, split="train", train_ratio=0.8, seed=42):
        self.batch_transform = batch_transform
        self.split = split
        self.seed = seed

        task_to_episodes = {}
        for episode_idx, episode in enumerate(all_episodes):
            task = episode[0]["task"]
            if task not in task_to_episodes:
                task_to_episodes[task] = []
            task_to_episodes[task].append(episode_idx)

        selected_episodes = set()
        rng = np.random.RandomState(seed)
        for task, episodes in task_to_episodes.items():
            n_train = int(len(episodes) * train_ratio)
            shuffled_episodes = episodes.copy()
            rng.shuffle(shuffled_episodes)
            if split == "train":
                selected_episodes.update(shuffled_episodes[:n_train])
            else:
                selected_episodes.update(shuffled_episodes[n_train:])

        self.all_episodes = [all_episodes[i] for i in sorted(selected_episodes)]
        self._num_steps = sum(len(ep) for ep in self.all_episodes)

    def __len__(self):
        return self._num_steps

    def __iter__(self):
        for episode in self.all_episodes:
            for step in episode:
                yield self.batch_transform(step)


# ---- Collator ----
@dataclass
class PaddedCollator:
    model_max_length: int
    pad_token_id: int

    def __call__(self, instances):
        input_ids = [i['input_ids'] for i in instances]
        labels = [i['labels'] for i in instances]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids_padded = input_ids_padded[:, :self.model_max_length]
        labels_padded = labels_padded[:, :self.model_max_length]
        attn_mask = input_ids_padded.ne(self.pad_token_id)
        pixel_values = torch.stack([i['pixel_values'] for i in instances])

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attn_mask,
            'labels': labels_padded,
            'pixel_values': pixel_values,
        }


def load_libero_data(data_path, max_episodes=None):
    data = np.load(data_path, allow_pickle=True)
    images = data["images"]
    actions = data["actions"]
    tasks = data["tasks"]
    boundaries = data["episode_boundaries"]

    episodes = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        episode = []
        for j in range(start, end):
            episode.append({
                "image": images[j],
                "action": actions[j],
                "task": str(tasks[j]),
            })
        episodes.append(episode)
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Baseline OpenVLA accuracy on LIBERO data")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--unnorm-key", type=str, default=None,
                        help="Key into model.norm_stats for BOUNDS_Q99 action normalization "
                             "(e.g. 'libero_spatial_no_noops'). Auto-detected if not set.")
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Max batches to evaluate")
    args = parser.parse_args()

    # Load model
    torch_dtype = torch.bfloat16 if args.half_precision else torch.float32
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()

    # Resolve action normalization key and build normalizer (BOUNDS_Q99, same as training)
    norm_stats = model.norm_stats
    unnorm_key = args.unnorm_key
    if unnorm_key is None:
        # Auto-detect: prefer task-suite-named key, fall back to first available
        candidate = args.model_name.split("/")[-1].replace("openvla-7b-finetuned-", "")
        if candidate in norm_stats:
            unnorm_key = candidate
        elif f"{candidate}_no_noops" in norm_stats:
            unnorm_key = f"{candidate}_no_noops"
        else:
            unnorm_key = next(iter(norm_stats))
    assert unnorm_key in norm_stats, (
        f"unnorm_key '{unnorm_key}' not in model.norm_stats. Available: {list(norm_stats.keys())}"
    )
    logger.info(f"Using action norm_stats key: '{unnorm_key}'")
    action_normalizer = make_action_normalizer(norm_stats[unnorm_key]["action"])

    # Build batch transform (same tokenization as feedback training, now with correct normalization)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = SinglePassBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        action_normalizer=action_normalizer,
        predict_stop_token=True,
    )

    # Load data
    logger.info(f"Loading data: {args.data_path}")
    all_episodes = load_libero_data(args.data_path, max_episodes=args.max_episodes)
    total_steps = sum(len(ep) for ep in all_episodes)
    logger.info(f"Loaded {len(all_episodes)} episodes, {total_steps} total steps")

    # Create datasets
    train_dataset = LIBERODataset(all_episodes, batch_transform, split="train")
    val_dataset = LIBERODataset(all_episodes, batch_transform, split="val")

    collator = PaddedCollator(
        model_max_length=model.config.text_config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator)

    num_patches = model.vision_backbone.featurizer.patch_embed.num_patches  # 256
    action_token_begin_idx = action_tokenizer.action_token_begin_idx

    # Evaluate
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        total_correct = 0
        total_action_tokens = 0
        n_batches = 0
        all_l1 = []

        logger.info(f"Evaluating on {split_name} split...")
        with torch.no_grad():
            for batch in tqdm(loader, desc=split_name):
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Cast pixel_values to model dtype (bfloat16 if --half-precision)
                if batch["pixel_values"].dtype != torch_dtype:
                    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch_dtype)

                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    return_dict=True,
                )

                total_loss += output.loss.item()
                n_batches += 1

                # Action token accuracy (same logic as training)
                if output.logits is not None:
                    labels = batch["labels"]
                    # Logits are shifted: logit at position i predicts token at position i+1
                    # For action accuracy, extract logits at positions corresponding to action tokens
                    action_logits = output.logits[:, num_patches:-1]
                    action_preds = action_logits.argmax(dim=2)
                    action_gt = labels[:, 1:].to(action_preds.device)

                    # Only count action tokens (those above action_token_begin_idx)
                    mask = action_gt > action_token_begin_idx
                    if mask.any():
                        correct = (action_preds == action_gt) & mask
                        total_correct += correct.sum().item()
                        total_action_tokens += mask.sum().item()

                        # L1 loss on continuous actions
                        pred_tokens = action_preds[mask].cpu().numpy()
                        gt_tokens = action_gt[mask].cpu().numpy()
                        pred_actions = action_tokenizer.decode_token_ids_to_actions(pred_tokens)
                        gt_actions = action_tokenizer.decode_token_ids_to_actions(gt_tokens)
                        l1 = np.abs(np.array(pred_actions) - np.array(gt_actions)).mean()
                        all_l1.append(l1)

                if args.max_steps and n_batches >= args.max_steps:
                    break

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_correct / max(total_action_tokens, 1)
        avg_l1 = np.mean(all_l1) if all_l1 else 0.0

        print(f"\n{'='*60}")
        print(f"BASE MODEL — {split_name.upper()} SPLIT")
        print(f"  Loss:              {avg_loss:.4f}")
        print(f"  Action token acc:  {avg_acc:.4f} ({avg_acc*100:.1f}%)")
        print(f"  Action L1 loss:    {avg_l1:.4f}")
        print(f"  Batches evaluated: {n_batches}")
        print(f"  Action tokens:     {total_action_tokens}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
