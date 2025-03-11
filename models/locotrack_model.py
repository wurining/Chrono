# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TAPIR models definition."""

import functools
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, Dict
import re

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import xformers.ops as xops
from xformers.ops import fmha

from models import utils


class FeatureGrids(NamedTuple):
  """Feature grids for a video, used to compute trajectories.

  These are per-frame outputs of the encoding resnet.

  Attributes:
    lowres: Low-resolution features, one for each resolution; 256 channels.
    hires: High-resolution features, one for each resolution; 64 channels.
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  highest: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]


class QueryFeatures(NamedTuple):
  """Query features used to compute trajectories.

  These are sampled from the query frames and are a full descriptor of the
  tracked points. They can be acquired from a query image and then reused in a
  separate video.

  Attributes:
    lowres: Low-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 256]
    hires: High-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 64]
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  highest: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]



class LocalAttentionLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            window_size: int = 13,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=window_size // 2,
            window_right=window_size // 2,
        )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = xops.memory_efficient_attention(q, k, v, attn_bias=self.bias)
        x = rearrange(x, 'b n h d -> b n (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalAdapter(nn.Module):
  """Temporal adapter for TAPIR."""

  def __init__(
    self, 
    input_channels, 
    intermed_channels, 
    num_layers,
    num_heads=4,
    window_size=13,
  ):
    super().__init__()
    self.norm = nn.GroupNorm(1, input_channels)
    self.conv1 = nn.Conv2d(input_channels, intermed_channels, kernel_size=3, padding=1)
    self.conv_pool = nn.Conv2d(intermed_channels, intermed_channels, kernel_size=4, stride=4)

    self.norm2 = nn.LayerNorm(intermed_channels)
    self.attn_temp = LocalAttentionLayer(
      dim=intermed_channels,
      num_heads=num_heads,
      window_size=window_size, 
    )
    self.conv_upsample = nn.ConvTranspose2d(intermed_channels, intermed_channels, kernel_size=4, stride=4)
    self.conv2 = nn.Conv2d(intermed_channels, input_channels, kernel_size=3, padding=1)

    nn.init.zeros_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)

  def forward(self, x):
    T = x.shape[-3]
    x = self.norm(x)
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    x = self.conv1(x) # Reduce channels
    x = F.relu(x)

    x = self.conv_pool(x) # Reduce spatial resolution
    x = F.relu(x)

    H, W = x.shape[-2:]
    x = rearrange(x, '(b t) c h w -> (b h w) t c', t=T)
    x_res = x
    x = self.norm2(x)
    x = self.attn_temp(x)
    x = x + x_res # Residual connection

    x = rearrange(x, '(b h w) t c -> (b t) c h w', t=T, h=H, w=W)
    x = self.conv_upsample(x)
    x = F.relu(x)

    x = self.conv2(x)
    x = rearrange(x, '(b t) c h w -> b c t h w', t=T)
    return x


def build_dino_adapter(input_channels, intermed_channels, num_layers):
  adapter = nn.ModuleList([
    TemporalAdapter(input_channels, intermed_channels, num_layers)
    for _ in range(num_layers)
  ])
  return adapter


class LocoTrack(nn.Module):
  """TAPIR model."""

  def __init__(
      self,
      bilinear_interp_with_depthwise_conv: bool = False,
      softmax_temperature: float = 20.0,
      initial_resolution: Tuple[int, int] = (256, 256),
      dino_size: str = 'small',
      dino_reg: bool = False,
      adapter_intermed_channels: int = 128,
  ):
    super().__init__()

    self.bilinear_interp_with_depthwise_conv = (
        bilinear_interp_with_depthwise_conv
    )

    self.softmax_temperature = softmax_temperature
    self.initial_resolution = tuple(initial_resolution)

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    input_channels = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1280,
    }

    backbone_arch = backbone_archs[dino_size]
    backbone_name = f"dinov2_{backbone_arch}{'_reg' if dino_reg else ''}"

    self.dino = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    self.adapter = build_dino_adapter(
      input_channels=input_channels[dino_size], 
      intermed_channels=adapter_intermed_channels,
      num_layers=len(self.dino.blocks) - 1,
    )

    self.img_mult = 64

    self.occ_linear = nn.Linear(3, 2)

  def forward(
      self,
      video: torch.Tensor,
      query_points: torch.Tensor,
      feature_grids: Optional[FeatureGrids] = None,
      is_training: bool = False,
      query_chunk_size: Optional[int] = 64,
      get_query_feats: bool = False,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> Mapping[str, torch.Tensor]:
    """Runs a forward pass of the model.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      query_points: The query points for which we compute tracks.
      is_training: Whether we are training.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      get_query_feats: Return query features for other losses like contrastive.
        Not supported in the current version.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A dict of outputs, including:
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          in raster coordinates
        expected_dist: uncertainty estimate logits, of shape
          [batch, num_queries, num_frames], where higher indicates more likely
          to be far from the correct answer.
    """
    if get_query_feats:
      raise ValueError('Get query feats not supported in TAPIR.')

    if feature_grids is None:
      feature_grids = self.forward_dino(video)

    query_features = self.get_query_features(
        video,
        is_training,
        query_points,
        feature_grids,
        refinement_resolutions,
    )

    trajectories = self.estimate_trajectories(
        video.shape[-3:-1],
        is_training,
        feature_grids,
        query_features,
        query_points,
        query_chunk_size,
    )

    out = dict(
        occlusion=trajectories['occlusion'],
        tracks=trajectories['tracks'],
        expected_dist=trajectories['expected_dist'],
        unrefined_occlusion=[trajectories['occlusion']],
        unrefined_tracks=[trajectories['tracks']],
        unrefined_expected_dist=[trajectories['expected_dist']],
    )

    return out

  def get_query_features(
      self,
      video: torch.Tensor,
      is_training: bool,
      query_points: torch.Tensor,
      feature_grids: Optional[FeatureGrids] = None,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> QueryFeatures:
    """Computes query features, which can be used for estimate_trajectories.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      query_points: The query points for which we compute tracks.
      feature_grids: If passed, we'll use these feature grids rather than
        computing new ones.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A QueryFeatures object which contains the required features for every
        required resolution.
    """

    feature_grid = feature_grids.lowres
    hires_feats = feature_grids.hires
    highest_feats = feature_grids.highest
    resize_im_shape = feature_grids.resolutions

    shape = video.shape
    # shape is [batch_size, time, height, width, channels]; conversion needs
    # [time, width, height]
    curr_resolution = (-1, -1)
    query_feats = []
    hires_query_feats = []
    highest_query_feats = []
    for i, resolution in enumerate(resize_im_shape):
      if utils.is_same_res(curr_resolution, resolution):
        query_feats.append(query_feats[-1])
        hires_query_feats.append(hires_query_feats[-1])
        highest_query_feats.append(highest_query_feats[-1])
        continue

      if len(feature_grid) > 0:
        position_in_grid = utils.convert_grid_coordinates(
            query_points,
            shape[1:4],
            feature_grid[i].shape[1:4],
            coordinate_format='tyx',
        )
        position_support = position_in_grid[..., None, :]
        position_support = rearrange(position_support, 'b n s c -> b (n s) c')
        interp = utils.map_coordinates_3d(
            feature_grid[i], position_support
        )

        query_feats.append(interp)

      if len(hires_feats) > 0:
        position_in_grid_hires = utils.convert_grid_coordinates(
            query_points,
            shape[1:4],
            hires_feats[i].shape[1:4],
            coordinate_format='tyx',
        )
        position_support_hires = position_in_grid_hires[..., None, :]
        position_support_hires = rearrange(position_support_hires, 'b n s c -> b (n s) c')
        hires_interp = utils.map_coordinates_3d(
            hires_feats[i], position_support_hires
        )

        hires_query_feats.append(hires_interp)

      if len(highest_feats) > 0:
        position_in_grid_highest = utils.convert_grid_coordinates(
            query_points,
            shape[1:4],
            highest_feats[i].shape[1:4],
            coordinate_format='tyx',
        )
        position_support_highest = position_in_grid_highest[..., None, :]
        position_support_highest = rearrange(position_support_highest, 'b n s c -> b (n s) c')
        highest_interp = utils.map_coordinates_3d(
            highest_feats[i], position_support_highest
        )

        highest_query_feats.append(highest_interp)

    return QueryFeatures(
        tuple(query_feats), tuple(hires_query_feats), tuple(highest_query_feats), tuple(resize_im_shape),
    )

  def estimate_trajectories(
      self,
      video_size: Tuple[int, int],
      is_training: bool,
      feature_grids: FeatureGrids,
      query_features: QueryFeatures,
      query_points_in_video: Optional[torch.Tensor],
      query_chunk_size: Optional[int] = None,
      causal_context: Optional[Dict[str, torch.Tensor]] = None,
      get_causal_context: bool = False,
  ) -> Mapping[str, Any]:
    """Estimates trajectories given features for a video and query features.

    Args:
      video_size: A 2-tuple containing the original [height, width] of the
        video.  Predictions will be scaled with respect to this resolution.
      is_training: Whether we are training.
      feature_grids: a FeatureGrids object computed for the given video.
      query_features: a QueryFeatures object computed for the query points.
      query_points_in_video: If provided, assume that the query points come from
        the same video as feature_grids, and therefore constrain the resulting
        trajectories to (approximately) pass through them.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      causal_context: If provided, a dict of causal context to use for
        refinement.
      get_causal_context: If True, return causal context in the output.

    Returns:
      A dict of outputs, including:
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          in raster coordinates
        expected_dist: uncertainty estimate logits, of shape
          [batch, num_queries, num_frames], where higher indicates more likely
          to be far from the correct answer.
    """
    del is_training

    def train2orig(x):
      return utils.convert_grid_coordinates(
          x,
          self.initial_resolution[::-1],
          video_size[::-1],
          coordinate_format='xy',
      )

    occ_iters = []
    pts_iters = []
    expd_iters = []

    infer = functools.partial(
        self.tracks_from_cost_volume,
        im_shp=feature_grids.lowres[0].shape[0:2]
        + self.initial_resolution
        + (3,),
    )

    num_queries = query_features.lowres[0].shape[1]

    for ch in range(0, num_queries, query_chunk_size):
      chunk = query_features.lowres[0][:, ch:ch + query_chunk_size]

      if query_points_in_video is not None:
        infer_query_points = query_points_in_video[
            :, ch : ch + query_chunk_size
        ]
        num_frames = feature_grids.lowres[0].shape[1]
        infer_query_points = utils.convert_grid_coordinates(
            infer_query_points,
            (num_frames,) + video_size,
            (num_frames,) + self.initial_resolution,
            coordinate_format='tyx',
        )
      else:
        infer_query_points = None

      points, occlusion, expected_dist = infer(
          chunk,
          feature_grids.lowres[0],
          infer_query_points,
      )
      pts_iters.append(train2orig(points))
      occ_iters.append(occlusion)
      expd_iters.append(expected_dist)

    occlusion = torch.cat(occ_iters, dim=1)
    points = torch.cat(pts_iters, dim=1)
    expd = torch.cat(expd_iters, dim=1)

    out = dict(
        occlusion=occlusion,
        tracks=points,
        expected_dist=expd,
    )
    return out
  
  def tracks_from_cost_volume(
      self,
      interp_feature: torch.Tensor,
      feature_grid: torch.Tensor,
      query_points: Optional[torch.Tensor],
      im_shp=None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts features into tracks by computing a cost volume.

    The computed cost volume will have shape
      [batch, num_queries, time, height, width], which can be very
      memory intensive.

    Args:
      interp_feature: A tensor of features for each query point, of shape
        [batch, num_queries, channels, heads].
      feature_grid: A tensor of features for the video, of shape [batch, time,
        height, width, channels, heads].
      query_points: When computing tracks, we assume these points are given as
        ground truth and we reproduce them exactly.  This is a set of points of
        shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
        raster coordinates.
      im_shp: The shape of the original image, i.e., [batch, num_frames, time,
        height, width, 3].

    Returns:
      A 2-tuple of the inferred points (of shape
        [batch, num_points, num_frames, 2] where each point is [x, y]) and
        inferred occlusion (of shape [batch, num_points, num_frames], where
        each is a logit where higher means occluded)
    """

    cost_volume = torch.einsum(
        'bnc,bthwc->tbnhw',
        interp_feature,
        feature_grid,
    )

    shape = cost_volume.shape
    batch_size, num_points = cost_volume.shape[1:3]

    pos = rearrange(cost_volume, 't b n h w -> b n t h w')
    pos_sm = pos.reshape(pos.size(0), pos.size(1), pos.size(2), -1)
    softmaxed = F.softmax(pos_sm * self.softmax_temperature, dim=-1)
    pos = softmaxed.view_as(pos)
  
    points = utils.heatmaps_to_points(pos, im_shp, query_points=query_points)

    occlusion = torch.cat(
      [
        torch.mean(cost_volume, dim=(-1, -2))[..., None],
        torch.amax(cost_volume, dim=(-1, -2))[..., None],
        torch.amin(cost_volume, dim=(-1, -2))[..., None],
      ], dim=-1
    )
    occlusion = self.occ_linear(occlusion.detach())
    expected_dist = rearrange(occlusion[..., 1:2], 't b n () -> b n t', t=shape[0])
    occlusion = rearrange(occlusion[..., 0:1], 't b n () -> b n t', t=shape[0])

    return points, occlusion, expected_dist

  def forward_dino(
    self,
    video: torch.Tensor,
    use_bfloat16: bool = True,
    img_mult: int = 64,
  ):
    """
    Args:
      video: torch.Tensor, shape [batch, time, height, width, 3], normalized to [-1, 1]
    """
    IMAGENET_DEFAULT_MEAN = torch.tensor((0.485, 0.456, 0.406), device=video.device)
    IMAGENET_DEFAULT_STD = torch.tensor((0.229, 0.224, 0.225), device=video.device)

    B, T = video.shape[:2]
    video = (video + 1) / 2
    video = (video - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
    video = rearrange(video, 'b t h w c -> (b t) c h w')

    img_mult = self.img_mult
    vid_size = (14 * img_mult, 14 * img_mult)
    H_f, W_f = img_mult, img_mult

    # add forward hooks for dino
    hooks = []
    for block_idx, block in enumerate(self.dino.blocks[:-1]):
      def hook_fn(module, input, output, block_index=block_idx):
        cls_token = output[:, :-H_f * W_f] # B, 4097, 768
        input_feat = rearrange(output[:, -H_f * W_f:], '(b t) (h w) c -> b c t h w', b=B, h=H_f, w=W_f)
        input_feat = self.adapter[block_index](input_feat)
        input_feat = rearrange(input_feat, 'b c t h w -> (b t) (h w) c')
        return output + torch.cat([torch.zeros_like(cls_token), input_feat], dim=1) # Residual connection

      hook = block.register_forward_hook(hook_fn)
      hooks.append(hook)

    video_resized = F.interpolate(video, size=vid_size, mode='bilinear', align_corners=False)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bfloat16): # Use bfloat16 for DINO
      video_feat = self.dino.get_intermediate_layers(video_resized, n=1, return_class_token=False)
    video_feat = rearrange(video_feat[-1], '(b t) (h w) c -> b t h w c', b=B, h=H_f, w=W_f)

    for hook in hooks:
      hook.remove()

    video_feat = video_feat / torch.sqrt(
      torch.maximum(
          torch.sum(torch.square(video_feat), axis=-1, keepdims=True),
          torch.tensor(1e-6, device=video_feat.device),
      )
    )
    
    return FeatureGrids(
      tuple([video_feat,]), 
      tuple([]),
      tuple([]),
      ((256, 256),)
    )

  def inference(
      self, 
      video : Union[np.ndarray, torch.Tensor],
      query_points : torch.Tensor, 
      query_chunk_size : int = 64,
      resolution : Tuple[int, int] = (256, 256),
      query_format : str = 'tyx',
    ) -> dict:
    """
    Run inference on LocoTrack model.
    Args:
      model: LocoTrack model
      video: np.ndarray or torch.Tensor, shape [batch, time, height, width, 3], normalized to [-1, 1]
        if np.ndarray, it will be converted to torch.Tensor
        if dtype is uint8, it will be converted to float32 and normalized to [-1, 1]
      query_points: torch.Tensor, shape [batch, num_points, 3]
      query_chunk_size: int, default 64
      resolution: Tuple[int, int], default (256, 256)
      query_format: str, default 'tyx', query points format
    Returns:
      dict with keys: 'tracks', 'occlusion'
    """
    assert video.shape[-1] == 3, f'video shape should be [batch, time, height, width, 3], got {video.shape}'
    device = next(self.parameters()).device

    # query_format is not tyx, then convert query_points to tyx
    query_shuffle_ind = [query_format.index(c) for c in 'tyx']
    query_points = query_points[..., query_shuffle_ind].to(device)
    
    if isinstance(video, np.ndarray):
      video = torch.from_numpy(video).to(device)
    
    if video.dtype == torch.uint8:
      video = video.float() / 255.0 * 2 - 1

    B, _, H, W, _ = video.shape
    if (H, W) != resolution:
      video = rearrange(video, 'b t h w c -> (b t) c h w')
      video = F.interpolate(video, resolution, mode='bilinear', align_corners=False)  
      video = rearrange(video, '(b t) c h w -> b t h w c', b=B)

      query_points = query_points.clone()
      query_points[..., 1] = query_points[..., 1] / H * resolution[0]
      query_points[..., 2] = query_points[..., 2] / W * resolution[1]

    out = self.forward(video, query_points, query_chunk_size=query_chunk_size)
    tracks, occlusion, expected_dist = out['tracks'], out['occlusion'], out['expected_dist']

    tracks = tracks * torch.tensor([W / resolution[1], H / resolution[0]], device=tracks.device)
    
    pred_occ = torch.sigmoid(occlusion)
    pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))
    pred_occ = pred_occ > 0.5  # threshold

    return {
      'tracks': tracks,
      'occlusion': pred_occ,
    }
    
