// Architecture descriptors for the long-FFT backend, following the cuButterfly
// layer separation: G (graph) / A (mapping) / P (processing unit) /
// L (layout) / F (kernel realization, see src/ascendc/) / Q (selection, step 6) /
// H (hardware profile). Plan construction queries query_lowering() BEFORE any
// allocation or launch; an unsupported tuple returns an explicit reason and
// never silently swaps the unit or mapping. All numeric H values mirror
// config/ascend910_93_profile.json (tests/test_descriptors.py keeps them in
// sync); nothing here is imported from a foreign microarchitecture.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
// UB 资源公式与内核同源（PR #2 阶段 3）：AB_LT_* / AB_*_UB_BYTES 由
// src/ascendc/fft_long.cpp 与本文件共同 include，公式只维护一处。
#include "butterfly/long_fft_ub.h"
#include "butterfly/fft_k.hpp"

namespace butterfly {

// ---- G: transform graph / numerical semantics (input-independent) ----
enum class Precision { FP32, FP16 };
// C2C is the only mapped direction today; R2C/C2R envelopes stay enforced by
// the fft_check preflight until a real-direction mapping exists (the query
// rejects them explicitly rather than borrowing the C2C mapping).
enum class Direction { C2C_FWD, R2C, C2R };

struct TransformSpec {
  uint32_t n = 0;          // transform length, power of two
  uint32_t batch = 0;      // batch: extra data-dimension work, not a mapping dimension
  Precision precision = Precision::FP32;
  Direction direction = Direction::C2C_FWD;
};

// ---- L: layouts ----
enum class Layout { InterleavedComplex, TwiddledStage, NaturalOrder, HalfSpectrumReal };

inline const char* layout_name(Layout l) {
  switch(l){
    case Layout::InterleavedComplex: return "interleaved_complex";
    case Layout::TwiddledStage:      return "twiddled_stage";
    case Layout::NaturalOrder:       return "natural_order";
    case Layout::HalfSpectrumReal:   return "half_spectrum_real";
  }
  return "unknown";
}
inline const char* precision_name(Precision p){
  return p==Precision::FP32 ? "fp32" : "fp16";
}

// ---- A: architecture mapping (the four unfoldings + roles/residence) ----
// Ud: independent data tiles/AIVs active concurrently.
// Td: successive tiles reused by the same worker while state stays live.
// Us: stage services/roles replicated concurrently.
// Ts: dependent stage groups reused by one stage service in place.
enum class Residence { None, CellResident, BlockResident, WorkerResident };
// Where the inter-stage boundary is materialized when it leaves on-chip storage.
enum class BoundaryHome { HostMemory, DeviceGM, OnChip };

inline const char* residence_name(Residence r){
  switch(r){
    case Residence::None:           return "none";
    case Residence::CellResident:   return "cell-resident";
    case Residence::BlockResident:  return "block-resident";
    case Residence::WorkerResident: return "worker-resident";
  }
  return "unknown";
}

struct ArchitectureMapping {
  int stage_partition = 0;          // number of stage groups (segments in the chain)
  std::vector<uint32_t> stage_lengths;  // point length per stage group; stays a parameter
  int stage_space = 0;              // Us
  int stage_time = 0;               // Ts
  int data_space = 0;               // Ud
  int data_time = 0;                // Td
  std::vector<int> role_stages;     // role slot -> stage index it serves
  Residence residence = Residence::None;
  Layout boundary_layout = Layout::TwiddledStage;
  BoundaryHome boundary_home = BoundaryHome::HostMemory;
  int pipeline_buffers = 1;
  // Row-FFT resource overrides (0 = derive): mirror the AB_FOLD_D / AB_PLANE_K
  // test hooks so the descriptor gate and the launch resolve the SAME (D,K)
  // for every stage (R0.1).  fft_check fills these from the environment
  // before query_lowering and reuses the same parse in prepPass.
  uint32_t row_fft_fold_d  = 0;
  uint32_t row_fft_plane_k = 0;
};

// ---- P: replaceable processing unit capability ----
enum class SyncScope { None, AIVIntraCore, CoreIntraGroup, CoreInterGroup };

inline const char* sync_scope_name(SyncScope s){
  switch(s){
    case SyncScope::None:           return "none";
    case SyncScope::AIVIntraCore:   return "aiv_intra_core";
    case SyncScope::CoreIntraGroup: return "core_intra_group";
    case SyncScope::CoreInterGroup: return "core_inter_group";
  }
  return "unknown";
}

struct ProcessingUnitCapability {
  std::string core;                 // unit id, e.g. "kfft_fwd_local_fft"
  Precision precision = Precision::FP32;
  int min_local_log_n = 0;          // smallest supported local transform 2^k
  int max_local_log_n = 0;          // largest supported local transform 2^k
  size_t ub_bytes = 0;              // UB bytes the unit claims; must fit H.ub_bytes_per_core
  SyncScope synchronization_scope = SyncScope::None;
  std::vector<Layout> supported_layouts;
  bool supports_multi_role = false;
};

// First unit: the current AscendC row FFT (src/ascendc/fft_radix2.cpp, kfft_fwd).
// Its [64,4096] envelope is the measured UB budget; it owns no partitioning,
// role scheduling or boundary policy (see addendum §5).
inline ProcessingUnitCapability local_fft_capability(){
  ProcessingUnitCapability u;
  u.core = "kfft_fwd_local_fft";
  u.precision = Precision::FP32;
  u.min_local_log_n = 6;            // 64
  u.max_local_log_n = 12;           // 4096
  u.ub_bytes = 196608;              // claims the full per-AIV UB; per-length fit via the envelope
  u.synchronization_scope = SyncScope::AIVIntraCore;
  u.supported_layouts = {Layout::InterleavedComplex, Layout::TwiddledStage,
                         Layout::NaturalOrder};
  u.supports_multi_role = false;
  return u;
}

// ---- H: Ascend hardware profile (mirrors config/ascend910_93_profile.json) ----
struct HardwareProfile {
  const char* profile_id = "";
  const char* soc = "";
  int aicore_num = 0;
  int vector_core_num = 0;
  size_t l2_bytes = 0;
  size_t global_mem_total_bytes = 0;
  size_t ub_bytes_per_core = 0;
  int vec_lane_fp32 = 0;
};

inline const HardwareProfile& builtin_hardware_profile(){
  static const HardwareProfile p = {
    "ascend910_93", "Ascend910_9382",
    24,             // aicore_num
    48,             // vector_core_num (AIV)
    201326592ull,   // l2_bytes
    65787658240ull, // global_mem_total_bytes
    196608ull,      // ub_bytes_per_core (192 KiB)
    128             // vec_lane_fp32
  };
  return p;
}

// ---- Q / lowering contract ----
// One record per real kernel launch of the executable lowering; every count in
// LoweringResult is derived from this manifest so metadata and runtime cannot
// drift apart (PR #2 stage 2).
enum class LaunchKind { RowFFT, TransposeIn, Twiddle, TransposeBoundary, TransposeOut };

inline const char* launch_kind_name(LaunchKind k){
  switch(k){
    case LaunchKind::RowFFT:            return "row_fft";
    case LaunchKind::TransposeIn:       return "transpose_in";
    case LaunchKind::Twiddle:           return "twiddle";
    case LaunchKind::TransposeBoundary: return "transpose_boundary";
    case LaunchKind::TransposeOut:      return "transpose_out";
  }
  return "unknown";
}

struct LaunchRecord {
  LaunchKind kind;
  int stage;          // stage index of the transform this launch belongs to
  int boundary_edge;  // 1 if this launch consumes a materialized inter-stage
                      // boundary (GM or host); the edge count sums these
};

// ---- F: per-kernel resource records (plan-level UB model, PR #2 stage 3) ----
// All lowered kernels run on AIV vector cores with intra-core synchronization;
// the UB peak formulas come from the shared header so they track the kernels.
enum class CoreKind { AIVVectorCore };

inline const char* core_kind_name(CoreKind k){
  switch(k){
    case CoreKind::AIVVectorCore: return "aiv_vector_core";
  }
  return "unknown";
}

struct KernelResource {
  const char* kernel;     // kernel entry point, e.g. "kfft_fwd"
  CoreKind core;          // block/core class the launch occupies
  size_t ub_bytes;        // peak UB footprint for the mapped shape
  SyncScope sync_scope;   // synchronization range inside one launch
  const char* constraint; // shape envelope this peak assumes
};

// Single source for per-stage (D,K): the descriptor gate and the runtime
// launch must resolve identical values (R0.1).  launch_rows is the row count
// the launch actually issues (long chain: (n/len)*batch; short path: batch);
// overrides come from ArchitectureMapping (filled from AB_FOLD_D/AB_PLANE_K).
struct RowFftPlan {
  uint32_t fold_d;   // batch fold coefficient packed into arg byte 0
  uint32_t plane_k;  // plane factor packed into arg byte 1
};
inline RowFftPlan resolve_row_fft(uint32_t len, uint32_t launch_rows,
                                  uint32_t fold_d_override,
                                  uint32_t plane_k_override){
  RowFftPlan p;
  p.fold_d  = fold_d_override ? fold_d_override
                              : bfly::foldDFor(len, launch_rows, 48u);
  if(p.fold_d < 1u) p.fold_d = 1u;
  p.plane_k = plane_k_override ? plane_k_override : bfly::planeKFor(len);
  // K 合法性（R0.1）：rows = n/K >= 8 是矢量算子 32B 对齐的硬约束
  // （否则 AIV 抛 507035）。AB_PLANE_K 指定非法 K 时两处同时回落到规则值，
  // descriptor 与 launch 依旧同源 —— 与 kernel 的 arg 打包保持一致。
  if(p.plane_k < 8u || p.plane_k * 8u > len) p.plane_k = bfly::planeKFor(len);
  return p;
}

inline KernelResource row_fft_resource(uint32_t len, uint32_t launch_rows,
                                       uint32_t fold_d_override,
                                       uint32_t plane_k_override){
  const RowFftPlan p = resolve_row_fft(len, launch_rows,
                                       fold_d_override, plane_k_override);
  return KernelResource{"kfft_fwd", CoreKind::AIVVectorCore,
                        (size_t)AB_ROW_FFT_UB_BYTES(len, p.fold_d, p.plane_k),
                        SyncScope::AIVIntraCore,
                        "row length power of two in [64,4096]; "
                        "UB(n,D,K) with the launch's fold D and plane K"};
}
inline KernelResource transpose_resource(){
  return KernelResource{"kfft_lt_tr", CoreKind::AIVVectorCore,
                        (size_t)AB_TRANSPOSE_UB_BYTES,
                        SyncScope::AIVIntraCore,
                        "fixed LT_H x LT_W tile, shape independent"};
}
inline KernelResource twiddle_resource(uint32_t len){
  return KernelResource{"kfft_lt_tw", CoreKind::AIVVectorCore,
                        (size_t)AB_TWIDDLE_UB_BYTES(len),
                        SyncScope::AIVIntraCore,
                        "half-row chunking over one stage length"};
}

// Plan-level UB for serial launches is the PEAK of the per-kernel records,
// never a sum: the runtime issues the chain one launch at a time on one
// stream. Concurrent/resident roles would sum per live role -- no such
// lowering exists yet (see the not-lowered checks below).
inline KernelResource plan_peak(const std::vector<KernelResource>& ks){
  KernelResource best = ks.empty()
      ? KernelResource{"<empty>", CoreKind::AIVVectorCore, 0,
                       SyncScope::None, "n/a"}
      : ks.front();
  for(const auto& k : ks) if(k.ub_bytes > best.ub_bytes) best = k;
  return best;
}

struct LoweringResult {
  // Two-level judgment (PR #2 stage 2):
  //   abstract_feasible = the (spec, mapping, unit, hardware) tuple is
  //     resource- and legality-feasible;
  //   supported = the repository contains an executable lowering for it
  //     (today: the serial two-segment chain on HostMemory/DeviceGM). The
  //     selection layer may only pick lowerings where supported == true.
  bool abstract_feasible = false;
  bool supported = false;
  std::string reason;                       // explicit when supported == false
                                            // ("abstract-feasible-but-not-lowered: ..."
                                            //  when abstract_feasible == true)
  std::vector<LaunchRecord> launch_manifest;  // real launches when supported
  int visible_launches = 0;                 // == launch_manifest.size()
  int materialized_gm_boundaries = 0;       // stage edges that touch GM/host memory
                                            // (sum of boundary_edge flags)
  size_t plan_ub_bytes = 0;                 // serial plan UB peak over the manifest
  Residence resident_subgraph = Residence::None;
  bool whole_transform_on_chip = false;
  bool host_assisted = false;               // boundary crosses host memory today
};

// Selection contract: only executable lowerings are selectable.
inline bool selectable(const LoweringResult& r){ return r.supported; }

// Legality of (mapping, unit, hardware) for one transform spec. Pure function:
// no allocation, no launch, no silent core/mapping substitution.
inline LoweringResult query_lowering(const TransformSpec& spec,
                                     const ArchitectureMapping& m,
                                     const ProcessingUnitCapability& unit,
                                     const HardwareProfile& hw){
  LoweringResult r;
  auto reject = [&r](std::string why){ r.supported=false; r.reason=std::move(why); return r; };

  if(spec.precision != unit.precision)
    return reject("precision mismatch: transform is " + std::string(precision_name(spec.precision)) +
                  ", unit " + unit.core + " provides " + precision_name(unit.precision));
  if(spec.direction != Direction::C2C_FWD)
    return reject(std::string("direction not supported by mapping for unit ") + unit.core);
  if(spec.n < 64 || (spec.n & (spec.n-1)))
    return reject("transform length must be a power of two >= 64");

  if(m.stage_partition <= 0 || m.stage_lengths.empty())
    return reject("mapping declares no stage partition");
  if((int)m.stage_lengths.size() != m.stage_partition)
    return reject("stage_partition inconsistent with stage lengths");
  uint64_t product = 1;
  for(uint32_t len : m.stage_lengths) product *= len;
  if(product != spec.n)
    return reject("stage lengths product " + std::to_string(product) +
                  " does not match transform length " + std::to_string(spec.n));
  for(uint32_t len : m.stage_lengths){
    if(len < 64 || (len & (len-1)))
      return reject("stage length must be a power of two >= 64");
    int lg = 0; while((1u<<lg) < len) lg++;
    if(lg < unit.min_local_log_n || lg > unit.max_local_log_n)
      return reject("stage length " + std::to_string(len) + " (log2 " + std::to_string(lg) +
                    ") outside unit " + unit.core + " range [" +
                    std::to_string(unit.min_local_log_n) + "," +
                    std::to_string(unit.max_local_log_n) + "]");
  }

  if(m.stage_space < 1) return reject("Us (stage services) must be >= 1");
  if(m.stage_time  < 1) return reject("Ts (stage reuse) must be >= 1");
  if(m.data_time   < 1) return reject("Td (tile reuse) must be >= 1");
  if(m.data_space  < 1) return reject("Ud (concurrent data tiles) must be >= 1");
  if(m.data_space > hw.vector_core_num)
    return reject("Ud " + std::to_string(m.data_space) + " exceeds AIV count " +
                  std::to_string(hw.vector_core_num));
  if(m.stage_space > hw.vector_core_num)
    return reject("Us " + std::to_string(m.stage_space) + " exceeds AIV count " +
                  std::to_string(hw.vector_core_num));
  if(m.pipeline_buffers < 1) return reject("pipeline_buffers must be >= 1");

  for(int stage : m.role_stages)
    if(stage < 0 || stage >= m.stage_partition)
      return reject("role allocation references stage " + std::to_string(stage) +
                    " of " + std::to_string(m.stage_partition));

  bool layout_ok = false;
  for(Layout l : unit.supported_layouts) if(l == m.boundary_layout) layout_ok = true;
  if(!layout_ok)
    return reject(std::string("boundary layout ") + layout_name(m.boundary_layout) +
                  " not supported by unit " + unit.core);

  // UB is checked per manifest AFTER the not-lowered gate below: each
  // physical kernel contributes its own peak (PR #2 stage 3), so the plan
  // model cannot accept a DeviceGM chain on a UB too small for transpose.

  if(m.residence == Residence::BlockResident && !unit.supports_multi_role)
    return reject("block residence requires a multi-role unit: " + unit.core);
  if(m.residence == Residence::WorkerResident && !unit.supports_multi_role)
    return reject("worker residence requires a multi-role unit: " + unit.core);
  if((m.residence == Residence::BlockResident || m.residence == Residence::WorkerResident) &&
     unit.synchronization_scope == SyncScope::None)
    return reject(std::string("residence ") + residence_name(m.residence) +
                  " requires a synchronization scope on unit " + unit.core);
  if(m.boundary_home == BoundaryHome::OnChip && m.residence != Residence::BlockResident)
    return reject("on-chip boundary requires block residence under one synchronized owner");

  // GM footprint of one execute (3 complex tensors + margin, same accounting as G0).
  const uint64_t gm = 24ull * (uint64_t)spec.n * (uint64_t)spec.batch;
  if(gm > (40ull<<30))
    return reject("gm footprint " + std::to_string(gm) + " bytes exceeds 40 GiB guard");
  if(gm > (uint64_t)hw.global_mem_total_bytes)
    return reject("gm footprint exceeds device global memory");

  // --- level 1 passed: the tuple is resource- and legality-feasible ---
  r.abstract_feasible = true;

  // --- level 2: does the repository hold an executable lowering? ---
  // The runtime implements exactly today's serial chain: one stage service,
  // sequential reuse of the two stage groups, one concurrent tile grid sized
  // to the AIV count, single-buffer staging, 1:1 role slots, no residency,
  // twiddled-stage boundary in either GM home. Anything else is abstractly
  // feasible at best and must not be reported executable.
  std::string not_lowered;
  if(m.stage_partition != 2)
    not_lowered = "runtime implements the two-segment chain only (stage_partition=2)";
  else if(m.stage_space != 1)
    not_lowered = "Us " + std::to_string(m.stage_space) +
                  ": only a single stage service is lowered";
  else if(m.stage_time != m.stage_partition)
    not_lowered = "Ts " + std::to_string(m.stage_time) +
                  ": only sequential reuse across the stage groups is lowered";
  else if(m.data_space != (int)hw.vector_core_num)
    not_lowered = "Ud " + std::to_string(m.data_space) +
                  ": only one concurrent tile grid sized to the AIV count is lowered";
  else if(m.data_time != 1)
    not_lowered = "Td " + std::to_string(m.data_time) +
                  ": no tile-reuse lowering exists";
  else if(m.pipeline_buffers != 1)
    not_lowered = "pipeline_buffers " + std::to_string(m.pipeline_buffers) +
                  ": only single-buffer staging is lowered";
  else if(m.role_stages != std::vector<int>{0, 1})
    not_lowered = "role_stages must map the two stages 1:1";
  else if(m.residence != Residence::None)
    not_lowered = std::string("residence ") + residence_name(m.residence) +
                  ": no resident-subgraph lowering exists yet";
  else if(m.boundary_layout != Layout::TwiddledStage)
    not_lowered = std::string("boundary layout ") + layout_name(m.boundary_layout) +
                  ": only the twiddled-stage boundary is lowered";
  else if(m.boundary_home != BoundaryHome::HostMemory &&
          m.boundary_home != BoundaryHome::DeviceGM)
    not_lowered = "boundary home OnChip has no executable lowering";
  if(!not_lowered.empty()){
    r.supported = false;
    r.reason = "abstract-feasible-but-not-lowered: " + not_lowered;
    return r;
  }

  // --- executable: derive the manifest, then derive every count from it ---
  std::vector<LaunchRecord> man;
  if(m.boundary_home == BoundaryHome::DeviceGM){
    // addendum §3 device chain: transpose-in -> FFT-1 -> twiddle ->
    // transpose-boundary -> FFT-2 -> transpose-out (all on device, no host hop).
    man = {
      {LaunchKind::TransposeIn,       0, 0},
      {LaunchKind::RowFFT,            0, 0},
      {LaunchKind::Twiddle,           0, 0},
      {LaunchKind::TransposeBoundary, 1, 1},  // consumes the GM stage edge
      {LaunchKind::RowFFT,            1, 0},
      {LaunchKind::TransposeOut,      1, 0},
    };
  }else{ // HostMemory: host does transpose/twiddle/reorder between the two FFTs
    man = {
      {LaunchKind::RowFFT, 0, 0},
      {LaunchKind::RowFFT, 1, 1},              // pass2 consumes the host edge
    };
  }
  r.launch_manifest = std::move(man);
  r.visible_launches = (int)r.launch_manifest.size();

  // Per-kernel UB fit, checked in launch order (transpose launches come
  // first in the device manifest, so a UB below the transpose peak is
  // rejected naming kfft_lt_tr; row FFT and twiddle follow).
  const size_t ub_have = unit.ub_bytes < hw.ub_bytes_per_core ? unit.ub_bytes
                                                             : hw.ub_bytes_per_core;
  std::vector<KernelResource> used;
  for(const auto& rec : r.launch_manifest){
    KernelResource kr{"", CoreKind::AIVVectorCore, 0, SyncScope::None, ""};
    switch(rec.kind){
      case LaunchKind::RowFFT: {
        // Same (n,rows,D,K) the launch resolves in prepPass: rows is the
        // launch's row count, i.e. (n/len)*batch for the two-segment chain.
        const uint32_t len   = m.stage_lengths[rec.stage];
        const uint32_t rows  = (spec.n / len) * spec.batch;
        kr = row_fft_resource(len, rows,
                              m.row_fft_fold_d, m.row_fft_plane_k);
        break;
      }
      case LaunchKind::Twiddle:
        kr = twiddle_resource(m.stage_lengths[rec.stage]); break;
      case LaunchKind::TransposeIn:
      case LaunchKind::TransposeBoundary:
      case LaunchKind::TransposeOut:
        kr = transpose_resource(); break;
    }
    if(kr.ub_bytes > ub_have){
      r.abstract_feasible = false;
      r.supported = false;
      r.reason = std::string("UB overflow: kernel ") + kr.kernel +
                 " (" + kr.constraint + ") needs " +
                 std::to_string(kr.ub_bytes) + " bytes, have " +
                 std::to_string(ub_have);
      return r;
    }
    used.push_back(kr);
  }
  const KernelResource peak = plan_peak(used);

  r.supported = true;
  r.reason.clear();
  r.plan_ub_bytes = peak.ub_bytes;
  int edges = 0;
  for(const auto& rec : r.launch_manifest) edges += rec.boundary_edge;
  r.materialized_gm_boundaries = edges;
  r.resident_subgraph = m.residence;
  r.whole_transform_on_chip = false;
  r.host_assisted = (m.boundary_home == BoundaryHome::HostMemory && edges > 0);
  return r;
}

// Default mapping for the current G1 two-segment host-assisted chain:
// one stage service, one local-FFT role serving both stages, boundary
// materialized through host memory (the device-materialized variant of
// addendum §3 is a different BoundaryHome, not a different unit).
inline ArchitectureMapping default_long_mapping(uint32_t n1, uint32_t n2,
                                                const HardwareProfile& hw){
  ArchitectureMapping m;
  m.stage_partition = 2;
  m.stage_lengths = {n1, n2};
  m.stage_space = 1;                       // Us: single stage service today
  m.stage_time  = 2;                       // Ts: two dependent groups, sequential reuse
  m.data_space  = hw.vector_core_num;      // Ud: one concurrent tile per AIV
  m.data_time   = 1;                       // Td: one tile per worker per stage
  m.role_stages = {0, 1};                  // one role slot per stage group
  m.residence = Residence::None;
  m.boundary_layout = Layout::TwiddledStage;
  m.boundary_home = BoundaryHome::HostMemory;
  m.pipeline_buffers = 1;
  return m;
}

} // namespace butterfly
