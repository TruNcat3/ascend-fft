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
struct ExecutionGroup {
  int stage = 0;      // index into mapping.stage_lengths
  int launches = 0;   // kernel launches this group needs
};

struct LoweringResult {
  bool supported = false;
  std::string reason;                       // explicit when supported == false
  std::vector<ExecutionGroup> execution_groups;
  int visible_launches = 0;
  int materialized_gm_boundaries = 0;       // stage edges that touch GM/host memory
  Residence resident_subgraph = Residence::None;
  bool whole_transform_on_chip = false;
  bool host_assisted = false;               // boundary crosses host memory today
};

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

  uint32_t max_stage = 0;
  for(uint32_t len : m.stage_lengths) if(len > max_stage) max_stage = len;
  // UB need: data tile + twiddle tile + index/constant scratch (see G1 envelope).
  const size_t ub_need = 16ull*max_stage + 4096ull;
  const size_t ub_have = unit.ub_bytes < hw.ub_bytes_per_core ? unit.ub_bytes
                                                             : hw.ub_bytes_per_core;
  if(ub_need > ub_have)
    return reject("UB overflow: need " + std::to_string(ub_need) + " bytes, have " +
                  std::to_string(ub_have));

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

  r.supported = true;
  r.reason.clear();
  for(size_t i=0;i<m.stage_lengths.size();i++)
    r.execution_groups.push_back(ExecutionGroup{(int)i, 1});
  r.visible_launches = (int)m.stage_lengths.size();
  r.materialized_gm_boundaries = (m.boundary_home == BoundaryHome::OnChip)
                                   ? 0 : (int)m.stage_lengths.size()-1;
  r.resident_subgraph = m.residence;
  r.whole_transform_on_chip = (m.boundary_home == BoundaryHome::OnChip &&
                               r.materialized_gm_boundaries == 0);
  r.host_assisted = (m.boundary_home == BoundaryHome::HostMemory &&
                     r.materialized_gm_boundaries > 0);
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
