// Legality self-test for the architecture descriptors (addendum §2 step).
// Emits machine-readable lines consumed by tests/test_descriptors.py:
//   H,<key>,<value>                      hardware profile (must match config JSON)
//   U,<core>,<min_log>,<max_log>,<ub>,<scope>,<multi_role>
//   CASE,<name>,<SUPPORTED|UNSUPPORTED>,<reason>
//   STRUCT,<name>,groups,<n>/launches/<n>/gm_boundaries/<n>/host_assisted/<0|1>/on_chip/<0|1>
// Build: g++ -std=c++17 -Iinclude tests/test_descriptors.cpp -o build/test_descriptors
#include "butterfly/descriptors.hpp"
#include <cstdio>

using namespace butterfly;

static void emit_case(const char* name, const LoweringResult& r){
  printf("CASE,%s,%s,%s\n", name, r.supported ? "SUPPORTED" : "UNSUPPORTED",
         r.reason.c_str());
}

static void emit_struct(const char* name, const LoweringResult& r){
  printf("STRUCT,%s,groups,%zu,launches,%d,gm_boundaries,%d,host_assisted,%d,on_chip,%d\n",
         name, r.execution_groups.size(), r.visible_launches,
         r.materialized_gm_boundaries, (int)r.host_assisted,
         (int)r.whole_transform_on_chip);
}

int main(){
  const HardwareProfile& hw = builtin_hardware_profile();
  printf("H,profile_id,%s\n", hw.profile_id);
  printf("H,soc,%s\n", hw.soc);
  printf("H,aicore_num,%d\n", hw.aicore_num);
  printf("H,vector_core_num,%d\n", hw.vector_core_num);
  printf("H,l2_bytes,%zu\n", hw.l2_bytes);
  printf("H,global_mem_total_bytes,%zu\n", hw.global_mem_total_bytes);
  printf("H,ub_bytes_per_core,%zu\n", hw.ub_bytes_per_core);
  printf("H,vec_lane_fp32,%d\n", hw.vec_lane_fp32);

  const ProcessingUnitCapability unit = local_fft_capability();
  printf("U,%s,%d,%d,%zu,%s,%d\n", unit.core.c_str(), unit.min_local_log_n,
         unit.max_local_log_n, unit.ub_bytes,
         sync_scope_name(unit.synchronization_scope), (int)unit.supports_multi_role);

  // --- supported: today's G1 envelope at both ends + batch independence ---
  {
    const auto m = default_long_mapping(64, 128, hw);           // n=8192 split
    emit_case("default_8192", query_lowering({8192, 1}, m, unit, hw));
    emit_struct("default_8192", query_lowering({8192, 1}, m, unit, hw));
    emit_case("default_65536_b3", query_lowering({65536, 3}, default_long_mapping(256, 256, hw), unit, hw));
    emit_case("batch_4096_independent", query_lowering({8192, 4096}, m, unit, hw));
    emit_struct("batch_4096_independent", query_lowering({8192, 4096}, m, unit, hw));
    auto cell = m;
    cell.residence = Residence::CellResident;
    emit_case("cell_resident_ok", query_lowering({8192, 1}, cell, unit, hw));
  }

  // --- unsupported: explicit reasons, one per legality rule ---
  {
    auto m = default_long_mapping(64, 128, hw);
    TransformSpec s{8192, 1};

    { auto bad = s; bad.precision = Precision::FP16;
      emit_case("precision_fp16", query_lowering(bad, m, unit, hw)); }
    { auto bad = s; bad.direction = Direction::R2C;
      emit_case("direction_r2c_unmapped", query_lowering(bad, m, unit, hw)); }
    { emit_case("bad_power_of_two", query_lowering({1000, 1}, m, unit, hw)); }
    { const auto m2 = default_long_mapping(64, 8192, hw);        // stage log2 13 > 12
      emit_case("stage_outside_unit_range", query_lowering({524288, 1}, m2, unit, hw)); }
    { auto bad = m; bad.data_space = 64;
      emit_case("ud_exceeds_aiv", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.stage_space = 96;
      emit_case("us_exceeds_aiv", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.stage_partition = 3;
      emit_case("partition_mismatch", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.role_stages = {0, 5};
      emit_case("role_out_of_range", query_lowering(s, bad, unit, hw)); }
    { const auto m2 = default_long_mapping(256, 256, hw);       // product 65536 != 8192
      emit_case("stage_product_mismatch", query_lowering(s, m2, unit, hw)); }
    { auto bad = m; bad.boundary_layout = Layout::HalfSpectrumReal;
      emit_case("layout_real_on_c2c", query_lowering(s, bad, unit, hw)); }
    { auto small = unit; small.ub_bytes = 32768;
      const auto m2 = default_long_mapping(64, 2048, hw);        // ub_need 36864 > 32768
      emit_case("ub_overflow", query_lowering({131072, 1}, m2, small, hw)); }
    { auto bad = m; bad.residence = Residence::BlockResident;
      emit_case("block_resident_needs_multi_role", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.boundary_home = BoundaryHome::OnChip;
      emit_case("onchip_without_block_residence", query_lowering(s, bad, unit, hw)); }
    { const auto m2 = default_long_mapping(256, 256, hw);
      emit_case("gm_guard_overflow", query_lowering({65536, 1u<<27}, m2, unit, hw)); }
  }
  return 0;
}
